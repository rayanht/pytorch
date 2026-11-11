"""ramjet test: run the tests under the paths you name on the GPUs you name, or list
them. The work is calls to a Modal Function, which Modal spreads over as many
containers as it gives you, each taking one call at a time. The first calls each name
as many test files as a container has cores: the container collects their tests and
leaves out those whose tci constraints (torch/testing/_internal/tci) its machine
doesn't meet. The rest are cut into pieces of about PIECE_SECONDS by CI's time for
each test, and each piece is one more call, which a container runs through
run_test.py: the containers that collected take the first pieces."""

import contextlib
import gzip
import json
import math
import statistics
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import modal

from tools.ramjet import sync, times, worker
from tools.ramjet.common import repo_root
from tools.ramjet.lineage import Build
from tools.ramjet.logger import log
from tools.ramjet.run import Resources
from tools.testing.discover_tests import TESTS


# A container's resources unless --cpu and --memory say otherwise: CORES_PER_GPU per
# GPU, or as many without one, with MIB_PER_CORE of memory each
CORES_PER_GPU = 4
MIB_PER_CORE = 4096
# How long a container waits for a call before it stops. When listing, the least
# Modal allows. When running, long enough for the containers that collected to get
# the pieces, which are cut once CI has given the tests' times
SCALEDOWN_SECONDS = 2
HANDOVER_SECONDS = 15
PROGRESS_SECONDS = 15  # between the lines that say how far the containers are
PIECE_SECONDS = 150
# A container's seconds per second of CI, and its seconds to start a test file: fitted
# on L4 containers running test/dynamo
SLOWDOWN = 2.28
START_SECONDS = 5.4
# CI's mean for a test, for the tests of a file it has timed none of
GUESS_SECONDS = 0.3
# A test's outcomes, as torchci's reports name them, that fail a run, and the order
# the summary counts every outcome in: flaky and not_run are ramjet's own
FAILED = ("failed", "error", "crashed", "timed_out", "xpassed")
OUTCOMES = ("passed", "flaky", "skipped", "xfailed", *FAILED, "not_run")
# What PyTorch says before the command that reruns a failed test
REPRO = "To execute this test, run the following from the base repo dir:"


@dataclass
class Piece:
    """What one container runs: tests of some files, each file started once."""

    seconds: float = 0.0  # what it should take
    tests: dict[str, list[str]] = field(default_factory=dict)  # node ids, by their file


def count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def names(paths: list[str]) -> dict[str, list[str]]:
    """run_test.py's name for each test file the paths cover, with what the paths
    ask of it: the file, for all its tests, or node ids in it. A path is a test
    file, a directory of them or a node id, from the directory you are in."""
    test_dir = repo_root() / "test"
    asked: dict[str, list[str]] = {}
    for path in paths:
        file, _, node = path.partition("::")
        target = Path(file).resolve()
        found: list[str] = []
        if target.is_dir() and not node and target.is_relative_to(test_dir):
            under = "" if target == test_dir else f"{target.relative_to(test_dir)}/"
            # TESTS also names C++ binaries and tests that have no file of their own
            found = [name for name in TESTS if name.startswith(under)]
            found = [name for name in found if (test_dir / f"{name}.py").is_file()]
        elif target.is_file() and target.is_relative_to(test_dir):
            name = target.relative_to(test_dir).with_suffix("").as_posix()
            found = [name] if name in TESTS else []
        if not found:
            log.error("%s is not a test file or directory that run_test.py runs", path)
            log.error("with pytest instead: python -m tools.ramjet -- pytest %s", path)
            sys.exit(1)
        for name in found:
            file_id = f"test/{name}.py"
            asked.setdefault(name, []).append(f"{file_id}::{node}" if node else file_id)
    return asked


def waiting(containers: int, on: str, seconds: float) -> str:
    """What to say while no call has come back: how many containers Modal has
    started for them, or how long it has given none."""
    if containers:
        verb = "is" if containers == 1 else "are"
        return f"{count(containers, 'container')} {verb} starting"
    waited = f"{seconds:.0f} s" if seconds < 60 else f"{seconds // 60:.0f} min"
    return f"Modal has given no {on} in {waited}"


@contextlib.contextmanager
def progress(line: Callable[[float], str]) -> Iterator[None]:
    """Until the block ends, log every PROGRESS_SECONDS what line makes of the
    seconds since it began."""
    done, start = threading.Event(), time.monotonic()

    def report() -> None:
        while not done.wait(PROGRESS_SECONDS):
            try:
                log.info("  %s", line(time.monotonic() - start))
            except Exception as error:  # Modal didn't say: the next line may
                log.debug("no progress line: %r", error)

    reporter = threading.Thread(target=report, name="ramjet-progress", daemon=True)
    reporter.start()
    try:
        yield
    finally:
        done.set()
        reporter.join()


@contextlib.contextmanager
def fleet(
    image: Build, commit: str, resources: Resources, ramjet_dir: Path, idle: int
) -> Iterator[modal.Cls]:
    """For the block, the class whose containers boot from the build image and take
    your changes since its commit. A container takes one call at a time and stops
    once it has had none for idle seconds."""
    changes = sync.diff(image.commit, commit)
    changed = [path for path, status in changes.items() if status != "D"]
    deleted = [path for path, status in changes.items() if status == "D"]
    with (
        tempfile.NamedTemporaryFile(dir=ramjet_dir, suffix=".tar") as tar,
        tempfile.NamedTemporaryFile("w", dir=ramjet_dir) as gone,
    ):
        if changed:  # given no paths, git archives the whole tree
            sync.archive(commit, changed, Path(tar.name))
        gone.write("\0".join(deleted))
        gone.flush()
        files = (
            modal.Image.from_id(image.image_id)
            # The layer Modal's builder adds: on the build image alone, which is a
            # sandbox's snapshot, a Function's container can't start
            .run_commands("true")
            # Mounted as each container starts, so runs share the image
            .add_local_file(tar.name, worker.CHANGES)
            .add_local_file(gone.name, worker.DELETED)
        )
        app = modal.App("ramjet-test")
        # A piece runs for longer than Modal lets a call by default
        limits = {"scaledown_window": idle, "timeout": worker.CALL_SECONDS}
        workers = app.cls(image=files, **limits, **resources._asdict())(worker.Worker)
        with app.run():  # its containers stop when ramjet does, Ctrl-C included
            yield workers


def collect(
    workers: modal.Cls, resources: Resources, names: list[str], pytest_args: list[str]
) -> Iterator[worker.Collected | Exception]:
    """Each test file's tests, in the order of names and as each call comes back. A
    call that Modal lost comes back as its exception, once for each of its files."""
    method = workers().collect
    # Collecting a file is one process, so a call names one file per core
    cores = resources.cpu
    calls = [names[i : i + cores] for i in range(0, len(names), cores)]
    back: list[worker.Collected | Exception] = []

    def line(seconds: float) -> str:
        if not back:
            containers = method.get_current_stats().num_total_runners
            on = resources.gpu or "container"
            return f"no file collected yet: {waiting(containers, on, seconds)}"
        ids = {r["container"] for r in back if not isinstance(r, Exception)}
        sizes = count(len(names), "file"), count(len(ids), "container")
        return "{} of {} collected by {}".format(len(back), *sizes)

    with progress(line):
        kwargs = {"pytest_args": pytest_args}
        results = method.map(calls, kwargs=kwargs, return_exceptions=True)
        for call, result in zip(calls, results, strict=True):
            files = [result] * len(call) if isinstance(result, Exception) else result
            back.extend(files)
            yield from files


def timed(
    kept: dict[str, list[str]], machine: dict[str, Any]
) -> dict[str, list[tuple[str, float]]]:
    """Each test with the seconds it should take in a container: SLOWDOWN times CI's
    time for it on machines with the same GPUs, else on any machine, else the median
    of its file's tests."""
    files = {f"test/{name}.py": len(ids) for name, ids in kept.items()}
    device, gpus = machine["device"], machine["gpus"]
    ci = times.fetch(files, device, gpus)
    same = guessed = 0
    estimates = {}
    for name, ids in kept.items():
        seconds = []  # CI's for each test, or None for one it has never run
        for nodeid in ids:
            here, anywhere = ci.get(nodeid, (None, None))
            seconds.append(anywhere if here is None else here)
            same += here is not None
        known = [s for s in seconds if s is not None]
        typical = statistics.median(known) if known else GUESS_SECONDS
        guessed += len(ids) - len(known)
        scaled = [SLOWDOWN * (typical if s is None else s) for s in seconds]
        estimates[name] = list(zip(ids, scaled))
    others = sum(files.values()) - same - guessed
    on = f"{gpus} x {device}" if gpus else "no GPU"
    sources = count(same, "test"), on, others, guessed
    log.debug("CI's times: %s with %s, %d on other machines, %d guessed", *sources)
    if guessed:
        log.info("guessed the time of %s that CI has none for", count(guessed, "test"))
    return estimates


def shard(timed: dict[str, list[tuple[str, float]]]) -> list[Piece]:
    """Pieces of about PIECE_SECONDS, the longest first. A file that takes longer is
    cut into runs of consecutive tests, and shorter ones share a piece: one run of a
    file per piece, so that a piece starts each of its files once."""
    runs: list[tuple[float, str, list[str]]] = []
    for name, tests in timed.items():
        seconds, ids = START_SECONDS, []
        for nodeid, estimate in tests:
            if ids and seconds + estimate > PIECE_SECONDS:
                runs.append((seconds, name, ids))
                seconds, ids = START_SECONDS, []
            seconds += estimate
            ids.append(nodeid)
        runs.append((seconds, name, ids))
    pieces: list[Piece] = []
    for seconds, name, ids in sorted(runs, key=lambda run: -run[0]):
        room = PIECE_SECONDS - seconds
        fits = (p for p in pieces if p.seconds <= room and name not in p.tests)
        piece = next(fits, None)  # the first with room, of the longest so far
        if piece is None:
            piece = Piece()
            pieces.append(piece)
        piece.seconds += seconds
        piece.tests[name] = ids
    return sorted(pieces, key=lambda piece: -piece.seconds)


def gather(
    workers: modal.Cls,
    resources: Resources,
    names: dict[str, list[str]],
    pytest_args: list[str],
) -> tuple[dict[str, list[str]], Counter[str], int]:
    """The tests asked for that these containers can run, by file; the containers,
    counted by what tci read of them; and how many files couldn't be collected."""
    on = resources.gpu or "a CPU container"
    log.info("collecting %s on %s", count(len(names), "file"), on)
    results = collect(workers, resources, list(names), pytest_args)
    failed = 0
    kept: dict[str, list[str]] = {}
    left_out: Counter[str] = Counter()  # tests, by the tci constraint that fails
    machines: Counter[str] = Counter()
    # strict, so that collect runs to its end, where its progress lines stop
    for (name, asked), result in zip(names.items(), results, strict=True):
        if isinstance(result, Exception):  # Modal lost the call
            error = repr(result)
        else:
            took = result["seconds"], result["container"]
            log.debug("test/%s.py took %.1f s on container %s", name, *took)
            error = result["error"].rstrip()
            machines[json.dumps(result["machine"])] += 1
            # A node id also asks for the tests under it and for its parametrizations
            under = tuple(f"{a}{mark}" for a in asked for mark in ("::", "["))
            # A test that can run here has None; one left out, its constraint
            verdicts = {**dict.fromkeys(result["ids"]), **result["left_out"]}
            for nodeid, unmet in verdicts.items():
                if nodeid not in asked and not nodeid.startswith(under):
                    continue
                if unmet:
                    log.debug("left out %s: %s", nodeid, unmet)
                    left_out[unmet] += 1
                else:
                    kept.setdefault(name, []).append(nodeid)
        if error:
            log.error("collecting test/%s.py failed:\n%s", name, error)
            failed += 1
    files = count(len(names) - failed, "file")
    log.info("found %s in %s", count(sum(map(len, kept.values())), "test"), files)
    if left_out:
        tests = count(left_out.total(), "test")
        log.info("left out %s that %s can't run:", tests, on)
        for unmet, n in left_out.most_common():
            log.info("  %d with %s", n, unmet)
    return kept, machines, failed


def plan(kept: dict[str, list[str]], machines: Counter[str]) -> list[Piece]:
    """The pieces to run the tests in, sized for the machine most containers are."""
    if len(machines) > 1:  # Modal may serve an H100 with an H200
        log.warning("containers differ: %s", ", ".join(machines))
    pieces = shard(timed(kept, json.loads(machines.most_common(1)[0][0])))
    for number, piece in enumerate(pieces, 1):
        tests = count(sum(map(len, piece.tests.values())), "test")
        sizes = tests, count(len(piece.tests), "file")
        log.debug("piece %d, about %.0f s: %s of %s", number, piece.seconds, *sizes)
    minutes = math.ceil(sum(piece.seconds for piece in pieces) / 60)
    sizes = count(len(pieces), "piece"), count(minutes, "container-minute")
    log.info("%s: %s, the longest about %.0f s", *sizes, pieces[0].seconds)
    return pieces


def run(
    workers: modal.Cls, on: str, pieces: list[Piece], pytest_args: list[str]
) -> Iterator[worker.Ran | Exception]:
    """What each piece's container returns, as pieces finish. A call that Modal lost
    comes back as its exception."""
    method = workers().run
    back: list[worker.Ran | Exception] = []

    def line(seconds: float) -> str:
        stats = method.get_current_stats()
        if not back and not stats.num_total_runners:
            return f"no piece done yet: {waiting(0, on, seconds)}"
        done = len(back), count(len(pieces), "piece")
        rest = stats.num_running_inputs, stats.backlog
        return "{} of {} done, {} running, {} waiting".format(*done, *rest)

    with progress(line):
        calls = [(number, piece.tests) for number, piece in enumerate(pieces, 1)]
        options = {"kwargs": {"args": pytest_args}, "return_exceptions": True}
        for result in method.starmap(calls, order_outputs=False, **options):
            back.append(result)
            yield result


def outcomes(reports: dict[str, str], sent: list[str]) -> dict[str, tuple[str, str]]:
    """Each test sent to a piece, with its outcome and what its last failing run
    said, from torchci's reports: the outcome of its last run, flaky if that one
    passed after one that failed, and not_run without any."""
    keys = {}  # a report names a test by its file, innermost class and name
    for nodeid in sent:
        path, bracket, parameters = nodeid.partition("[")
        file, *classes, case = path.split("::")
        suite = classes[-1] if classes else ""
        keys[file, suite, f"{case}{bracket}{parameters}"] = nodeid
    runs: dict[str, list[tuple[int, int, str, str]]] = {}
    for report in reports.values():
        for line in report.split("\n"):  # splitlines also breaks inside strings
            try:
                run = json.loads(line)
            except ValueError:  # empty, or torn as its process died
                continue
            key = run.get("file"), run.get("suite"), run.get("case_name")
            if run["type"] == "run" and key in keys:
                when = run["started_at"], run["rerun_number"]
                attempt = *when, run["outcome"], run["outcome_summary"]
                runs.setdefault(keys[key], []).append(attempt)
    verdicts = {}
    for nodeid in sent:
        attempts = sorted(runs.get(nodeid, []))
        failures = [said for *_, outcome, said in attempts if outcome in FAILED]
        outcome = attempts[-1][2] if attempts else "not_run"
        if outcome == "passed" and failures:
            outcome = "flaky"
        said = failures[-1] if failures else attempts[-1][3] if attempts else ""
        verdicts[nodeid] = outcome, said
    return verdicts


def execute(
    workers: modal.Cls,
    resources: Resources,
    pieces: list[Piece],
    pytest_args: list[str],
    directory: Path,
    rate: Decimal,
    rerun: str,
) -> int:
    """Run the pieces, logging each failure as its piece finishes and the totals at
    the end, with every piece's output and reports kept in the directory. Print each
    test that failed or didn't run, a JSON object per line, and return how many.
    rerun is the ramjet command that runs what follows it on the warm runner."""
    on = resources.gpu or "a CPU container"
    cost = rate * Decimal(sum(piece.seconds for piece in pieces)) / 3600
    log.info("running %s on %s, about $%.2f", count(len(pieces), "piece"), on, cost)
    (directory / "reports").mkdir(parents=True)
    start = time.monotonic()
    seconds, containers = 0.0, set()
    verdicts: dict[str, tuple[str, str]] = {}
    for ran in run(workers, resources.gpu or "container", pieces, pytest_args):
        if isinstance(ran, Exception):  # which doesn't say whose piece it was
            log.error("Modal lost a piece: %r", ran)
            continue
        number, code = ran["piece"], ran["code"]
        ended = number, ran["seconds"], ran["container"], code
        log.debug("piece %d took %.0f s on container %s: exit code %s", *ended)
        seconds += ran["seconds"]
        containers.add(ran["container"])
        output = directory / f"piece-{number}.log"
        output.write_bytes(gzip.decompress(ran["output"]))
        for name, report in ran["reports"].items():
            (directory / "reports" / name).write_text(report)
        sent = [i for ids in pieces[number - 1].tests.values() for i in ids]
        found = outcomes(ran["reports"], sent)
        verdicts.update(found)
        for nodeid, (outcome, said) in found.items():
            if outcome in FAILED:
                log.error("failed: %s", nodeid)
                log.error("  %s", said.partition("\n")[0])
            elif outcome == "flaky":
                log.warning("flaky: %s", nodeid)
        if missing := sum(outcome == "not_run" for outcome, _ in found.values()):
            minutes = worker.GIVE_UP_SECONDS // 60
            how = f"stopped after {minutes} min" if code is None else f"exited {code}"
            lost = number, how, count(missing, "test"), output
            log.error("piece %d %s: %s didn't run, see %s", *lost)
    failing = []
    for piece in pieces:
        for nodeid in (i for ids in piece.tests.values() for i in ids):
            outcome, said = verdicts.setdefault(nodeid, ("not_run", ""))
            if outcome in FAILED or outcome == "not_run":
                failing.append((nodeid, outcome, said))
    tally = Counter(outcome for outcome, _ in verdicts.values())
    counts = [f"{tally[o]} {o.replace('_', ' ')}" for o in OUTCOMES if tally[o]]
    log.info("%s: %s", count(len(verdicts), "test"), ", ".join(counts))
    wall = "{} min {} s".format(*divmod(int(time.monotonic() - start), 60))
    used = count(math.ceil(seconds / 60), "container-minute")
    spent = rate * Decimal(seconds) / 3600
    fleet_size = count(len(containers), "container")
    log.info("%s on %s: %s, about $%.2f", wall, fleet_size, used, spent)
    log.info("logs and reports in %s", directory)
    for nodeid, outcome, _ in failing:
        print(json.dumps({"nodeid": nodeid, "outcome": outcome}))
    # PyTorch ends a failure's message with the command that runs the test alone
    alone = [said.partition(REPRO)[2] for *_, said in failing if REPRO in said]
    if alone:
        again = rerun, alone[0].strip().partition("\n")[0]
        log.info("rerun one on the warm runner: %s %s", *again)
    return len(failing)


def run_tests(
    image: Build,
    commit: str,
    resources: Resources,
    names: dict[str, list[str]],
    pytest_args: list[str],
    ramjet_dir: Path,
    listing: bool,
    rate: Decimal,
    rerun: str,
) -> int:
    """Run the tests asked for that these containers can run, or print each with its
    piece, one JSON object per line. Return the exit code: 1 if a test failed or
    didn't run, or if a file's tests couldn't be collected. rate is what a container
    bills an hour, and rerun the ramjet command for the warm runner."""
    idle = SCALEDOWN_SECONDS if listing else HANDOVER_SECONDS
    with fleet(image, commit, resources, ramjet_dir, idle) as workers:
        kept, machines, failed = gather(workers, resources, names, pytest_args)
        if not kept:
            return 1 if failed else 0
        pieces = plan(kept, machines)
        if listing:
            piece_of: dict[str, int] = {}
            for number, piece in enumerate(pieces, 1):
                for ids in piece.tests.values():
                    piece_of.update(dict.fromkeys(ids, number))
            for ids in kept.values():
                for nodeid in ids:
                    print(json.dumps({"nodeid": nodeid, "piece": piece_of[nodeid]}))
        else:
            directory = ramjet_dir / time.strftime("test-%Y%m%d-%H%M%S")
            work = (workers, resources, pieces, pytest_args, directory)
            failed += execute(*work, rate, rerun)
    return 1 if failed else 0
