"""The class ramjet test runs on Modal. Each container boots from the build image,
brings its /pytorch to your working tree once, and then serves calls. It is imported
in the containers, so it uses only the standard library and modal."""

import functools
import gzip
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypedDict

import modal


ROOT = "/pytorch"
# Your changes since the build image's commit, where test.py mounts them
CHANGES = "/tmp/ramjet/changes.tar"
DELETED = "/tmp/ramjet/deleted"  # paths, NUL-separated
TAIL = 2000  # characters of a failed collect's output to return
# pytest's plugin that deselects the tests whose tci constraints a machine doesn't meet
TCI = "torch.testing._internal.tci.plugin"
# A container gives up on a file's tests, which it lists in seconds, after
# COLLECT_SECONDS, and on a piece, which is planned to take a few minutes, after
# GIVE_UP_SECONDS; Modal ends any call a little later, at CALL_SECONDS
COLLECT_SECONDS = 5 * 60
GIVE_UP_SECONDS = 30 * 60
CALL_SECONDS = GIVE_UP_SECONDS + 5 * 60
# CI's list of disabled tests, fetched as run_test.py fetches it in CI, to where
# --import-disabled-tests reads it
DISABLED = "from tools.stats import import_test_stats as stats; "
DISABLED += "stats.get_disabled_tests('test')"


class Collected(TypedDict):
    ids: list[str]  # as pytest prints them: test/test_nn.py::TestNN::test_linear
    left_out: dict[str, str]  # the tests tci leaves out here: the constraint unmet
    error: str  # the end of the output if collecting failed, else empty
    seconds: float  # how long collecting took
    container: str  # Modal's id for the container that did it
    machine: dict[str, Any]  # the container as tci reads it: its GPUs' name, count...


class Ran(TypedDict):
    piece: int  # the number test.py gave it
    code: int | None  # run_test.py's exit code; None if GIVE_UP_SECONDS ended it
    reports: dict[str, str]  # torchci's reports by file name: a JSON line per test run
    output: bytes  # what run_test.py printed, gzipped
    seconds: float  # how long the piece took
    container: str  # Modal's id for the container that ran it


def environ() -> dict[str, str]:
    """The environment of what a container runs in /pytorch."""
    # Modal's PYTHONPATH holds its own packages and the tools/ it mounts to import
    # this module, which would shadow /pytorch/tools; the image sets none
    return {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}


def machine() -> dict[str, Any]:
    """This container, as tci reads it."""
    command = [sys.executable, "-m", "torch.testing._internal.tci.machine"]
    output = {"capture_output": True, "text": True}
    done = subprocess.run(command, cwd=f"{ROOT}/test", env=environ(), **output)
    if done.returncode != 0:
        raise RuntimeError(f"can't read the machine:\n{done.stderr[-TAIL:]}")
    return json.loads(done.stdout.splitlines()[-1])


class Worker:
    machine: Future[dict[str, Any]]  # being read since sync

    @modal.enter()
    def sync(self) -> None:
        """Once per container, what run.sync_files does to a runner: rm for the files
        deleted, then a tar of those changed."""
        for path in filter(None, Path(DELETED).read_text().split("\0")):
            Path(ROOT, path).unlink(missing_ok=True)
        if Path(CHANGES).stat().st_size:  # empty when no file changed
            subprocess.run(["tar", "-xf", CHANGES], cwd=ROOT, check=True)
        # Read while the first files are collected: it takes as long as importing torch
        self.machine = ThreadPoolExecutor(1).submit(machine)

    @modal.method()
    def collect(self, names: list[str], pytest_args: list[str]) -> list[Collected]:
        """collect_file's answer for each test file named. The files are collected at
        once: a call names one per core."""
        with ThreadPoolExecutor(len(names)) as pool:
            one = functools.partial(self.collect_file, pytest_args=pytest_args)
            return list(pool.map(one, names))

    def collect_file(self, name: str, pytest_args: list[str]) -> Collected:
        """The tests run_test.py would run here from one of its test files, without
        those that tci leaves out on this machine."""
        # Launched as run_test.py launches it, so the file's __main__ guard decides
        # here too: pytest alone collects files that guard never lets run
        launch = [sys.executable, "-bb", f"{name}.py", "--use-pytest", "--collect-only"]
        where = {"cwd": f"{ROOT}/test", "env": environ(), "timeout": COLLECT_SECONDS}
        output = {"capture_output": True, "text": True, "errors": "replace"}
        start = time.monotonic()
        with tempfile.NamedTemporaryFile() as deselected:
            plugin = ["-p", TCI, f"--tci-deselected={deselected.name}"]
            # Last, so a -v among your arguments can't turn the ids into a tree
            command = [*launch, *plugin, *pytest_args, "--verbosity=-1"]
            try:
                done = subprocess.run(command, **where, **output)
            except subprocess.TimeoutExpired:
                slow = f"still collecting after {COLLECT_SECONDS} s"
                done = subprocess.CompletedProcess(command, 1, "", slow)
            left_out = deselected.read()  # nothing, if the file started no pytest
        failed = done.returncode != 0
        prefix = f"test/{name}.py::"  # the other lines are the file's own output
        lines = [] if failed else done.stdout.splitlines()
        return {
            "ids": [line for line in lines if line.startswith(prefix)],
            "left_out": json.loads(left_out) if left_out and not failed else {},
            "error": (done.stderr + done.stdout)[-TAIL:] if failed else "",
            "seconds": time.monotonic() - start,
            "container": os.environ.get("MODAL_TASK_ID", ""),
            "machine": self.machine.result(),
        }

    @modal.method()
    def run(self, piece: int, tests: dict[str, list[str]], args: list[str]) -> Ran:
        """Run a piece: some tests, by node id, of some of run_test.py's test files.
        run_test.py runs them as CI does: a few files at once, a failure again in a
        new process, and without the tests on CI's slow and disabled lists."""
        python, env, start = sys.executable, environ(), time.monotonic()
        with tempfile.TemporaryDirectory() as scratch:
            ids, reports = Path(scratch, "ids"), Path(scratch, "reports")
            ids.write_text("\n".join(i for run in tests.values() for i in run))
            launch = [python, "test/run_test.py", "--keep-going", "--pipe-logs"]
            launch += [f"--save-torchci-reports={reports}", "-i", *tests, "--"]
            # What run_test.py hands each of its test files, which all read the ids
            launch += [f"--nodeid-file={ids}", "--import-slow-tests"]
            launch += ["--import-disabled-tests", *args]
            with open(Path(scratch, "output"), "w+b") as printed:
                output = {"stdout": printed, "stderr": subprocess.STDOUT}
                subprocess.run([python, "-c", DISABLED], cwd=ROOT, env=env, **output)
                # A session of its own, so that giving up ends its test processes too
                session = {"start_new_session": True, **output}
                process = subprocess.Popen(launch, cwd=ROOT, env=env, **session)
                try:
                    code = process.wait(timeout=GIVE_UP_SECONDS)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    code = None
                printed.seek(0)
                said = gzip.compress(printed.read())
            found = sorted(reports.glob("*.jsonl"))
            return {
                "piece": piece,
                "code": code,
                "reports": {path.name: path.read_text() for path in found},
                "output": said,
                "seconds": time.monotonic() - start,
                "container": os.environ.get("MODAL_TASK_ID", ""),
            }
