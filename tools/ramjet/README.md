# ramjet

ramjet builds your worktree on [Modal](https://modal.com) and runs a command, such as pytest, on a runner with the GPUs you name, so you can test on GPUs your machine does not have:

```
python -m tools.ramjet --gpu H100 -- pytest test/test_nn.py -k linear
```

The runner, the sandbox your command runs in, takes `modal shell`'s options. `--gpu` takes GPUs as Modal names them, in any case, with `:N` for N of them: `L4`, `A100-80GB`, `H100`, `B200:8`; without it the runner has no GPU. `--cpu` sets its physical cores, 1 to 64 in powers of two, and `--memory` its memory in MiB, 8192 to 344064; by default it gets 8 cores per GPU, or 8 without one, and 8192 MiB per core. Modal has no CPU types, so the CPU model and its instruction set can't be chosen. The builder is always the same: 64 cores and no GPU. `--help` lists every choice. `--build-cuda` picks the CUDA version to build with: any that CI builds an image for and that has an official arch list, 12.6, 13.2 and 13.4 today. The default is the version of PyTorch's default wheel. `--python-version` picks the Python version, the same in the builder and the runner, which boots from the builder's image: any PyTorch ships wheels for, 3.11 to 3.15 with the free-threaded `t` builds. The default is the one in CI's image, 3.11 today. `--log-level` sets the least severe messages ramjet shows: `debug`, `info` (the default), `warn` or `error`. Everything after `--` is the command, run as given in the directory you run ramjet from: `pytest test/test_nn.py -k linear`, `nvidia-smi`, or `bash -c '...'` for pipes and `&&`.

## Running tests

`test` runs the tests under the paths you name on as many containers as Modal gives you, each with the GPUs you name:

```
python -m tools.ramjet --gpu L4 test test/dynamo
```

ramjet collects the tests and cuts them into pieces of about 150 seconds, as `test --list` shows them (below), and each piece runs in a container of its own: the containers that collected take the first pieces, and Modal starts more for the rest. The container runs it through `run_test.py`, as CI does: a few test files at once, a failed test again in the same process and then in new ones, and without the tests on CI's slow and disabled lists. Arguments after `--` go to every test file as well as to the collection; `-m` doesn't work there, since `run_test.py` sets its own. Every 15 seconds ramjet logs how many pieces are done, how many are running and how many wait for a container. As a piece finishes, each of its tests that failed is logged with the first line of its error; a test that failed and then passed when run again is flaky, which is a warning. A container gives up on a piece after 30 minutes, and the tests it didn't get to count as not run. At the end ramjet logs how many tests passed, were skipped and failed, the containers' time with about what it cost, and the directory that holds each piece's output and its tests' reports (`torch/testing/_internal/torchci/README.md`). stdout gets a line of JSON for each test that failed or didn't run, such as `{"nodeid": "test/dynamo/test_misc.py::MiscTests::test_cast", "outcome": "failed"}`, and the exit code is 1 if there is one, or if a file couldn't be collected. To work on a failure, run it alone on the warm runner with `--`; ramjet logs that command for the first one.

## Listing tests

`test --list` prints the tests under the paths you name that the GPU you name can run, each with the piece it would run in, one line of JSON each, without running them:

```
python -m tools.ramjet --gpu L4 test --list test/dynamo -- -k guard
```

The test files are collected in parallel: Modal spreads them over as many containers as it gives you, and a container collects as many at once as it has cores. Meanwhile ramjet logs every 15 seconds how many files have come back and from how many containers, or how long Modal has given it none: Modal can run out of a GPU for minutes. Each file is launched the way `run_test.py` launches it, so the list is what would run from that file on that GPU. A test can say which machines it runs on with the decorators in `torch/testing/_internal/tci`, such as `@tci.only(tci.cuda)` or `@tci.needs(tci.gpus >= 2)`. The container leaves out the tests whose constraints it doesn't meet, and ramjet logs how many, by constraint; `--log-level debug` names each one. A path is a test file, a directory of them, or a node id such as `test/test_nn.py::TestNN`, among the test files `run_test.py` runs; for others, use `-- pytest`. Arguments after `--` go to pytest's collection. A container gets 4 cores and 16384 MiB per GPU, or as much without one, unless `--cpu` and `--memory` say otherwise, and bills as a Modal Function, at a third of a sandbox's rates for cores and memory. The tests are then cut into pieces of about 150 seconds, one container's work each: a file that takes longer is cut into runs of consecutive tests, and short files share a piece. Each line of the list has its test's `piece`, and the log says how many pieces there are and how much container time they add up to. A piece is sized by CI's time for each of its tests over the last 7 days, on machines with the same GPUs as the container or, for a test those don't run, on any machine; a test CI has no time for counts as the median of its file, and the log says how many were guessed. The times come from CI's database through the ci-metrics skill's wrapper (`.agents/skills/ci-metrics`), which needs a `gh` login that HUD accepts. Without one ramjet warns and sizes the pieces as if all tests took the same time. If a file can't be collected, ramjet shows the end of its output and exits 1. `--log-level debug` shows how long each file took, which container collected it, how many tests CI timed on the same GPUs and what is in each piece.

## Setup

```
pip install modal
modal token new
```

## How it works

1. Your working tree, edits and untracked files included, becomes a commit. HEAD, the index and the stash are left alone.
2. If the build reads none of the files that changed since the last build, there is no build. Otherwise a 64-core builder boots from the last build image, fetches a git bundle of the new commits, rebuilds incrementally and saves itself as the next build image. The first build clones pytorch/pytorch into CI's CUDA image and takes 1-1.5 hours when the compile cache has none of its compiles. Builders share that cache: ccache looks up each compile in the build image's own cache, then in the `ramjet-ccache` Modal Volume, which every builder in your Modal workspace reads and adds to. Each build deletes the Volume's entries written more than 14 days ago.
3. A runner of the size you ask for boots from the build image, or the warm one from your last run is reused. It gets a tar of the files that changed and runs your command.

## Costs

The builder bills about $13 an hour, so a first build costs $13-20 and an incremental one much less. A runner bills its GPUs plus its cores and memory: 8 cores and 64 GiB per GPU unless `--cpu` and `--memory` say otherwise. After a run it stays warm for 15 minutes, or until idling has cost $2.50, whichever comes first (`IDLE_MAX_SECONDS` and `IDLE_BUDGET` in `run.py`). `test`'s containers bill at a Function's rates, about $1.12 an hour for an L4 with 4 cores and 16 GiB, while they collect or run; in a run, one that finds no piece to take stops after 15 seconds. Before it runs the pieces, `test` logs about what they will cost by CI's times; a test that fails is run up to 9 times, so a change that breaks many tests costs more. A full build adds about 1.2 GB to the compile cache's Volume; Modal bills Volume storage past 1 TiB, at $0.09/GiB a month.

## Files

ramjet keeps its state in `/tmp/ramjet-<user>/<worktree folder>-<hash of its path>`, or under `$TMPDIR` if you set it: the build images per build config (`lineage.json`), the last build's log (`build.log`), a warm runner per runner size, and a `test-<time>` directory for each run of `test`. Build images expire after 5 days; ramjet rebuilds images older than 4 days so a lineage outlives them. A machine that clears `/tmp` of files not written for some days would take the lineage with it, so each run refreshes the timestamps of the lineage files it uses.

## Limits

- Builds use the official arch list for their CUDA version. Before building, ramjet stops if that list, or your `TORCH_CUDA_ARCH_LIST`, can't run on `--gpu`: 12.6's stops at sm_90, so it can't run on Blackwell GPUs such as B200, B300 or RTX PRO 6000. `--build-cuda` and your `BUILD_*`, `USE_*`, `CMAKE_BUILD_TYPE`, `DEBUG`, `REL_WITH_DEB_INFO` and `TORCH_CUDA_ARCH_LIST` variables change the build; each combination gets its own lineage.
- CI's CUDA images come with one Python. ramjet installs another `--python-version` the way CI's Dockerfile installs Python (`install_python.sh` into a venv first on `PATH`), as its own lineage. CI runs CUDA tests only on 3.11 and 3.12, so other versions with CUDA are untested there.
- A lineage stays on the version of CI's image it started on, so pulling a `.ci/docker` change doesn't restart it. To move to the current version, delete `lineage.json`.
- The command gets no terminal and no input, so interactive programs such as a bare `bash` or `python` don't work.
- Your whole Modal workspace shares the compile cache: a build links whatever objects other builders put in it.
