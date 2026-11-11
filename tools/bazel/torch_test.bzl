"""Tests of one PyTorch test file: a target for each of its cases.

The cases of a file are known only once it is imported, with torch built, so
they cannot be worked out while BUILD files load. They are recorded instead:
`bazel run //test:<name>.list` lists them and writes them to .test_cases/, which
the next Bazel command reads back (see extensions.bzl). The same target, as a
test, fails when the record is not what the file has now.

A case runs in a build action, and its test only reports what that found. Unlike
a test, a build action can tell Bazel which of its inputs it did not use: here,
the torch Python files that no process of the case opened. Bazel then keeps the
result until one of the others changes.
"""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@rules_python//python:py_binary.bzl", "py_binary")
load("//tools/bazel:extensions.bzl", "RECORDS")
load("//tools/bazel:torch_tree.bzl", "TorchTreeInfo")

# What a process that imports torch may use.
CPUS = 4

TorchCasesInfo = provider(
    doc = "The cases a test file has now.",
    fields = {
        "listed": "File: a line for each case, its target's name and its pytest id.",
        "test": "string: the test, as test/run_test.py names it.",
    },
)

def _cpus(_os, _input_count):
    return {"cpu": CPUS}

def _run(ctx, mode, outputs, case = [], env = {}, **kwargs):
    """Runs the file's program (run_test.py) with torch as inputs it may leave unused."""
    tree = ctx.attr._tree[TorchTreeInfo]
    unused = ctx.actions.declare_file(ctx.label.name + ".unused")
    args = ctx.actions.args()
    args.add_all([mode, ctx.attr.file] + case)
    args.add_all([tree.python] + outputs + [unused])
    ctx.actions.run(
        executable = ctx.attr.runner[DefaultInfo].files_to_run,
        arguments = [args],
        inputs = depset([tree.python], transitive = [tree.files]),
        outputs = outputs + [unused],
        unused_inputs_list = unused,
        env = env | {"MKL_NUM_THREADS": str(CPUS), "OMP_NUM_THREADS": str(CPUS)},
        use_default_shell_env = True,
        resource_set = _cpus,
        **kwargs
    )

_LIST = """#!/bin/bash
# bazel run records the cases in the workspace. bazel test checks the record.
if [ -n "$BUILD_WORKSPACE_DIRECTORY" ]; then
    record="$BUILD_WORKSPACE_DIRECTORY/{records}/{test}"
    mkdir -p "$(dirname "$record")" && cat "{listed}" > "$record" || exit 1
    echo "Recorded the $(wc -l < "$record") cases of {test}."
    exit
fi
cmp -s "{recorded}" "{listed}" && exit
echo "{test} does not have the cases that are recorded for it. To record them:"
echo "    bazel run {label}"
diff "{recorded}" "{listed}" | grep '^[<>]' | head -20
exit 1
"""

def _torch_list_impl(ctx):
    test = ctx.attr.file.removesuffix(".py")
    listed = ctx.actions.declare_file(ctx.label.name + ".cases")
    _run(
        ctx,
        "--list",
        [listed],
        mnemonic = "TorchTestList",
        progress_message = "Listing the cases of test/" + ctx.attr.file,
    )
    recorded = ctx.actions.declare_file(ctx.label.name + ".recorded")
    ctx.actions.write(recorded, "".join(["%s\t%s\n" % case for case in ctx.attr.cases.items()]))
    script = ctx.actions.declare_file(ctx.label.name)
    label = "//%s:%s" % (ctx.label.package, ctx.label.name)
    paths = {"listed": listed.short_path, "recorded": recorded.short_path}
    ctx.actions.write(script, _LIST.format(label = label, records = RECORDS, test = test, **paths), is_executable = True)
    return [
        DefaultInfo(executable = script, runfiles = ctx.runfiles([listed, recorded])),
        TorchCasesInfo(listed = listed, test = test),
    ]

_REPORT = """#!/bin/bash
# Reports what the action that ran the case found.
cp "{xml}" "$XML_OUTPUT_FILE"
status=$(< "{status}")
[ "$status" = 0 ] || cat "{log}"
exit "$status"
"""

def _torch_case_impl(ctx):
    results = {kind: ctx.actions.declare_file("%s.%s" % (ctx.label.name, kind)) for kind in ("xml", "log", "status")}
    _run(
        ctx,
        "--run",
        results.values(),
        case = [ctx.attr.case],
        env = {
            # pytest takes more options from this variable.
            "PYTEST_ADDOPTS": ctx.attr._pytest_args[BuildSettingInfo].value,
            # A result is a build output, so Bazel keeps it until an input changes,
            # even if the case failed. Changing this variable runs the case again.
            "RERUN": ctx.attr._rerun[BuildSettingInfo].value,
        },
        # Bounded by --local_resources=gpu in .bazelrc.
        execution_requirements = {"resources:gpu:1": ""},
        mnemonic = "TorchTestRun",
        progress_message = "Testing %{label}",
    )
    script = ctx.actions.declare_file(ctx.label.name)
    paths = {kind: result.short_path for kind, result in results.items()}
    ctx.actions.write(script, _REPORT.format(**paths), is_executable = True)
    return [DefaultInfo(executable = script, runfiles = ctx.runfiles(results.values()))]

_ATTRS = {
    "file": attr.string(doc = "The test file, as a path in test/."),
    "runner": attr.label(
        executable = True,
        # Not "exec": the cases run against the torch this build is for.
        cfg = "target",
        doc = "Program that runs the test file: run_test.py with what the file needs, torch aside.",
    ),
    "_tree": attr.label(default = "//:torch_tree"),
}

_torch_list_test = rule(
    implementation = _torch_list_impl,
    test = True,
    attrs = _ATTRS | {"cases": attr.string_dict(doc = "The recorded cases: pytest ids by target name.")},
)

_torch_case_test = rule(
    implementation = _torch_case_impl,
    test = True,
    attrs = _ATTRS | {
        "case": attr.string(doc = "The pytest id of the case."),
        "_pytest_args": attr.label(default = "//tools/bazel:pytest_args"),
        "_rerun": attr.label(default = "//tools/bazel:rerun"),
    },
)

def torch_test(name, srcs, cases = {}, data = [], deps = [], tags = [], target_compatible_with = [], **kwargs):
    """Declares the tests of test/<name>.py: <name>/<case> for each recorded case.

    <name> itself runs them all, and <name>.list checks, or records, the cases.

    Args:
      name: the test, like "test_nn" or "inductor/test_torchinductor".
      srcs: the Python files under test/ that it imports, itself included.
      cases: the recorded cases, each one's pytest id by the name of its target.
      data: the other files under test/ that it reads.
      deps: the Python packages that it imports, torch aside.
      tags: tags of the tests.
      target_compatible_with: where the tests can run.
      **kwargs: what else the file needs, as arguments of py_binary.
    """
    py_binary(
        name = name + ".bin",
        srcs = srcs + ["//tools/bazel:run_test.py"],
        # The whole file in one process: bazel run //test:<name>.bin -- -k <case>
        args = [name + ".py", "--use-pytest"],
        # torch is not among the program's files: each action gets it as inputs
        # of its own. This only has the tree built, and sitecustomize.py puts it
        # on the path.
        data = data + ["//:torch_tree"],
        main = "//tools/bazel:run_test.py",
        # Every process would otherwise compile the modules it imports.
        pyc_collection = "include_pyc",
        tags = ["manual"],
        target_compatible_with = target_compatible_with,
        deps = deps + ["//tools/bazel:site"],
        **kwargs
    )

    # The test actions only report, or compare two short files.
    common = dict(file = name + ".py", runner = name + ".bin", size = "small", tags = tags, target_compatible_with = target_compatible_with)
    _torch_list_test(name = name + ".list", cases = cases, **common)
    for case, id in cases.items():
        _torch_case_test(name = "%s/%s" % (name, case), case = id, **common)
    native.test_suite(name = name, tests = [name + ".list"] + ["%s/%s" % (name, case) for case in cases], tags = tags)

_RECORD = """#!/bin/bash
# Records the cases of every test file in the workspace.
while read -r test listed; do
    record="$BUILD_WORKSPACE_DIRECTORY/{records}/$test"
    mkdir -p "$(dirname "$record")" && cat "$listed" > "$record" || exit 1
done < "{lists}"
echo "Recorded the cases of $(wc -l < "{lists}") test files."
"""

def _torch_lists_impl(ctx):
    listed = [target[TorchCasesInfo] for target in ctx.attr.tests]
    lists = ctx.actions.declare_file(ctx.label.name + ".lists")
    ctx.actions.write(lists, "".join(["%s %s\n" % (info.test, info.listed.short_path) for info in listed]))
    script = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(script, _RECORD.format(lists = lists.short_path, records = RECORDS), is_executable = True)
    return [DefaultInfo(executable = script, runfiles = ctx.runfiles([lists] + [info.listed for info in listed]))]

torch_lists = rule(
    implementation = _torch_lists_impl,
    executable = True,
    attrs = {"tests": attr.label_list(providers = [TorchCasesInfo], doc = "The <name>.list of each test file.")},
    doc = "With bazel run, records the cases of every test file at once.",
)
