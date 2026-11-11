"""Repositories derived from the source tree."""

# Where `bazel run //test:<name>.list` records the cases of a test file, from the
# top of the workspace (see torch_test.bzl).
RECORDS = ".test_cases"

def _test_list_impl(rctx):
    root = rctx.workspace_root
    script = rctx.path(Label("//tools/bazel:list_tests.py"))
    records = root.get_child(RECORDS)
    rctx.watch_tree(root.get_child("test"))
    rctx.watch(root.get_child("tools/testing/discover_tests.py"))
    rctx.watch(script)
    rctx.watch(records)
    if records.exists:
        rctx.watch_tree(records)

    # -B keeps the interpreter from writing bytecode into the source tree.
    result = rctx.execute([rctx.path(rctx.attr._python), "-B", script, root, records])
    if result.return_code != 0:
        fail("could not list tests: %s" % result.stderr)
    listing = json.decode(result.stdout)
    rctx.file("tests.bzl", "TESTS = %r\nCASES = %r\n" % (listing["tests"], listing["cases"]))
    rctx.file("BUILD.bazel", "")

_test_list = repository_rule(
    implementation = _test_list_impl,
    attrs = {"_python": attr.label(default = "@python_3_14_host//:python")},
    doc = "Provides tests.bzl with the test files that test/run_test.py runs and their recorded cases.",
)

def _torch_impl(_mctx):
    _test_list(name = "torch_tests")

torch = module_extension(implementation = _torch_impl)
