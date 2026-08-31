import difflib
import importlib.util
import pathlib as pl

import pytest

repo_root = pl.Path(__file__).parent.parent


def _load_api_surface_module():
    path = repo_root / ".github" / "scripts" / "api_surface.py"
    spec = importlib.util.spec_from_file_location("api_surface", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.domainless
def test_api_surface_unchanged():
    """The public API surface must match its committed baseline.

    The baseline (autotest/api_surface.txt) is a sorted,
    one-fact-per-line snapshot of exports, __init__ signatures, declared
    input/variable/parameter name sets, control options, and metadata.
    Any diff means the public API changed: if intentional, regenerate
    the baseline with

        python .github/scripts/api_surface.py --write

    and, for a removal, rename, or other breaking line change, add an
    entry under Breaking Changes in doc/whats-new.rst. Added lines are
    additive changes and need only the regeneration.
    """
    api_surface = _load_api_surface_module()
    current = api_surface.generate()
    baseline = api_surface.BASELINE.read_text()
    if current == baseline:
        return
    diff = "\n".join(
        difflib.unified_diff(
            baseline.splitlines(),
            current.splitlines(),
            fromfile="committed baseline",
            tofile="current package",
            lineterm="",
        )
    )
    raise AssertionError(
        "The public API surface changed. See this test's docstring "
        f"for how to proceed.\n{diff}"
    )
