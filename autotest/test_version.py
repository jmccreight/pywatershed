import pathlib as pl

import pytest

import pywatershed as pws


@pytest.mark.domainless
def test_version_files_agree():
    """version.txt and pywatershed/version.py must always agree.

    These files are set together by .github/scripts/update_version.py
    but nothing else keeps them from drifting apart between releases.
    See .github/RELEASE.md.
    """
    repo_root = pl.Path(__file__).parent.parent
    version_txt = (repo_root / "version.txt").read_text().strip()
    msg = (
        f"version.txt ('{version_txt}') and pywatershed.__version__ "
        f"('{pws.__version__}') disagree. Reconcile them by running\n"
        "    python .github/scripts/update_version.py -v <version>\n"
        "from the repository root, where <version> is the anticipated "
        "next release with a .dev0 suffix (e.g. 3.0.0.dev0), per "
        ".github/RELEASE.md. If pywatershed is installed non-editably, "
        "reinstall after updating so the installed version matches."
    )
    assert version_txt == pws.__version__, msg
