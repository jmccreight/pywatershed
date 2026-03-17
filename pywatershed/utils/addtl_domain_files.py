# This utility gets additional doain files these files are not part of the pws
# repo.

import argparse
import hashlib
import urllib.request as request
import zipfile
from pathlib import Path
from shutil import rmtree

from pywatershed import constants

pkg_root_dir = constants.__pywatershed_root__
addtl_domains_dir = pkg_root_dir / "data/pywatershed_addtl_domains"

# URL and MD5 must be updated together when new version is released
addtl_domains_config = {
    "url": (
        "https://github.com/DOI-USGS/pywatershed/"
        "releases/download/2.0.3/pywatershed_addtl_domains.zip"
    ),
    "md5": "ae3b933fb733ea79818d67822cfb5d85",
}


def compute_md5(file_path: Path) -> str:
    """Compute MD5 checksum of a file."""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def download(force: bool = False) -> None:
    addtl_domains_file = pkg_root_dir / "data/pywatershed_addtl_domains.zip"

    # Check if zip file exists and verify its MD5
    if addtl_domains_file.exists():
        current_md5 = compute_md5(addtl_domains_file)
        if current_md5 != addtl_domains_config["md5"]:
            print(
                f"MD5 mismatch: expected {addtl_domains_config['md5']}, "
                f"got {current_md5}"
            )
            print(f"Removing {addtl_domains_dir} and {addtl_domains_file}")
            if addtl_domains_dir.exists():
                rmtree(addtl_domains_dir)
            addtl_domains_file.unlink()

    if force and addtl_domains_dir.exists():
        rmtree(addtl_domains_dir)
        if addtl_domains_file.exists():
            addtl_domains_file.unlink()

    if not addtl_domains_dir.exists() or not addtl_domains_file.exists():
        print(
            f"Downloading {addtl_domains_config['url']} to "
            f"{addtl_domains_file}"
        )
        request.urlretrieve(addtl_domains_config["url"], addtl_domains_file)

        # Verify MD5 checksum after download
        downloaded_md5 = compute_md5(addtl_domains_file)
        if downloaded_md5 != addtl_domains_config["md5"]:
            raise ValueError(
                f"Downloaded file MD5 mismatch: "
                f"expected {addtl_domains_config['md5']}, "
                f"got {downloaded_md5}"
            )

        if addtl_domains_dir.exists():
            rmtree(addtl_domains_dir)
        with zipfile.ZipFile(addtl_domains_file, "r") as zz:
            zz.extractall(pkg_root_dir / "data")

        assert addtl_domains_dir.exists()

    return


def get_addtl_domains_dir(domain: str | None = None) -> Path:
    """Get the path to the additional domains directory.

    Args:
        domain: Optional domain name. If provided, returns the path to the
                specific domain directory (e.g., "fgr_2yr").
                If None, returns the root additional domains directory.

    Returns:
        Path to the additional domains directory or domain-specific directory.

    Examples:
        >>> get_addtl_domains_dir()  # -> .../data/pywatershed_addtl_domains
        >>> get_addtl_domains_dir(
        ...     "fgr_2yr"
        ... )  # -> .../data/pywatershed_addtl_domains/fgr_2yr
    """
    if domain is None:
        return addtl_domains_dir
    return addtl_domains_dir / domain


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download additional domain files for pywatershed"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download by removing existing files first",
    )
    args = parser.parse_args()
    download(force=args.force)
