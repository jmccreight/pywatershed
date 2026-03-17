# This utility gets GIS files needed for certain visualizations and
# calculations. These GIS files are not part of the pws repo.

import argparse
import hashlib
import urllib.request as request
import zipfile
from pathlib import Path
from shutil import rmtree

from pywatershed import constants

pkg_root_dir = constants.__pywatershed_root__
gis_dir = pkg_root_dir / "data/pywatershed_gis"

# URL and MD5 must be updated together when new version is released
gis_config = {
    "url": (
        "https://github.com/DOI-USGS/pywatershed/"
        "releases/download/1.1.0/pywatershed_gis.zip"
    ),
    "md5": "dca0395639e62b84e9f82558d64529f3",
}


def compute_md5(file_path: Path) -> str:
    """Compute MD5 checksum of a file."""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def download(force: bool = False) -> None:
    gis_file = pkg_root_dir / "data/pywatershed_gis.zip"

    # Check if zip file exists and verify its MD5
    if gis_file.exists():
        current_md5 = compute_md5(gis_file)
        if current_md5 != gis_config["md5"]:
            print(
                f"MD5 mismatch: expected {gis_config['md5']}, "
                f"got {current_md5}"
            )
            print(f"Removing {gis_dir} and {gis_file}")
            if gis_dir.exists():
                rmtree(gis_dir)
            gis_file.unlink()

    if force and gis_dir.exists():
        rmtree(gis_dir)
        if gis_file.exists():
            gis_file.unlink()

    if not gis_dir.exists() or not gis_file.exists():
        print(f"Downloading {gis_config['url']} to {gis_file}")
        request.urlretrieve(gis_config["url"], gis_file)

        # Verify MD5 checksum after download
        downloaded_md5 = compute_md5(gis_file)
        if downloaded_md5 != gis_config["md5"]:
            raise ValueError(
                f"Downloaded file MD5 mismatch: "
                f"expected {gis_config['md5']}, "
                f"got {downloaded_md5}"
            )

        if gis_dir.exists():
            rmtree(gis_dir)
        with zipfile.ZipFile(gis_file, "r") as zz:
            zz.extractall(pkg_root_dir / "data")

        assert gis_dir.exists()

    return


def get_gis_dir(domain: str | None = None) -> Path:
    """Get the path to the GIS directory.

    Args:
        domain: Optional domain name. If provided, returns the path to the
                specific domain's GIS directory (e.g., "drb_2yr").
                If None, returns the root GIS directory.

    Returns:
        Path to the GIS directory or domain-specific GIS directory.

    Examples:
        >>> get_gis_dir()  # Returns .../data/pywatershed_gis
        >>> get_gis_dir("drb_2yr")  # Returns .../data/pywatershed_gis/drb_2yr
    """
    if domain is None:
        return gis_dir
    return gis_dir / domain


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download GIS files for pywatershed"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download by removing existing files first",
    )
    args = parser.parse_args()
    download(force=args.force)
