"""Utility functions for running notebooks from the repository."""

import os
import pathlib as pl


def get_repo_root() -> pl.Path:
    """Get repository root when running from notebooks.

    This function finds the repository root by looking for pyproject.toml
    starting from the current working directory and walking up the tree.
    This works regardless of whether pywatershed is installed in editable
    or non-editable mode, as long as the notebook is being run from within
    the cloned repository.

    Returns:
        Path to the repository root directory.

    Raises:
        FileNotFoundError: If the repository root cannot be found.

    Examples:
        >>> repo_root = get_repo_root()
        >>> assert (repo_root / "pyproject.toml").exists()
    """
    cwd = pl.Path(os.getcwd()).resolve()

    # Try to find pyproject.toml going up the tree
    current = cwd
    for _ in range(10):  # Don't go too far up
        if (current / "pyproject.toml").exists():
            # Verify it's the pywatershed repo by checking for examples/
            if (current / "examples").exists():
                return current
        if current.parent == current:  # Reached filesystem root
            break
        current = current.parent

    raise FileNotFoundError(
        "Could not find pywatershed repository root. "
        "Notebooks must be run from within the cloned repository. "
        f"Current directory: {cwd}"
    )


def get_test_data_dir() -> pl.Path:
    """Get path to test_data directory in the repository.

    This is useful for notebooks that need access to test data files
    that are stored in the repository but not distributed with the
    installed package.

    Returns:
        Path to the test_data directory.

    Raises:
        FileNotFoundError: If test_data directory cannot be found.

    Examples:
        >>> test_data_dir = get_test_data_dir()
        >>> assert test_data_dir.exists()
    """
    repo_root = get_repo_root()
    test_data_dir = repo_root / "test_data"

    if not test_data_dir.exists():
        raise FileNotFoundError(
            f"test_data directory not found at {test_data_dir}. "
            "The repository may be incomplete or corrupted."
        )

    return test_data_dir
