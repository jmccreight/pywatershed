"""Utilities for locating and compiling PRMS/GSFLOW executables.

The binary naming conventions mirror those in test_data/generate/conftest.py.
Compilation logic mirrors autotest/ci_local.sh.
"""

import pathlib as pl
import shutil
import subprocess
import sys
from platform import processor
from typing import Optional


def _get_repo_root() -> pl.Path:
    from pywatershed.utils.notebook_utils import get_repo_root

    return get_repo_root()


def _get_bin_dir() -> pl.Path:
    return _get_repo_root() / "bin"


def _get_src_dir() -> pl.Path:
    return _get_repo_root() / "prms_src"


def get_prms_exe_name(exe_desc: str = "prms") -> str:
    """Return the binary filename for the given exe_desc and current platform.

    Parameters
    ----------
    exe_desc : str
        Description string from the control file's ``executable_desc`` field,
        or a plain version/type string such as ``"5.2.1.1"``, ``"gsflow"``,
        or ``"prms"``.

    Returns
    -------
    str
        The binary filename (no directory component).

    Raises
    ------
    NotImplementedError
        When a pre-built binary is not available for the current platform.
    ValueError
        When the platform is not recognised.
    """
    exe_desc_lower = exe_desc.lower()
    platform = sys.platform.lower()
    proc = processor()

    if "gsflow" in exe_desc_lower:
        if platform == "win32":
            return "gsflow_2.4.0_gfortran_windows_dbl_prec.exe"
        elif platform == "darwin":
            if proc == "arm":
                return "gsflow_2.4.0_ifort_apple_silicon_dbl_prec"
            else:
                raise NotImplementedError(
                    f"GSFLOW binary not yet provided for {platform}:intel"
                )
        elif platform == "linux":
            return "gsflow_2.4.0_gfortran_linux_dbl_prec"
        else:
            raise ValueError(f"Unsupported platform: {platform}")

    elif "5.2.1.1" in exe_desc_lower:
        if platform == "win32":
            return "prms_5.2.1.1_gfort_win_dbl_prec.exe"
        elif platform == "darwin":
            if proc == "arm":
                return "prms_5.2.1.1_gfortran_apple_silicon_dbl_prec"
            else:
                raise NotImplementedError(
                    "PRMS 5.2.1.1 binary not yet provided for "
                    f"{platform}:intel"
                )
        elif platform == "linux":
            return "prms_5.2.1.1_gfort_linux_dbl_prec"
        else:
            raise ValueError(f"Unsupported platform: {platform}")

    elif "5.2.1" in exe_desc_lower:
        # note: this check must come after the "5.2.1.1" check above.
        # The repository's prms_src/prms5.2.1 source (with cascades and
        # G0-precision CBH output patches) compiled on-demand.
        if platform == "win32":
            return "prms_5.2.1_gfort_win_dbl_prec.exe"
        elif platform == "darwin":
            if proc == "arm":
                return "prms_5.2.1_gfortran_apple_silicon_dbl_prec"
            else:
                return "prms_5.2.1_gfortran_mac_intel_dbl_prec"
        elif platform == "linux":
            return "prms_5.2.1_gfort_linux_dbl_prec"
        else:
            raise ValueError(f"Unsupported platform: {platform}")

    else:
        # Default PRMS binary
        if platform == "win32":
            return "prms_win_gfort_dbl_prec.exe"
        elif platform == "darwin":
            if proc == "arm":
                return "prms_mac_m1_ifort_dbl_prec"
            else:
                return "prms_mac_intel_gfort_dbl_prec"
        elif platform == "linux":
            return "prms_linux_gfort_dbl_prec"
        else:
            raise ValueError(f"Unsupported platform: {platform}")


def get_prms_exe_path(exe_desc: str = "prms") -> pl.Path:
    """Return the full path to the PRMS/GSFLOW binary in the repo bin/ dir.

    Parameters
    ----------
    exe_desc : str
        See :func:`get_prms_exe_name`.

    Returns
    -------
    pathlib.Path
        Absolute path to the binary (may or may not exist yet).
    """
    return (_get_bin_dir() / get_prms_exe_name(exe_desc)).resolve()


def compile_prms(
    source: str = "5.2.1.1",
    force: bool = False,
) -> pl.Path:
    """Compile PRMS from source and return the path to the resulting binary.

    Parameters
    ----------
    source : str
        Source version to compile.  One of ``"5.2.1"`` (the repository
        source with cascades and G0-precision CBH output patches) or
        ``"5.2.1.1"``.
    force : bool
        If ``True``, recompile even when the binary already exists.
        Default is ``False`` (skip if binary present).

    Returns
    -------
    pathlib.Path
        Absolute path to the compiled binary.

    Raises
    ------
    ValueError
        If *source* is not a supported value.
    RuntimeError
        If compilation succeeds but the expected output binary is missing.
    subprocess.CalledProcessError
        If any subprocess step fails.
    """
    compilable_sources = ("5.2.1", "5.2.1.1")
    if source not in compilable_sources:
        raise ValueError(
            f"Unsupported source '{source}'. Compilable sources are "
            f"{compilable_sources}."
        )

    binary_path = get_prms_exe_path(source)
    src_dir = _get_src_dir() / f"prms{source}"

    if binary_path.exists() and not force:
        print(f"PRMS {source} binary already exists: {binary_path}")
        return binary_path

    print(
        f"\n{'*' * 30}\n"
        f"Compiling PRMS {source}\n"
        f"{'*' * 30}\n"
        f"Source dir : {src_dir}\n"
        f"Binary dest: {binary_path}\n"
    )

    orig_dir = pl.Path.cwd()
    try:
        import os

        os.chdir(src_dir)

        # Prefer the compilers of the active python environment (e.g.
        # conda-forge gcc/gfortran) over whatever is on the PATH: local
        # PATHs can put stale/mismatched toolchains first. The compiler
        # names must remain plain "gcc"/"gfortran" as the makelists key
        # their flags on these exact names.
        sub_env = os.environ.copy()
        env_bin = pl.Path(sys.prefix) / "bin"
        if (env_bin / "gfortran").exists():
            sub_env["PATH"] = str(env_bin) + os.pathsep + sub_env["PATH"]

        # Clean previous build
        subprocess.run(
            ["make", "clean", "MAKE=make"],
            check=True,
            env=sub_env,
        )

        # Build with double precision
        subprocess.run(
            [
                "make",
                "DBL_PREC=true",
                "FC=gfortran",
                "CC=gcc",
                "MAKE=make",
            ],
            check=True,
            env=sub_env,
        )

        compiled_bin = src_dir / "bin" / "prms"
        if not compiled_bin.exists():
            raise RuntimeError(
                "Compilation appeared to succeed but "
                f"bin/prms not found in {src_dir}"
            )

        _get_bin_dir().mkdir(parents=True, exist_ok=True)
        shutil.copy(compiled_bin, binary_path)
        print(f"Successfully compiled and installed to {binary_path}")

    finally:
        os.chdir(orig_dir)

    return binary_path


def get_or_compile_prms_exe(
    exe_desc: str = "prms",
    force: bool = False,
    compile_source: Optional[str] = None,
) -> pl.Path:
    """Return the PRMS/GSFLOW executable, compiling from source if needed.

    For ``exe_desc`` values that map to a compilable source (currently only
    ``"5.2.1.1"``), the binary will be compiled automatically when it is not
    present (or when *force* is ``True``).  For other variants the binary
    must already exist.

    Parameters
    ----------
    exe_desc : str
        See :func:`get_prms_exe_name`.
    force : bool
        Passed through to :func:`compile_prms`.  Forces recompilation even
        when the binary already exists.
    compile_source : str or None
        Override the source version to compile.  When ``None`` the source
        version is inferred from *exe_desc*.

    Returns
    -------
    pathlib.Path
        Absolute path to the executable.

    Raises
    ------
    FileNotFoundError
        If the binary does not exist and cannot be compiled.
    """
    exe_path = get_prms_exe_path(exe_desc)

    if exe_path.exists() and not force:
        return exe_path

    # Determine which source to compile
    source = compile_source
    if source is None:
        if "5.2.1.1" in exe_desc.lower():
            source = "5.2.1.1"
        elif "5.2.1" in exe_desc.lower():
            source = "5.2.1"

    if source is not None:
        return compile_prms(source=source, force=force)

    if not exe_path.exists():
        raise FileNotFoundError(
            f"Executable not found and no compilable source is available "
            f"for exe_desc='{exe_desc}': {exe_path}"
        )

    return exe_path
