import os
import pathlib as pl
import sys
from fnmatch import fnmatch
from platform import processor
from typing import List
from warnings import warn

import pytest

import pywatershed as pws

# Note: Either PRMS or GSFLOW exectuables may be executed. The choice is
# triggered by the control file field executable_desc. If "gsflow" is
# found in the lower case version of its value, then gsflow is used. Otherwise,
# PRMS is used.

test_data_dir = pl.Path("..")

# Subset this to speed up tests by eliminating some domains
all_domain_dirs = sorted(
    [path for path in test_data_dir.iterdir() if path.is_dir()]
)

# This would change to handle other/additional schedulers
domain_globs_schedule = ["*conus*"]

final_var_names = ["through_rain", "infil", "seg_lateral_inflow"]


def get_ctl_exe_desc(ctl_file):
    import warnings

    with warnings.catch_warnings():
        # This is the only way to silence "invalid" options.
        warnings.simplefilter("ignore")
        ctl = pws.Control.load_prms(
            ctl_file,
            keep_unused_options=True,
            warn_unused_options=False,
        )

    if "executable_desc" in ctl.options.keys():
        exe_desc = ctl.options["executable_desc"][0].lower()
    else:
        exe_desc = "prms"

    return exe_desc


def pytest_addoption(parser):
    parser.addoption(
        "--force",
        required=False,
        action="store_true",
        default=False,
        help=("Force run without scheduler present"),
    )

    parser.addoption(
        "--domain",
        required=False,
        action="append",
        default=[],
        help=(
            "Domain(s) to run (name of domain dir and NOT path to it). "
            "You can pass multiples of this argument. If not used, "
            "all self-contained domains in test_data/ will be run."
        ),
    )

    parser.addoption(
        "--control_pattern",
        required=False,
        action="append",
        default=[],
        help=(
            "Control file glob(s) (NOT path. Can drop '.control'). "
            "You can pass multiples of this argument. If not used, "
            "all control files in each domain will be run."
        ),
    )

    parser.addoption(
        "--write_log",
        help=(
            "Write the PRMS/GSFLOW stdout to log matching control file name."
        ),
        action="store_true",
    )


@pytest.fixture(scope="function")
def exe(simulation):
    exe_desc = get_ctl_exe_desc(simulation["control_file"])
    platform = sys.platform.lower()
    if "gsflow" in exe_desc:
        if platform == "win32":
            pytest.skip(f"GSFLOW binary not yet provided for {platform}")
        elif platform == "darwin":
            if processor() == "arm":
                exe_name = "gsflow_2.4.0_ifort_apple_silicon_dbl_prec"
            else:
                pytest.skip(
                    f"GSFLOW binary not yet provided for {platform}:intel"
                )
        elif platform == "linux":
            pytest.skip(f"GSFLOW binary not yet provided for {platform}")

    elif "5.2.1.1" in exe_desc:
        if platform == "win32":
            exe_name = "prms_5.2.1.1_gfort_win_dbl_prec.exe"
        elif platform == "darwin":
            if processor() == "arm":
                exe_name = "prms_5.2.1.1_ifort_apple_silicon_dbl_prec"
            else:
                pytest.skip(
                    "PRMS 5.2.1.1 binary not yet provided for "
                    f"{platform}:intel"
                )
        elif platform == "linux":
            exe_name = "prms_5.2.1.1_gfort_linux_dbl_prec"
    else:
        if platform == "win32":
            exe_name = "prms_win_gfort_dbl_prec.exe"
        elif platform == "darwin":
            if processor() == "arm":
                exe_name = "prms_mac_m1_ifort_dbl_prec"
            else:
                exe_name = "prms_mac_intel_gfort_dbl_prec"
        elif platform == "linux":
            exe_name = "prms_linux_gfort_dbl_prec"

    # <<
    exe_pth = pl.Path(f"../../bin/{exe_name}").resolve()

    return exe_pth


def scheduler_active():
    slurm = os.getenv("SLURM_JOB_ID") is not None
    pbs = os.getenv("PBS_JOBID") is not None
    # add more scheduler checks
    return slurm or pbs


def enforce_scheduler(test_dir):
    """Enforce the use of scheduler

    Args:
        test_dir: the domain directory to run or schedule

    Return:
        True if must use a scheduler, False if not
    """

    if scheduler_active():
        return False
    glob_match = list(
        fnmatch(str(test_dir), gg) for gg in domain_globs_schedule
    )
    if any(glob_match):
        msg = (
            f"Skipping domain '{test_dir}' which must be scheduled or use "
            "--force to override skip"
        )
        warn(msg, UserWarning)
        return True

    return False


def collect_simulations(
    domain_list: list,
    control_pattern_list,
    force: bool = True,
    verbose: bool = False,
    write_log: bool = False,
):
    simulations = {}
    for dom_dir in all_domain_dirs:
        # ensure this is a self-contained run (all files in repo)
        if not (dom_dir / "prcp.cbh").exists():
            # this is kind of a silly check... until something better needed
            continue

        # filter selected domains
        if len(domain_list) and (dom_dir.name not in domain_list):
            continue

        # optionally enforce scheduler
        if not force:
            skip = enforce_scheduler(dom_dir)
            if skip:
                continue

        control_file_candidates = sorted(dom_dir.glob("*.control"))

        # filter against control pattern
        control_files = []
        for control in control_file_candidates:
            if not len(control_pattern_list):
                control_files += [control]
            else:
                for gg in control_pattern_list:
                    if gg in control.name:
                        control_files += [control]

        for control in control_files:
            id = f"{dom_dir.name}:{control.with_suffix('').name}"
            ctl = pws.Control.load_prms(control, warn_unused_options=False)
            output_dir = dom_dir / ctl.options["netcdf_output_dir"]
            simulations[id] = {
                "ws": dom_dir,
                "control_file": control,
                "output_dir": output_dir,
                "write_log": write_log,
            }

    # make sure all requested domains were found
    if len(domain_list) and (len(simulations) < len(domain_list)):
        requested = set(domain_list)
        found = [pl.Path(dd).name for dd in simulations.keys()]
        requested_not_found = requested.difference(found)
        msg = (
            f"The following requested domains were not found: "
            f"{requested_not_found}"
        )
        pytest.exit(msg)

    if verbose:
        print("\nrun_domains.py found the following domains to run:\n")
        print(f"{list(simulations.keys())}")

    return simulations


def collect_csv_files(simulations: dict) -> List[tuple]:
    csv_files = []
    for key, value in simulations.items():
        control = value["control_file"]
        csv_files_dom = sorted(value["output_dir"].glob("*.csv"))
        csv_files += [
            (control, ff) for ff in csv_files_dom if ff.name != "stats.csv"
        ]
    return csv_files


def pytest_generate_tests(metafunc):
    domain_list = metafunc.config.getoption("domain")
    control_pattern_list = metafunc.config.getoption("control_pattern")
    force = metafunc.config.getoption("force")
    write_log = metafunc.config.getoption("write_log")

    simulations = collect_simulations(
        domain_list, control_pattern_list, force=force, write_log=write_log
    )
    control_csv_files = collect_csv_files(simulations)

    if "control_csv_file" in metafunc.fixturenames:
        ids = [
            ff.parent.parent.name
            + ":"
            + cc.with_suffix("").name
            + ":"
            + ff.name
            for cc, ff in control_csv_files
        ]
        metafunc.parametrize("control_csv_file", control_csv_files, ids=ids)

    if "control_soltab_file" in metafunc.fixturenames:
        # The single/combined soltab output file was from PRMS 5.2.1. In that
        # case this parameterization doesnt really make sense because it's
        # always the same, but this is useful because it also identifies the
        # ws/domain_dir. The processing of this parameter returns 3 individual
        # output netcdf files: soltab_potsw, soltab_horad_potsw, soltab_sunhrs.
        # In PRMS 5.3+ the soltab file was split into 4 output files:
        # soltab_sunhrs.csv, soltab_potsw.csv, obliquity.csv, and
        # solar_declination.csv. I've modified GSFLOW 2.4.1 to also output
        #  soltab_horad_potsw.csv. Sadly, the output format does not match
        # PRMS output files, there is no date column, presumably because it
        # has a doy dimension 1-366.
        control_soltab_files = []
        for kk, vv in simulations.items():
            exe_desc = get_ctl_exe_desc(vv["control_file"])
            if "5.2.1.1" in exe_desc:
                soltab_name = "soltab_debug_5.2.1.1"
            else:
                soltab_name = "soltab_debug"

            control_soltab_files += [
                (vv["control_file"], vv["ws"] / soltab_name)
            ]

        # <
        ids = [
            ff.parent.name + ":" + cc.with_suffix("").name + ":" + ff.name
            for cc, ff in control_soltab_files
        ]

        metafunc.parametrize(
            "control_soltab_file",
            control_soltab_files,
            ids=ids,
            scope="session",
        )

    if "simulation" in metafunc.fixturenames:
        metafunc.parametrize(
            "simulation",
            list(simulations.values()),
            ids=list(simulations.keys()),
            scope="session",
        )

    if "control_final_file" in metafunc.fixturenames:
        control_final_files = [
            (vv["control_file"], vv["output_dir"] / var)
            for kk, vv in simulations.items()
            for var in final_var_names
        ]
        ids = [
            ff.parent.parent.name
            + ":"
            + cc.with_suffix("").name
            + ":"
            + ff.name
            for cc, ff in control_final_files
        ]

        # these are not really file names they are domain/key_var
        metafunc.parametrize(
            "control_final_file", control_final_files, ids=ids, scope="session"
        )
