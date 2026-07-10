#!/usr/bin/env python
"""Run 3: pywatershed below snow, forced by GSFLOW (run 1) outputs.

The below-snow process chain — PRMSRunoffAg, PRMSSoilzoneAg(ObsET),
PRMSGroundwater, PRMSChannel — is run with everything above it (solar,
atmosphere/transpiration, canopy, snow) supplied from run 1's GSFLOW
outputs. Together with run 2 (full pywatershed), this tests the working
hypothesis that GSFLOW-vs-pywatershed divergence is dominated by
PRMSSnow (and above) reproducibility.

The file inputs required by the below-snow chain were determined with
pws.Model.solve_inputs (see ag_comparison/CLAUDE.md). Per phase they
are: 12 GSFLOW output variables (converted CSV -> NetCDF), the derived
diagnostic through_rain (which additionally needs pk_ice_change and
freeh2o_change, derived here from GSFLOW pk_ice/freeh2o), plus ag_frac
(static for spinup, dynamic for analysis) and, for analysis,
aet_observed (from the domain). Each phase gets its own self-contained
input directory (gsflow_nc_{phase}/) so the domain dir is not touched.

Note there are no frost/transpiration classes in run 3: transp_on is an
input, taken directly from GSFLOW (whose analysis used dynamic frost).

Phases (default all, in order; conversion requires run 1 to have been
run):

    python run.py [convert spinup analysis]

The spinup and analysis are chained by a full-model restart at
2000-12-31, as in run 2. Outputs -> output_spinup/, output_analysis/;
restart files -> restart/.
"""

import argparse
import pathlib as pl
import shutil
import sys

import numpy as np
import xarray as xr

import pywatershed as pws

HERE = pl.Path(__file__).resolve().parent

# reuse the test-data generation machinery for conversion/diagnostics
sys.path.insert(0, str(HERE.parent.parent / "test_data" / "generate"))
from prms_convert_to_netcdf import convert_csv_to_nc  # noqa: E402
from prms_diagnostic_variables import (  # noqa: E402
    diagnose_final_vars_to_nc,
)

PHASES = ("convert", "spinup", "analysis")

# per pws.Model.solve_inputs on the process chains below: file inputs
# supplied by GSFLOW outputs (through_rain is derived, not direct)
GSFLOW_INPUT_VARS = [
    "hru_intcpevap",
    "intcp_changeover",
    "net_ppt",
    "net_rain",
    "net_snow",
    "pkwater_equiv",
    "potet",
    "pptmix_nopack",
    "snow_evap",
    "snowcov_area",
    "snowmelt",
    "transp_on",
]
# needed (as _change) by the through_rain diagnostic
CHANGE_PARENT_VARS = ["pk_ice", "freeh2o"]

SPINUP_PROCESSES = [
    pws.PRMSRunoffAg,
    pws.PRMSSoilzoneAg,
    pws.PRMSGroundwater,
    pws.PRMSChannel,
]

ANALYSIS_PROCESSES = [
    pws.PRMSRunoffAg,
    pws.PRMSSoilzoneAgObsET,
    pws.PRMSGroundwater,
    pws.PRMSChannel,
]


def control_file(phase):
    name = {
        "spinup": "spinup_2000.control",
        "analysis": "analysis_2001.control",
    }[phase]
    return HERE.parent / "01_gsflow" / name


def input_dir(phase):
    return HERE / f"gsflow_nc_{phase}"


def write_static_ag_frac(out_dir, domain_dir, params):
    """Static ag_frac.nc (time x nhru) from the parameter value, as in
    run 2 (but written to the run-3 input dir, not the domain dir)."""
    ag = xr.load_dataarray(domain_dir / "tmin.nc")
    ag[:, :] = params.parameters["ag_frac"]
    ag.name = "ag_frac"
    ag.to_netcdf(out_dir / "ag_frac.nc")


def convert_phase(phase, domain_dir):
    """CSV -> NetCDF for the GSFLOW-supplied inputs of one phase, plus
    derived variables and domain-supplied files, into gsflow_nc_{phase}."""
    gsflow_out = HERE.parent / "01_gsflow" / f"output_{phase}"
    out_dir = input_dir(phase)
    out_dir.mkdir(exist_ok=True)
    print(f"Converting GSFLOW {phase} outputs -> {out_dir}")

    for vv in GSFLOW_INPUT_VARS + CHANGE_PARENT_VARS:
        convert_csv_to_nc(vv, gsflow_out, out_dir)

    # pk_ice_change/freeh2o_change: current minus previous. The initial
    # previous value is the PRMSSnow initial condition (zero) for spinup;
    # for analysis it chains from the last day of the spinup phase.
    for vv in CHANGE_PARENT_VARS:
        current = xr.open_dataarray(out_dir / f"{vv}.nc")
        prev = current.copy()
        prev[:] = np.roll(current.values, 1, axis=0)
        if phase == "spinup":
            prev[0, :] = 0.0  # PRMSSnow init value for pk_ice and freeh2o
        else:
            spinup_da = xr.open_dataarray(input_dir("spinup") / f"{vv}.nc")
            prev[0, :] = spinup_da.values[-1, :]
        change = current - prev
        change.rename(f"{vv}_change").to_netcdf(out_dir / f"{vv}_change.nc")

    # through_rain from the .nc files just written
    success = diagnose_final_vars_to_nc(
        "through_rain",
        out_dir,
        control_file(phase),
        output_dir=out_dir,
    )
    assert success, "through_rain derivation failed"

    # domain-supplied inputs, per phase (as in run 2 but into out_dir)
    control = pws.Control.load_prms(
        control_file(phase), warn_unused_options=False
    )
    params = pws.parameters.PrmsParameters.load(
        domain_dir / control.options["parameter_file"]
    )
    if phase == "spinup":
        write_static_ag_frac(out_dir, domain_dir, params)
    else:
        shutil.copy2(
            domain_dir / "dyn_ag_frac.param", out_dir / "ag_frac.param"
        )
        shutil.copy2(
            domain_dir / "aet_observed.nc", out_dir / "aet_observed.nc"
        )


def gsflow_output_vars(control, processes):
    """The GSFLOW control's nhruOutVar_names, filtered to vars this
    process chain actually produces (as in run 2)."""
    requested = control.options.get("nhruOutVar_names")
    if requested is None:
        return None
    requested = [str(v) for v in requested]
    available = set()
    for cls in processes:
        available |= set(cls.get_variables())
    keep = [v for v in requested if v in available]
    dropped = [v for v in requested if v not in available]
    if dropped:
        print(
            f"  Note: {len(dropped)} GSFLOW output vars not produced by "
            f"this process chain, skipped: {sorted(dropped)}"
        )
    return keep


def run_phase(phase, domain_dir, restart_dir):
    """Run one below-snow phase forced by the converted GSFLOW outputs."""
    processes = SPINUP_PROCESSES if phase == "spinup" else ANALYSIS_PROCESSES
    out_dir = HERE / f"output_{phase}"
    out_dir.mkdir(exist_ok=True)

    control = pws.Control.load_prms(
        control_file(phase),
        warn_unused_options=False,
        keep_unused_options=True,
    )
    control.options["input_dir"] = input_dir(phase)
    control.options["netcdf_output_dir"] = out_dir
    control.options["iter_aet_flag"] = phase == "analysis"
    control.options["intcp_changeover_in_net_rain"] = False
    control.options["imbalance_behavior"] = "warn"
    if phase == "spinup":
        control.options["restart_write"] = str(restart_dir)
        control.options["restart_write_freq"] = "y"
    else:
        control.options["restart_read"] = str(restart_dir)

    out_vars = gsflow_output_vars(control, processes)
    if out_vars:
        control.options["netcdf_output_var_names"] = out_vars

    params = pws.parameters.PrmsParameters.load(
        domain_dir / control.options["parameter_file"]
    )

    # the process-list form is fine here: all four class names are in
    # pws.base.model.process_order_nhm (unlike run 2's TranspFrost classes)
    print(f"=== {phase}: below-snow chain forced by GSFLOW outputs ===")
    model = pws.Model(processes, control=control, parameters=params)
    model.run(finalize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phases",
        nargs="*",
        choices=list(PHASES),
        default=list(PHASES),
        help="phases to run (default: all, in canonical order)",
    )
    args = parser.parse_args()
    phases = [pp for pp in PHASES if pp in args.phases]

    domain_dir = pws.utils.get_addtl_domains_dir("fgr_ag_2yr")
    restart_dir = HERE / "restart"
    restart_dir.mkdir(exist_ok=True)

    if "convert" in phases:
        convert_phase("spinup", domain_dir)
        convert_phase("analysis", domain_dir)
    if "spinup" in phases:
        run_phase("spinup", domain_dir, restart_dir)
    if "analysis" in phases:
        run_phase("analysis", domain_dir, restart_dir)

    print(f"Done: {', '.join(phases)}")


if __name__ == "__main__":
    main()
