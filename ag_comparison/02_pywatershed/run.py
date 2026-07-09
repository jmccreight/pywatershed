#!/usr/bin/env python
"""Run 2: full pywatershed, exactly parallel to run 1 (GSFLOW).

Two chained phases on the fgr_ag_2yr domain, reusing run 1's controls:

  Year 1 spinup (2000): PRMSRunoffAg + PRMSSoilzoneAg, iter_aet_flag=False,
    static ag_frac; writes a full-model restart at 2000-12-31.
  Year 2 analysis (2001): PRMSRunoffAg + PRMSSoilzoneAgObsET,
    iter_aet_flag=True, dynamic ag_frac; reads that restart.

pywatershed sets init_time = start_time - time_step, so the year-2 control
(start 2001-01-01) reads init_time 2000-12-31 — matching the restart files
year 1 writes on the year boundary (restart_write_freq="y"). Restart options
set in control.options propagate to every process.

Approach follows examples/10_ag_irrigation_use.ipynb: read the PRMS control,
edit in memory for pywatershed options, and handle the ag_frac input by
writing files into the domain dir (static ag_frac.nc for spinup, dynamic
ag_frac.param for analysis), cleaned up in a finally block.

Run: python run.py    (outputs -> output_spinup/, output_analysis/)
"""

import pathlib as pl
import shutil

import xarray as xr

import pywatershed as pws

HERE = pl.Path(__file__).resolve().parent

WU_OL_PROCESSES = [
    pws.PRMSSolarGeometry,
    pws.PRMSAtmosphereTranspFrost,
    pws.PRMSCanopy,
    pws.PRMSSnow,
    pws.PRMSRunoffAg,
    pws.PRMSSoilzoneAg,
    pws.PRMSGroundwater,
    pws.PRMSChannel,
]

WU_ANALYSIS_PROCESSES = [
    pws.PRMSSolarGeometry,
    pws.PRMSAtmosphereTranspFrost,
    pws.PRMSCanopy,
    pws.PRMSSnow,
    pws.PRMSRunoffAg,
    pws.PRMSSoilzoneAgObsET,
    pws.PRMSGroundwater,
    pws.PRMSChannel,
]


def gsflow_output_vars(control, processes):
    """The GSFLOW control's nhruOutVar_names, filtered to vars pywatershed
    actually produces (so run 2 outputs what run 1 did, where possible)."""
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
            f"pywatershed, skipped: {sorted(dropped)}"
        )
    return keep


def load_control(
    control_file,
    domain_dir,
    output_dir,
    iter_aet_flag,
    processes,
    restart_write=None,
    restart_read=None,
):
    """Load a PRMS control and edit it in memory for pywatershed."""
    control = pws.Control.load_prms(
        control_file, warn_unused_options=False, keep_unused_options=True
    )
    control.options["input_dir"] = domain_dir
    control.options["netcdf_output_dir"] = output_dir
    control.options["iter_aet_flag"] = iter_aet_flag
    # pywatershed does not include intcp_changeover in net_rain.
    control.options["intcp_changeover_in_net_rain"] = False
    # GSFLOW analysis had non-fatal mass-balance messages; don't hard-fail.
    control.options["imbalance_behavior"] = "warn"

    out_vars = gsflow_output_vars(control, processes)
    if out_vars:
        control.options["netcdf_output_var_names"] = out_vars

    if restart_write is not None:
        control.options["restart_write"] = restart_write
        control.options["restart_write_freq"] = "y"
    if restart_read is not None:
        control.options["restart_read"] = restart_read

    return control


def load_params(control, domain_dir):
    return pws.parameters.PrmsParameters.load(
        domain_dir / control.options["parameter_file"]
    )


def write_static_ag_frac(domain_dir, params):
    """Static ag_frac.nc (time x nhru) from the parameter value, as in
    examples/10_ag_irrigation_use.ipynb. pywatershed finds ag_frac.nc before
    ag_frac.param, so the dynamic file must be absent for this to be used."""
    ag = xr.load_dataarray(domain_dir / "tmin.nc")
    ag[:, :] = params.parameters["ag_frac"]
    ag.name = "ag_frac"
    out = domain_dir / "ag_frac.nc"
    if out.exists():
        out.unlink()
    ag.to_netcdf(out)
    return out


def main():
    domain_dir = pws.utils.get_addtl_domains_dir("fgr_ag_2yr")
    restart_dir = HERE / "restart"
    out_spinup = HERE / "output_spinup"
    out_analysis = HERE / "output_analysis"
    for d in (restart_dir, out_spinup, out_analysis):
        d.mkdir(exist_ok=True)

    ctl_spinup = HERE.parent / "01_gsflow" / "spinup_2000.control"
    ctl_analysis = HERE.parent / "01_gsflow" / "analysis_2001.control"

    ag_nc = domain_dir / "ag_frac.nc"
    ag_param = domain_dir / "ag_frac.param"

    # ---- Year 1: spinup (2000) -> writes restart ----
    control = load_control(
        ctl_spinup,
        domain_dir,
        out_spinup,
        iter_aet_flag=False,
        processes=WU_OL_PROCESSES,
        restart_write=str(restart_dir),
    )
    params = load_params(control, domain_dir)
    if ag_param.exists():  # ensure static .nc (not dynamic) is used
        ag_param.unlink()
    write_static_ag_frac(domain_dir, params)
    try:
        print("=== Year 1 spinup (2000): PRMSSoilzoneAg, writes restart ===")
        model = pws.Model(WU_OL_PROCESSES, control=control, parameters=params)
        model.run(finalize=True)
    finally:
        if ag_nc.exists():
            ag_nc.unlink()

    # ---- Year 2: analysis (2001) -> reads restart ----
    control = load_control(
        ctl_analysis,
        domain_dir,
        out_analysis,
        iter_aet_flag=True,
        processes=WU_ANALYSIS_PROCESSES,
        restart_read=str(restart_dir),
    )
    params = load_params(control, domain_dir)
    if ag_nc.exists():  # ensure dynamic .param (not static) is used
        ag_nc.unlink()
    shutil.copy2(domain_dir / "dyn_ag_frac.param", ag_param)
    try:
        print(
            "=== Year 2 analysis (2001): PRMSSoilzoneAgObsET, reads restart"
            " ==="
        )
        model = pws.Model(
            WU_ANALYSIS_PROCESSES, control=control, parameters=params
        )
        model.run(finalize=True)
    finally:
        if ag_param.exists():
            ag_param.unlink()

    print("Done. Outputs in output_spinup/ and output_analysis/.")


if __name__ == "__main__":
    main()
