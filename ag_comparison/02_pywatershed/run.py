#!/usr/bin/env python
"""Run 2: full pywatershed, exactly parallel to run 1 (GSFLOW).

A year-1 spinup and two year-2 analysis variants on the fgr_ag_2yr
domain, reusing run 1's controls:

  spinup (2000): PRMSRunoffAg + PRMSSoilzoneAg, iter_aet_flag=False,
    static ag_frac, static frost; writes a full-model restart at
    2000-12-31.
  analysis_dynamic_frost (2001): PRMSRunoffAg + PRMSSoilzoneAgObsET,
    iter_aet_flag=True, dynamic ag_frac, PRMSAtmosphereTranspFrostDynamic
    with the domain's fall_frost.dyn/spring_frost.dyn; reads the restart.
    This is the GSFLOW-faithful configuration (dyn_*frost_flag=1 in the
    GSFLOW analysis control) and the primary comparison against run 1.
  analysis_static_frost (2001): as above but with the static
    PRMSAtmosphereTranspFrost. Quantifies the effect of dynamic frost
    dates on the results.

Both analysis variants read the same year-1 restart: spinup is
static-frost in GSFLOW and pywatershed alike.

pywatershed sets init_time = start_time - time_step, so the year-2
control (start 2001-01-01) reads init_time 2000-12-31 — matching the
restart files year 1 writes on the year boundary (restart_write_freq="y").
Restart options set in control.options propagate to every process.

Models are built with the model-dict form of pws.Model, NOT the
process-list form: the list form orders classes via
pws.base.model.process_order_nhm, which does not contain the TranspFrost
class names (they are silently dropped from the model order), and the
dict form is how fall_frost_dyn/spring_frost_dyn are passed to the
atmosphere process (extra keys in a process spec pass through to its
__init__).

The ag_frac input follows examples/10_ag_irrigation_use.ipynb: write
files into the domain dir (static ag_frac.nc for spinup, dynamic
ag_frac.param for analysis), cleaned up in a finally block.

Run (phases default to all three, in order):

    python run.py [spinup analysis_dynamic_frost analysis_static_frost]

Outputs -> output_spinup/, output_analysis_dynamic_frost/,
output_analysis_static_frost/; restart files -> restart/.
"""

import argparse
import pathlib as pl
import shutil

import xarray as xr

import pywatershed as pws

HERE = pl.Path(__file__).resolve().parent

PHASES = ("spinup", "analysis_dynamic_frost", "analysis_static_frost")

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

WU_ANALYSIS_PROCESSES_STATIC = [
    pws.PRMSSolarGeometry,
    pws.PRMSAtmosphereTranspFrost,
    pws.PRMSCanopy,
    pws.PRMSSnow,
    pws.PRMSRunoffAg,
    pws.PRMSSoilzoneAgObsET,
    pws.PRMSGroundwater,
    pws.PRMSChannel,
]

WU_ANALYSIS_PROCESSES_DYNAMIC = [
    pws.PRMSSolarGeometry,
    pws.PRMSAtmosphereTranspFrostDynamic,
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


def build_model_dict(control, params, processes, atmosphere_kwargs=None):
    """Build the model-dict form of the model (see module docstring for
    why the process-list form is not used). atmosphere_kwargs are added
    to the atmosphere process spec, passing through to its __init__."""
    model_dict = {"control": control}
    order = []
    for cls in processes:
        name = cls.__name__
        spec = {"class": cls, "parameters": params}
        if atmosphere_kwargs is not None and "TranspFrost" in name:
            spec = spec | atmosphere_kwargs
        model_dict[name] = spec
        order.append(name)
    model_dict["model_order"] = order
    return model_dict


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


def run_spinup(domain_dir, restart_dir):
    """Year 1 (2000): PRMSSoilzoneAg, static ag_frac, writes restart."""
    out_dir = HERE / "output_spinup"
    out_dir.mkdir(exist_ok=True)
    control = load_control(
        HERE.parent / "01_gsflow" / "spinup_2000.control",
        domain_dir,
        out_dir,
        iter_aet_flag=False,
        processes=WU_OL_PROCESSES,
        restart_write=str(restart_dir),
    )
    params = load_params(control, domain_dir)

    ag_nc = domain_dir / "ag_frac.nc"
    ag_param = domain_dir / "ag_frac.param"
    if ag_param.exists():  # ensure static .nc (not dynamic) is used
        ag_param.unlink()
    write_static_ag_frac(domain_dir, params)
    try:
        print("=== spinup (2000): PRMSSoilzoneAg, writes restart ===")
        model_dict = build_model_dict(control, params, WU_OL_PROCESSES)
        model = pws.Model(model_dict)
        model.run(finalize=True)
    finally:
        if ag_nc.exists():
            ag_nc.unlink()


def run_analysis(domain_dir, restart_dir, frost):
    """Year 2 (2001): PRMSSoilzoneAgObsET, dynamic ag_frac, reads the
    spinup restart. frost is "dynamic" or "static"."""
    out_dir = HERE / f"output_analysis_{frost}_frost"
    out_dir.mkdir(exist_ok=True)

    if frost == "dynamic":
        processes = WU_ANALYSIS_PROCESSES_DYNAMIC
        atmosphere_kwargs = {
            "fall_frost_dyn": domain_dir / "fall_frost.dyn",
            "spring_frost_dyn": domain_dir / "spring_frost.dyn",
        }
    else:
        processes = WU_ANALYSIS_PROCESSES_STATIC
        atmosphere_kwargs = None

    control = load_control(
        HERE.parent / "01_gsflow" / "analysis_2001.control",
        domain_dir,
        out_dir,
        iter_aet_flag=True,
        processes=processes,
        restart_read=str(restart_dir),
    )
    params = load_params(control, domain_dir)

    ag_nc = domain_dir / "ag_frac.nc"
    ag_param = domain_dir / "ag_frac.param"
    if ag_nc.exists():  # ensure dynamic .param (not static) is used
        ag_nc.unlink()
    shutil.copy2(domain_dir / "dyn_ag_frac.param", ag_param)
    try:
        print(
            f"=== analysis (2001), {frost} frost: PRMSSoilzoneAgObsET, "
            "reads restart ==="
        )
        model_dict = build_model_dict(
            control, params, processes, atmosphere_kwargs
        )
        model = pws.Model(model_dict)
        model.run(finalize=True)
    finally:
        if ag_param.exists():
            ag_param.unlink()


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
    # run in canonical order regardless of argument order
    phases = [pp for pp in PHASES if pp in args.phases]

    domain_dir = pws.utils.get_addtl_domains_dir("fgr_ag_2yr")
    restart_dir = HERE / "restart"
    restart_dir.mkdir(exist_ok=True)

    if "spinup" in phases:
        run_spinup(domain_dir, restart_dir)
    if "analysis_dynamic_frost" in phases:
        run_analysis(domain_dir, restart_dir, frost="dynamic")
    if "analysis_static_frost" in phases:
        run_analysis(domain_dir, restart_dir, frost="static")

    print(f"Done: {', '.join(phases)}")


if __name__ == "__main__":
    main()
