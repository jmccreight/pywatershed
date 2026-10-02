import contextlib
import os
import pathlib as pl
import shutil

from flopy import run_model

import pywatershed as pws

"""This module contains functions for running PRMS simulations.

This module provides functions for running PRMS simulations using the
pywatershed library. A control whose name contains "make_cbh_only"
generates the domain's CBH files (see test_run_prms).

"""


def test_exe_available(exe):
    assert exe.is_file(), f"'{exe}'...does not exist"
    assert os.access(exe, os.X_OK)
    print(f"\nUsing exectuable: {exe}\n")


def test_run_prms(simulation, exe):
    ws = pl.Path(simulation["ws"])

    # sagehen_gridded_5yr ships no forcing files: its 5609-cell CBH text
    # files (precip/tmax/tmin.day, ~200 MB each) are too large to
    # distribute, so PRMS generates them from the two-station sagehen.data
    # using the *_make_cbh_only.control (model_mode WRITE_CLIMATE). That
    # run then converts them to the netCDF files pywatershed reads
    # (prcp/tmax/tmin.nc, ~9 MB each, also gitignored). Every other control
    # for the domain reads the .day files, so the make_cbh_only run must
    # come first; below we either generate or require them.
    domain_dir_name = simulation["control_file"].parent.name
    control_file_name = simulation["control_file"].stem
    requires_cbh = domain_dir_name in ("sagehen_gridded_5yr",)
    run_cbh = requires_cbh and "make_cbh_only" in control_file_name
    if requires_cbh and not run_cbh:
        missing_cbh = [
            f"{var}.nc"
            for var in ["prcp", "tmax", "tmin"]
            if not (ws / f"{var}.nc").exists()
        ]
        if missing_cbh:
            msg = (
                "Input CBH files are missing for the domain "
                f"{domain_dir_name}: {missing_cbh}.\n"
                "Run the simulation for generating CBH files first."
            )
            raise IOError(msg)

    control_file = simulation["control_file"]
    output_dir = simulation["output_dir"]
    print(f"\n\n\n{'*' * 70}\n{'*' * 70}")
    print(
        "run_domains.py: "
        f"Running '{exe} {control_file.name}  -MAXDATALNLEN 60000' "
        f"in {ws}\n\n",
        flush=True,
    )

    # delete the existing output dir and re-create it
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    if "prms" in str(exe):
        success_msg = "Normal completion of PRMS"
    else:
        # gsflow in this case
        success_msg = "Normal completion of GSFLOW"

    # the command to run the model looks like this
    # exe control_file -MAXDATALNLEN 60000
    success, buff = run_model(
        exe,
        control_file.name,
        model_ws=ws,
        cargs=["-MAXDATALNLEN", "60000"],
        report=True,
        normal_msg=success_msg,
    )

    if simulation["write_log"]:
        with open(f"{control_file}.log", "w") as file:
            for line in buff:
                file.write(line + "\n")

    assert success, f"could not run prms model in '{ws}'"

    if run_cbh:
        # if this run is generating CBH files, convert PRMS outputs to netcdf
        # need to be in ws
        with contextlib.chdir(ws):
            cbh_nc_dir = pl.Path(".")
            cbh_files = [
                pl.Path("precip.day"),
                pl.Path("tmax.day"),
                pl.Path("tmin.day"),
            ]
            rename_vars = {
                "precip": "prcp",
                "tmaxf": "tmax",
                "tmax": "tmax",
                "tminf": "tmin",
                "tmin": "tmin",
            }
            control = pws.Control.load_prms(simulation["control_file"])
            parameter_file = control.options["parameter_file"]
            params = pws.parameters.PrmsParameters.load(parameter_file)
            for cbh_file in cbh_files:
                out_file = cbh_nc_dir / (rename_vars[cbh_file.stem] + ".nc")
                pws.utils.cbh_file_to_netcdf(
                    cbh_file,
                    params,
                    out_file,
                    complevel=9,
                    rename_vars=rename_vars,
                )

            prms_output_to_rm = [
                "potet.day",
                "swrad.day",
                # currently using these files to drive the PRMS model
                # may change that so they can be removed
                # "precip.day",
                # "tmin.day",
                # "tmax.day",
                "transp.day",
            ]
            for ff in prms_output_to_rm:
                pl.Path(ff).unlink()

    print(f"run_domains.py: End of domain {ws}\n", flush=True)
