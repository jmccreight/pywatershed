import os
import pathlib as pl
import shutil

from flopy import run_model


def test_exe_available(exe):
    assert exe.is_file(), f"'{exe}'...does not exist"
    assert os.access(exe, os.X_OK)


def test_run_prms(simulation, exe):
    ws = pl.Path(simulation["ws"])
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
        control_file,
        model_ws=ws,
        cargs=[
            "-MAXDATALNLEN",
            "60000",
        ],
        normal_msg=success_msg,
    )

    assert success, f"could not run prms model in '{ws}'"

    print(f"run_domains.py: End of domain {ws}\n", flush=True)
