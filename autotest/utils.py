import pathlib as pl
from types import MappingProxyType
from typing import Union

import numpy as np

print_ans = False


def assert_or_print(
    results,
    answers,
    test_name: str = None,
    print_ans: bool = False,
    close: bool = False,
):
    if test_name is not None:
        test_name = f"{test_name}:"
    n_space = 4
    if print_ans and (test_name is not None):
        print("\n")
        sp = "".join(n_space * [" "])
        n_space = n_space + 2
        print(f"{sp}{test_name}")

    # Always check every test in the answers
    for key in answers.keys():
        if print_ans:
            sp = "".join(n_space * [" "])
            print(f"{sp}{key}: {results[key]}")
        else:
            if close:
                np.testing.assert_allclose(
                    results[key], answers[key], err_msg=f"{test_name}{key}"
                )
            else:
                np.testing.assert_equal(
                    results[key], answers[key], err_msg=f"{test_name}{key}"
                )
            # if isinstance(results[key], (np.datetime64, np.timedelta64)):
            #    assert results[key] == answers[key], msg
            # else:
            #    assert np.isclose(results[key], answers[key]), msg

    # Always fail if printing answers
    assert not print_ans
    return


def assert_dicts_equal(dic1, dic2):
    assert set(dic1.keys()) == set(dic2.keys())

    # add cases as needed
    for kk, vv in dic1.items():
        if isinstance(vv, (dict, MappingProxyType)):
            assert_dicts_equal(dic1[kk], dic2[kk])
        elif isinstance(vv, np.ndarray):
            np.testing.assert_equal(dic1[kk], dic2[kk])
        else:
            if (
                isinstance(vv, float)
                and np.isnan(dic1[kk])
                and np.isnan(dic2[kk])
            ):
                continue
            assert dic1[kk] == dic2[kk]


def detect_prms_exe():
    import sys
    from platform import processor

    platform = sys.platform.lower()
    if platform == "win32":
        exe_name = "prms_win_gfort_dbl_prec.exe"
    elif platform == "darwin":
        if processor() == "arm":
            exe_name = "prms_mac_m1_ifort_dbl_prec"
        else:
            exe_name = "prms_mac_intel_gfort_dbl_prec"
    elif platform == "linux":
        exe_name = "prms_linux_gfort_dbl_prec"
    else:
        exe_name = "---"  # this will raise an error
    exe_pth = pl.Path(f"../bin/{exe_name}")
    return exe_pth


def run_prms(
    control_file: pl.Path,
    run_dir: Union[pl.Path, None] = None,
) -> None:
    import shutil

    from flopy import run_model

    from pywatershed import Control

    exe_path = detect_prms_exe()
    if run_dir is None:
        run_dir = control_file.parent

    control = Control.load_prms(control_file, warn_unused_options=False)
    output_dir = run_dir / control.options["netcdf_output_dir"]
    # delete the existing output dir and re-create it
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    print(
        f"Running '{control_file.name}' in {run_dir}\n\n",
        flush=True,
    )
    # the command to run the model looks like this
    # exe control_file -MAXDATALNLEN 60000
    success, buff = run_model(
        exe_path,
        control_file,
        model_ws=run_dir,
        cargs=[
            "-MAXDATALNLEN",
            "60000",
        ],
        normal_msg="Normal completion of PRMS",
    )

    if not success:
        raise RuntimeError(f"PRMS failed to run: {control_file.name}")

    return None
