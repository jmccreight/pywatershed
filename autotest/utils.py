from types import MappingProxyType

import numpy as np

from pywatershed.base.process import Process
from pywatershed.base.timeseries import TimeseriesArray

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


def check_timestep_results(
    process: Process,
    istep: int,
    ans: TimeseriesArray,
    tol: float,
    failfast: bool = False,
    verbose: bool = False,
):
    # print(process)
    all_success = True
    for key in ans.keys():
        # print(key)

        a1 = ans[key].current
        a2 = process[key]
        if not isinstance(a2, np.ndarray):
            a2 = a2.current

        if hasattr(process, "_active_hru_mask"):
            a1 = a1[process._active_hru_mask]
            a2 = a2[process._active_hru_mask]

        success_a = np.isclose(a2, a1, atol=tol, rtol=0.0)
        success_r = np.isclose(a2, a1, atol=0.0, rtol=tol)
        success = success_a | success_r
        if not success.all():
            all_success = False
            diff = a2 - a1
            diffmin = diff.min()
            diffmax = diff.max()
            if True:
                print(f"time step {istep}")
                print(f"output variable {key}")
                print(f"prms   {a1.min()}    {a1.max()}")
                print(f"pywatershed  {a2.min()}    {a2.max()}")
                print(f"diff   {diffmin}  {diffmax}")

                if verbose:
                    idx = np.where(~success)
                    for i in idx:
                        print(
                            f"hru {i} prms {a1[i]} pywatershed {a2[i]} "
                            f"diff {diff[i]}"
                        )
            if failfast:
                raise (ValueError)
        # <
        elif verbose:
            print(f"variable {key} matches")

    return all_success
