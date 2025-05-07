import pathlib as pl

import numpy as np
import xarray as xr

import pywatershed as pws


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


def assert_allclose(
    actual: np.ndarray,
    desired: np.ndarray,
    var_name: str = "",
    time_step: np.datetime64 = np.datetime64("NaT"),
    rtol: float = 1.0e-15,
    atol: float = 1.0e-15,
    equal_nan: bool = True,
    strict: bool = False,
    also_check_w_np: bool = True,
    print_max_errs: bool = False,
    verbosity: int = 0,
):
    """Reinvent np.testing.assert_allclose to get useful diagnostincs in debug

    Args:
        actual: Array obtained.
        desired: Array desired.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        equal_nan: If True, NaNs will compare equal.
        strict: If True, raise an ``AssertionError`` when either the shape or
            the data type of the arguments does not match. The special
            handling of scalars mentioned in the Notes section is disabled.
        also_check_w_np: first check using np.testing.assert_allclose using
            the same options.
        print_max_errs: bool=False. Print max abs and rel err for each var.
    """

    if also_check_w_np:
        np.testing.assert_allclose(
            actual,
            desired,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            # strict=strict,  # to add to newer versions of numpy
        )

    if strict:
        assert actual.shape == desired.shape
        assert isinstance(actual, type(desired))
        assert isinstance(desired, type(actual))

    if equal_nan:
        actual_nan = np.where(np.isnan(actual), True, False)
        desired_nan = np.where(np.isnan(desired), True, False)
        assert (actual_nan == desired_nan).all()
        if len(actual_nan) == len(actual):
            return

    abs_diff = abs(actual - desired)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_abs_diff = abs_diff / desired

    abs_close = abs_diff < atol
    rel_close = rel_abs_diff < rtol
    rel_close = np.where(np.isnan(rel_close), False, rel_close)
    close = abs_close | rel_close

    # an alternate implementation of the above used previously
    # success_a = np.isclose(a2, a1, atol=tol, rtol=0.0)
    # success_r = np.isclose(a2, a1, atol=0.0, rtol=tol)
    # success = success_a | success_r

    # assert close.all()
    success = close.all()
    if not success:
        diff = actual - desired
        diffmin = diff.min()
        diffmax = diff.max()
        if verbosity > 0:
            print(f"{time_step=}")
            print(f"{var_name=}")
            print(f"{desired.min()=}, {desired.max()=}")
            print(f"{actual.min()=}. {actual.max()=}")
            print(f"{diffmin=}, {diffmax=}")

            if verbosity > 0:
                idx = np.where(~close)
                for ii in idx:
                    print(f"{ii=}:  {desired[ii]=}, {actual[ii]=} {diff[ii]=}")
    # <
    elif verbosity > 0:
        print(f"success for {var_name}")

    assert success
    return None


def compare_in_memory(
    process: pws.base.Process,
    answers: dict[pws.base.adapter.AdapterNetcdf],
    rtol: float = 1.0e-15,
    atol: float = 1.0e-15,
    equal_nan: bool = True,
    strict: bool = False,
    also_check_w_np: bool = True,
    skip_missing_ans: bool = False,
    fail_after_all_vars: bool = True,
    verbosity: int = 0,
) -> None:
    # TODO: docstring
    # rename to "compare_process_in_memory"?
    # why not "compare_model_in_memory" and loop over all model processes?
    # could have both with the later just a wrapper on the former.

    fail_list = []
    for var in process.get_variables():
        if var not in answers.keys():
            if skip_missing_ans:
                continue
            else:
                msg = f"Variable '{var}' not found in the answers provided."
                raise KeyError(msg)

        if verbosity > 0:
            print(f"checking {var}")

        if not isinstance(answers[var], np.ndarray):
            answers[var].advance()

        if isinstance(process[var], pws.base.timeseries.TimeseriesArray):
            actual = process[var].current
        else:
            actual = process[var]

        if isinstance(answers[var], pws.base.adapter.Adapter):
            desired = answers[var].current
        else:
            desired = answers[var]

        if hasattr(process, "_active_hru_mask"):
            actual = actual[process._active_hru_mask]
            desired = desired[process._active_hru_mask]

        if not fail_after_all_vars:
            assert_allclose(
                actual,
                desired,
                atol=atol,
                rtol=rtol,
                equal_nan=equal_nan,
                strict=strict,
                also_check_w_np=also_check_w_np,
                var_name=var,
                time_step=process.control.current_time,
            )

        else:
            try:
                assert_allclose(
                    actual,
                    desired,
                    atol=atol,
                    rtol=rtol,
                    equal_nan=equal_nan,
                    strict=strict,
                    also_check_w_np=also_check_w_np,
                    var_name=var,
                    time_step=process.control.current_time,
                )
                if verbosity > 0:
                    print(f"compare_netcdfs all close for variable: {var}")

            except AssertionError:
                fail_list += [var]
                if verbosity > 0:
                    print(f"compare_netcdfs NOT all close for variable: {var}")

    if len(fail_list) > 0:
        assert False, f"compare_netcdfs failed for variables: {fail_list}"

    # <
    return None


def compare_model_in_memory(
    model: pws.base.Model,
    answers: dict,
    rtol: dict,
    atol: dict,
    equal_nan: bool = True,
    strict: bool = False,
    also_check_w_np: bool = True,
    skip_missing_ans: bool = False,
    fail_after_all_vars: bool = True,
    verbosity: int = 0,
) -> None:
    """Compare model values to answer dictionaries.

    Parameters:
        model: a pws.Model object.
        answers: A dictionary where each process name contains a dictionary of
          variable names with answers.
        atol: A dictionary of tolerances to use for each process.
        rtol: As for atol.
    """

    for process_name in model.processes.keys():
        if process_name not in answers.keys():
            continue

        # <
        _ = compare_in_memory(
            process=model.processes[process_name],
            answers=answers[process_name],
            atol=atol[process_name],
            rtol=rtol[process_name],
            equal_nan=equal_nan,
            strict=strict,
            also_check_w_np=also_check_w_np,
            skip_missing_ans=skip_missing_ans,
            fail_after_all_vars=fail_after_all_vars,
            verbosity=verbosity,
        )


def compare_netcdfs(
    var_list: list,
    results_dir: pl.Path,
    answers_dir: pl.Path,
    rtol: float = 1.0e-15,
    atol: float = 1.0e-15,
    equal_nan: bool = True,
    strict: bool = False,
    also_check_w_np: bool = True,
    print_var_max_errs: bool = False,
    fail_after_all_vars: bool = True,
    verbosity: int = 0,
):
    # TODO: docstring
    # TODO: improve error message
    # TODO: collect failures in a try and report at end
    fail_list = []
    for var in var_list:
        answer = xr.open_dataarray(
            answers_dir / f"{var}.nc", decode_timedelta=False
        )
        result = xr.open_dataarray(
            results_dir / f"{var}.nc", decode_timedelta=False
        )

        if not fail_after_all_vars:
            assert_allclose(
                actual=result.values,
                desired=answer.values,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
                strict=strict,
                also_check_w_np=also_check_w_np,
                print_max_errs=print_var_max_errs,
                var_name=var,
            )
            if verbosity > 0:
                print(f"compare_netcdfs all close for variable: {var}")

        else:
            try:
                assert_allclose(
                    actual=result.values,
                    desired=answer.values,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=equal_nan,
                    strict=strict,
                    also_check_w_np=also_check_w_np,
                    print_max_errs=print_var_max_errs,
                    var_name=var,
                )
                if verbosity > 0:
                    print(f"compare_netcdfs all close for variable: {var}")

            except AssertionError:
                fail_list += [var]
                if verbosity > 0:
                    print(f"compare_netcdfs NOT all close for variable: {var}")

    if len(fail_list) > 0:
        assert False, f"compare_netcdfs failed for variables: {fail_list}"
