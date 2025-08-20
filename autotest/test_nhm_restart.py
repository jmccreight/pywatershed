import numpy as np
import pytest

import pywatershed as pws

# this module/test can probably be renamed to drop "NHM" at some point. The
# NHM processes were just a convenient starting place for developing the
# restart capabilities.

dt_1d = np.timedelta64(24, "h")

# PRMS fails daily restart test starting on many days, including 1979-12-31
# as documented on nueva: ~/usgs/pywatershed/autotest/prms_restart_transp_on
# A potential fix was to simply promote transp_on to a restart variable. And
# that solves the restart test. Still pondering if that's appropriate in the
# bigger picture.
nhm_processes = [
    pws.PRMSSolarGeometry,
    pws.PRMSAtmosphere,
    pws.PRMSCanopy,
    # pws.PRMSSnow,  # not working/implemented
    # pws.PRMSRunoff,  # not working/implemented
    pws.PRMSSoilzone,
    pws.PRMSGroundwater,
    pws.PRMSChannel,
]

# The domain inputs start on 1979-01-01.
# These "a", "b", "c" in the restart test cant be achieved with the "f" option.
# So it is tested separately below. Below we show "start_times" and subtract a
# day to get init_times, since it's not always obvious what the last day of the
# month is. Restarts with "y" and "m" are always written on the last day of the
# period.
init_times_dict = {
    "d": {
        "a": np.datetime64("1979-12-31") - dt_1d,
        "b": np.datetime64("1980-01-01") - dt_1d,
        "c": np.datetime64("1980-01-02") - dt_1d,
    },
    "m": {
        "a": np.datetime64("1979-12-31") - dt_1d,
        "b": np.datetime64("1980-01-01") - dt_1d,
        "c": np.datetime64("1980-02-01") - dt_1d,
    },
    "y": {
        "a": np.datetime64("1979-12-31") - dt_1d,
        "b": np.datetime64("1980-01-01") - dt_1d,
        "c": np.datetime64("1980-12-31") - dt_1d,
    },
}
restart_freqs = list(init_times_dict.keys())


def get_control(simulation, init_time=None, end_time=None):
    control_name = simulation["name"].split(":")[1]
    if control_name != "nhm":
        pytest.skip("test_cbh_to_netcdf only for nhm configuration")
    control = pws.Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )
    control.options["budget_type"] = "error"
    if init_time is not None:
        control.edit_init_start_times(init_time)
    if end_time is not None:
        control.edit_end_time(end_time)
    return control


@pytest.fixture(scope="function")
def discretization(simulation):
    dis_hru_file = simulation["dir"] / "parameters_dis_hru.nc"
    return pws.Parameters.from_netcdf(dis_hru_file, encoding=False)


@pytest.fixture(scope="function")
def parameters(simulation):
    control = get_control(simulation)
    param_file = simulation["dir"] / control.options["parameter_file"]
    return pws.parameters.PrmsParameters.load(param_file)


@pytest.mark.parametrize("restart_freq", restart_freqs)
@pytest.mark.parametrize("Process", nhm_processes)
def test_restart(
    simulation,
    discretization,
    parameters,
    Process,
    tmp_path,
    restart_freq,
):
    """Perfect restart test.
    Test outline/goals:
    run 1, "ac": a -----> c starts at a, restart written at time b, ends at c
                     |
    run 2, "bc":     b -> c'
    confirm c == c` in all variables. We'll just do that in memory.
    """

    times = init_times_dict[restart_freq]

    output_dir = simulation["output_dir"]
    restart_dir = tmp_path / "restarts"

    input_variables = {
        kk: output_dir / f"{kk}.nc" for kk in Process.get_inputs()
    }
    # The "PRMS model" inputs, for PRMSAtmosphere, are one level up
    if Process.__name__ == "PRMSAtmosphere":
        for kk, vv in input_variables.items():
            if not vv.exists():
                input_variables[kk] = output_dir.parent / f"{kk}.nc"

    # run ac
    control_ac = get_control(simulation, times["a"], times["c"])
    run_args = {
        "control": control_ac,
        "discretization": discretization,
        "parameters": parameters,
        **input_variables,
    }
    run_args["restart_write"] = restart_dir
    run_args["restart_write_freq"] = restart_freq

    proc_ac = Process(**run_args)

    for istep in range(control_ac.n_times):
        control_ac.advance()
        proc_ac.advance()
        proc_ac.calculate(float(istep))
        proc_ac.output()

    proc_ac.finalize()

    # run bc
    control_bc = get_control(simulation, times["b"], times["c"])
    run_args = {
        "control": control_bc,
        "discretization": discretization,
        "parameters": parameters,
        **input_variables,
    }
    run_args["restart_read"] = restart_dir
    proc_bc = Process(**run_args)

    for istep in range(control_bc.n_times):
        control_bc.advance()
        proc_bc.advance()
        proc_bc.calculate(float(istep))

    proc_bc.finalize()

    # make sure we all at c
    assert control_ac.current_time == control_bc.current_time
    # print(f"Comparing states at time: {control_bc.current_time=}")

    # compare the end result that is in memory
    for vv in proc_ac.variables:
        ac_result = proc_ac[vv]
        bc_result = proc_bc[vv]
        if isinstance(proc_ac[vv], pws.TimeseriesArray):
            ac_result = ac_result.current
            bc_result = bc_result.current
        # <
        # TODO: just use equal, should be bit matched
        # Keep this around for checking Snow, Runoff, Soilzone

        if True:
            # np.testing.assert_allclose(ac_result, bc_result)
            np.testing.assert_equal(ac_result, bc_result)

        else:
            failed = False
            try:
                # np.testing.assert_allclose(ac_result, bc_result)
                np.testing.assert_equal(ac_result, bc_result)
            except AssertionError:
                failed = True
                print(f"FAILED: {vv}")

            if failed:
                raise AssertionError("Failed")


@pytest.mark.parametrize("Process", nhm_processes)
def test_restart_f(
    simulation,
    discretization,
    parameters,
    Process,
    tmp_path,
):
    """Test the "f" restart frequency option.
    The "f" option does not conform to the test above, because restarts can not
    be written at b when running from a->c. Since we can only write restarts at
    the end of the run
    Test outline/goals:
    run 1, "ac": a ------> c starts at a, ends at c
    run 2, "ab": a -> b'
                      | restart files
    run 3, "bc":      b -> c'
    confirm c == c` in all variables. We'll just do that in memory.
    """

    init_times = {
        "a": np.datetime64("1980-01-03"),
        "b": np.datetime64("1980-01-09"),
        "c": np.datetime64("1980-01-19"),
    }

    output_dir = simulation["output_dir"]
    restart_dir = tmp_path / "restarts"

    input_variables = {
        kk: output_dir / f"{kk}.nc" for kk in Process.get_inputs()
    }
    # The "PRMS model" inputs, for PRMSAtmosphere, are one level up
    if Process.__name__ == "PRMSAtmosphere":
        for kk, vv in input_variables.items():
            if not vv.exists():
                input_variables[kk] = output_dir.parent / f"{kk}.nc"

    def run_init_end(
        init_time, end_time, restart_write=False, restart_read=False
    ):
        # run ac
        control = get_control(simulation, init_time, end_time)
        run_args = {
            "control": control,
            "discretization": discretization,
            "parameters": parameters,
            **input_variables,
        }

        if restart_write:
            run_args["restart_write"] = restart_dir
            run_args["restart_write_freq"] = "f"  # test namesake

        if restart_read:
            run_args["restart_read"] = restart_dir

        proc = Process(**run_args)

        for istep in range(control.n_times):
            control.advance()
            proc.advance()
            proc.calculate(float(istep))
            proc.output()

        proc.finalize()
        return proc

    proc_ac = run_init_end(init_times["a"], init_times["c"])
    _ = run_init_end(init_times["a"], init_times["b"], restart_write=True)
    proc_bc = run_init_end(init_times["b"], init_times["c"], restart_read=True)

    # make sure we all at c
    assert proc_ac.control.current_time == proc_bc.control.current_time
    # print(f"Comparing states at time: {control_bc.current_time=}")

    # compare the end result that is in memory
    for vv in proc_ac.variables:
        ac_result = proc_ac[vv]
        bc_result = proc_bc[vv]
        if isinstance(proc_ac[vv], pws.TimeseriesArray):
            ac_result = ac_result.current
            bc_result = bc_result.current
        # <
        # TODO: just use equal, should be bit matched
        # Keep this around for checking Snow, Runoff, Soilzone

        if True:
            # np.testing.assert_allclose(ac_result, bc_result)
            np.testing.assert_equal(ac_result, bc_result)

        else:
            failed = False
            try:
                # np.testing.assert_allclose(ac_result, bc_result)
                np.testing.assert_equal(ac_result, bc_result)
            except AssertionError:
                failed = True
                print(f"FAILED: {vv}")

            if failed:
                raise AssertionError("Failed")
