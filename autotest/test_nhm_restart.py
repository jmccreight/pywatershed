import numpy as np
import pytest
import pywatershed as pws

# JLM TODO: could test the various restart freqs via a fixture


@pytest.fixture(scope="function")
def control_ac(simulation):
    control_name = simulation["name"].split(":")[1]
    if control_name != "nhm":
        pytest.skip("test_cbh_to_netcdf only for nhm configuration")
    control = pws.Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )
    control.options["budget_type"] = None
    return control


@pytest.fixture(scope="function")
def control_bc(simulation):
    control = pws.Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )
    control.edit_init_start_times(np.datetime64("1980-01-01"))
    control.options["budget_type"] = None
    return control


@pytest.fixture(scope="function")
def discretization(simulation):
    dis_hru_file = simulation["dir"] / "parameters_dis_hru.nc"
    return pws.Parameters.from_netcdf(dis_hru_file, encoding=False)


@pytest.fixture(scope="function")
def parameters(simulation, control_ac, request):
    param_file = simulation["dir"] / control_ac.options["parameter_file"]
    return pws.parameters.PrmsParameters.load(param_file)


nhm_processes = [
    pws.PRMSSolarGeometry,
    pws.PRMSAtmosphere,
    pws.PRMSCanopy,
    # pws.PRMSSnow,  #  what is going on here? hidden prognostic variables?
    # pws.PRMSRunoff,
    # pws.PRMSSoilzone,
    pws.PRMSGroundwater,
    pws.PRMSChannel,
]


@pytest.mark.parametrize("Process", nhm_processes)
def test_restart(
    simulation,
    control_ac,
    control_bc,
    discretization,
    parameters,
    Process,
    tmp_path,
):
    """Test outline/goals:
    run 1, "ac": a -----> c starts at a, restart written at time b, ends at c
                     |
    run 2, "bc":     b -> c'
    confirm c == c` in all variables. We'll just do that in memory.
    """
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
    run_args = {
        "control": control_ac,
        "discretization": discretization,
        "parameters": parameters,
        **input_variables,
    }
    # Solar and Atmosphere dont need restarts, these tests confirm: all diag
    if Process.__name__ not in ["PRMSSolarGeometry", "PRMSAtmosphere"]:
        run_args["restart_write"] = restart_dir
        run_args["restart_write_freq"] = "y"

    proc_ac = Process(**run_args)

    for istep in range(control_ac.n_times):
        control_ac.advance()
        proc_ac.advance()
        proc_ac.calculate(float(istep))
        proc_ac.output()

    proc_ac.finalize()

    # run bc
    run_args = {
        "control": control_bc,
        "discretization": discretization,
        "parameters": parameters,
        **input_variables,
    }
    if Process.__name__ not in ["PRMSSolarGeometry", "PRMSAtmosphere"]:
        run_args["restart_read"] = restart_dir
    proc_bc = Process(**run_args)

    for istep in range(control_bc.n_times):
        control_bc.advance()
        proc_bc.advance()
        proc_bc.calculate(float(istep))

    proc_bc.finalize()

    # make sure we all at c
    assert control_ac.current_time == control_bc.current_time

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
        # np.testing.assert_allclose(ac_result, bc_result)
        np.testing.assert_equal(ac_result, bc_result)
