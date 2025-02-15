import pathlib as pl

import numpy as np
import pytest
import pywatershed as pws


@pytest.fixture(scope="function")
def control_ac(simulation):
    return pws.Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )


@pytest.fixture(scope="function")
def control_bc(simulation):
    control = pws.Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )
    control.edit_init_start_times(control.init_time + np.timedelta64(365, "D"))
    return control


@pytest.fixture(scope="function")
def discretization(simulation):
    dis_hru_file = simulation["dir"] / "parameters_dis_hru.nc"
    return pws.Parameters.from_netcdf(dis_hru_file, encoding=False)


@pytest.fixture(scope="function")
def parameters(simulation, control_ac, request):
    param_file = simulation["dir"] / control_ac.options["parameter_file"]
    return pws.parameters.PrmsParameters.load(param_file)


# parameterize this to test the individual processes separately
def test_restart(
    simulation,
    control_ac,
    control_bc,
    discretization,
    parameters,
    tmp_path,
):
    """Test outline/goals:
    run 1, "ac": a -----> c starts at a, restart written at time b, ends at c
                     |
    run 2, "bc":     b -> c'
    confirm c == c` in all variables. We'll just do that in memory.
    I am first implementing the restart read capability. Then I will implement
    the restart write capability. This test will be partially fake until
    the write capability is done but creating the restart file offline from
    output files in the ac run.

    """
    Process = pws.PRMSGroundwater
    # Process = pws.PRMSSolarGeometry  # works
    # Process = pws.PRMSAtmosphere  # works

    output_dir = simulation["output_dir"]
    input_variables = {
        kk: output_dir / f"{kk}.nc" for kk in Process.get_inputs()
    }
    # The "PRMS model" inputs are one level up
    if Process.__name__ == "PRMSAtmosphere":
        for kk, vv in input_variables.items():
            if not vv.exists():
                input_variables[kk] = output_dir.parent / f"{kk}.nc"

    # run ac
    proc_ac = Process(
        control_ac,
        discretization,
        parameters,
        **input_variables,
        # budget_type=None,
    )

    for istep in range(control_ac.n_times):
        control_ac.advance()
        proc_ac.advance()
        proc_ac.calculate(float(istep))

    proc_ac.finalize()

    # run bc
    proc_bc = Process(
        control_bc,
        discretization,
        parameters,
        **input_variables,
        # budget_type=None,
    )

    print(control_bc.init_time)
    for istep in range(control_bc.n_times):
        control_bc.advance()
        print(control_bc.current_time)
        proc_bc.advance()
        proc_bc.calculate(float(istep))

    proc_bc.finalize()

    assert control_ac.current_time == control_bc.current_time

    # compare the end result
    for vv in proc_ac.variables:
        ac_result = proc_ac[vv]
        bc_result = proc_bc[vv]
        if isinstance(proc_ac[vv], pws.TimeseriesArray):
            ac_result = ac_result.current
            bc_result = bc_result.current
        # <
        np.testing.assert_allclose(ac_result, bc_result)

    asdf
    pass
