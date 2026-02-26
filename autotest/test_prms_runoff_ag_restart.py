"""Restart tests for PRMSRunoffAg.

This test module validates the restart capability of PRMSRunoffAg by
confirming that a continuous run produces identical results to a run that
was restarted from a saved state.

Test Domains
------------
Two test domains are used (configured via pytest fixtures):

1. **ucb_ag_spinup_2yr** (open-loop): Tests restart with basic dual-area
   runoff calculations without iterative AET matching.

2. **ucb_ag_analysis_2yr** (closed-loop): Tests restart with full model
   including observed AET data.

Test Strategy
-------------
Perfect restart test:
- Run 1 "ac": Starts at time a, writes restart at time b, ends at time c
- Run 2 "bc": Starts at time b (reading restart), ends at time c'
- Confirm all variables match: c == c' (bit-for-bit)

Restart Variables
-----------------
PRMSRunoffAg inherits get_restart_variables() from PRMSRunoff without
modification because it has no additional storage state variables. The
agricultural-specific outputs (infil_ag, infil_ag_hru, hru_sroff_ag) are
flux variables that are recalculated each timestep, not state variables
that need to be saved for restart.

The ag calculations use:
- ag_soil_moist_prev and ag_soil_rechr_prev: These come from PRMSSoilzoneAg
  as inputs to PRMSRunoffAg, not internal runoff state
- ag_area: Derived from the ag_frac parameter and recalculated as needed

All storage states (imperv_stor, dprst_vol_open, dprst_vol_clos, etc.) are
inherited from the parent class and properly saved/restored during restart.

Why Not Extend test_nhm_restart.py?
------------------------------------
This test is separate from test_nhm_restart.py because PRMSRunoffAg requires
process-specific handling that would complicate the generic NHM test:

1. Domain differences: test_nhm_restart.py assumes "nhm" configuration with
   1979-1980 time ranges, while PRMSRunoffAg uses ag-specific domains
   (ucb_ag_spinup_2yr, ucb_ag_analysis_2yr) with different time ranges.

2. Input handling: PRMSRunoffAg requires special handling for ag_frac
   (dynamic vs static parameter) and intcp_changeover_in_net_rain flag
   (based on GSFLOW/PRMS), which doesn't fit the simple input gathering
   pattern used in test_nhm_restart.py.

3. Separation of concerns: Keeping ag-specific restart tests separate
   maintains clarity and follows the pattern established by
   test_prms_soilzone_ag_restart.py.

"""

import pathlib as pl
from typing import Any, Optional

import numpy as np
import pytest

from pywatershed.base.adapter import adapter_factory
from pywatershed.base.control import Control
from pywatershed.base.timeseries import TimeseriesArray
from pywatershed.hydrology.prms_runoff_ag import PRMSRunoffAg
from pywatershed.parameters import Parameters, PrmsParameters

dt_1d = np.timedelta64(24, "h")

imbalance_behavior = "error"

# Restart frequency options and corresponding init times
# The "a", "b", "c" times for each restart frequency
# Restarts with "y" and "m" are always written on the last day of the period.
# The "f" option is tested separately since it writes restarts only at the end.


# For analysis domain (starts 2000-01-01)
init_times_dict: dict[str, dict[str, np.datetime64]] = {
    "d": {
        "a": np.datetime64("2000-12-20") - dt_1d,
        "b": np.datetime64("2000-12-25") - dt_1d,
        "c": np.datetime64("2000-12-31") - dt_1d,
    },
    "m": {
        "a": np.datetime64("2000-10-31") - dt_1d,
        "b": np.datetime64("2000-11-01") - dt_1d,
        "c": np.datetime64("2000-12-01") - dt_1d,
    },
    "y": {
        "a": np.datetime64("2000-12-31") - dt_1d,
        "b": np.datetime64("2001-01-01") - dt_1d,
        "c": np.datetime64("2001-12-31") - dt_1d,
    },
}

restart_freqs: list[str] = list(init_times_dict.keys())


def get_control(
    simulation: dict[str, Any],
    init_time: Optional[np.datetime64] = None,
    end_time: Optional[np.datetime64] = None,
) -> Control:
    """Load control and optionally adjust time bounds."""
    import warnings

    domain_name = simulation["name"].split(":")[0]
    if "fgr_ag_2yr" not in domain_name:
        pytest.skip("Only running restart test for fgr_ag_2yr")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        control = Control.load_prms(
            simulation["control_file"],
            warn_unused_options=False,
            keep_unused_options=True,
        )

    # Check if this is GSFLOW to set intcp_changeover_in_net_rain flag
    if "executable_desc" in control.options.keys():
        exe_desc = control.options["executable_desc"][0].lower()
    else:
        exe_desc = "prms"

    if "gsflow" not in exe_desc:
        pytest.skip(
            "Only testing PRMSSoilzoneAg for domains run with a GSFLOW exe."
        )

    if "gsflow" in exe_desc:
        control.options["intcp_changeover_in_net_rain"] = True

    else:
        control.options["intcp_changeover_in_net_rain"] = False

    if init_time is not None:
        control.edit_init_start_times(init_time)
    if end_time is not None:
        control.edit_end_time(end_time)

    return control


@pytest.fixture(scope="function")
def discretization(simulation: dict[str, Any]) -> Parameters:
    dis_hru_file = simulation["dir"] / "parameters_dis_hru.nc"
    return Parameters.from_netcdf(dis_hru_file, encoding=False)


@pytest.fixture(scope="function")
def parameters(simulation: dict[str, Any]) -> PrmsParameters:
    control = get_control(simulation)
    param_file = simulation["dir"] / control.options["parameter_file"]
    params = PrmsParameters.load(param_file)

    # Skip if sat_threshold is not high enough (causes dunnian_flow)
    if abs(params.parameters["sat_threshold"]).min() < 999.0:
        pytest.skip(
            "test_prms_runoff_ag_restart only valid when sat_threshold >= 999"
        )

    return params


def get_input_variables(
    simulation: dict[str, Any],
    control: Control,
    parameters: PrmsParameters,
) -> dict[str, Any]:
    """Build input variables dict for PRMSRunoffAg."""
    output_dir = simulation["output_dir"]

    input_variables: dict[str, Any] = {}
    for key in PRMSRunoffAg.get_inputs():
        if key in ["ag_frac"]:
            # Check for dynamic parameter file first
            dyn_ag_frac_file = simulation["dir"] / "dyn_ag_frac.param"
            if dyn_ag_frac_file.exists():
                nc_pth = dyn_ag_frac_file
            else:
                nc_pth = adapter_factory(
                    parameters.parameters[key].copy(),
                    key,
                    control,
                )
        else:
            nc_pth = output_dir / f"{key}.nc"

        input_variables[key] = nc_pth

    return input_variables


@pytest.mark.parametrize("restart_freq", restart_freqs)
def test_restart(
    simulation: dict[str, Any],
    discretization: Parameters,
    parameters: PrmsParameters,
    tmp_path: pl.Path,
    restart_freq: str,
) -> None:
    """Perfect restart test for PRMSRunoffAg.

    Test outline/goals:
    run 1, "ac": a -----> c starts at a, restart written at time b, ends at c
                     |
    run 2, "bc":     b -> c'
    confirm c == c' in all variables (bit-for-bit match).
    """
    times = init_times_dict[restart_freq]

    restart_dir = tmp_path / "restarts"

    # Run ac: continuous run from a to c, writing restart at b
    control_ac = get_control(simulation, times["a"], times["c"])
    input_variables = get_input_variables(simulation, control_ac, parameters)

    # Get intcp_changeover_in_net_rain flag
    intcp_changeover_in_net_rain = control_ac.options.get(
        "intcp_changeover_in_net_rain", False
    )

    run_args: dict[str, Any] = {
        "control": control_ac,
        "discretization": discretization,
        "parameters": parameters,
        **input_variables,
        "imbalance_behavior": imbalance_behavior,
    }
    run_args["restart_write"] = restart_dir
    run_args["restart_write_freq"] = restart_freq
    run_args["intcp_changeover_in_net_rain"] = intcp_changeover_in_net_rain

    proc_ac = PRMSRunoffAg(**run_args)

    for istep in range(control_ac.n_times):
        control_ac.advance()
        proc_ac.advance()
        proc_ac.calculate(float(istep))
        proc_ac.output()

    proc_ac.finalize()

    # Run bc: restart from b to c
    control_bc = get_control(simulation, times["b"], times["c"])
    input_variables = get_input_variables(simulation, control_bc, parameters)

    intcp_changeover_in_net_rain = control_bc.options.get(
        "intcp_changeover_in_net_rain", False
    )

    run_args = {
        "control": control_bc,
        "discretization": discretization,
        "parameters": parameters,
        **input_variables,
        "imbalance_behavior": imbalance_behavior,
    }
    run_args["restart_read"] = restart_dir
    run_args["intcp_changeover_in_net_rain"] = intcp_changeover_in_net_rain

    proc_bc = PRMSRunoffAg(**run_args)

    for istep in range(control_bc.n_times):
        control_bc.advance()
        proc_bc.advance()
        proc_bc.calculate(float(istep))

    proc_bc.finalize()

    # Verify both runs ended at the same time
    assert control_ac.current_time == control_bc.current_time, (
        f"End times don't match: {control_ac.current_time} vs "
        f"{control_bc.current_time}"
    )

    # Compare all variables - should be bit-for-bit identical
    for vv in proc_ac.variables:
        ac_result = proc_ac[vv]
        bc_result = proc_bc[vv]

        if isinstance(ac_result, TimeseriesArray):
            ac_result = ac_result.current
            bc_result = bc_result.current

        np.testing.assert_equal(
            ac_result,
            bc_result,
            err_msg=f"Variable {vv} differs between continuous and "
            f"restarted runs at time {control_bc.current_time}",
        )


def test_restart_f(
    simulation: dict[str, Any],
    discretization: Parameters,
    parameters: PrmsParameters,
    tmp_path: pl.Path,
) -> None:
    """Test the "f" restart frequency option for PRMSRunoffAg.

    The "f" option writes restarts only at the end of a run, so it requires
    a different test structure:
    run 1, "ac": a ------> c starts at a, ends at c
    run 2, "ab": a -> b'
                      | restart files written at end
    run 3, "bc":      b -> c'
    confirm c == c' in all variables.
    """
    # Use fixed times that work for both domains
    init_times = {
        "a": np.datetime64("2000-06-01"),
        "b": np.datetime64("2000-06-15"),
        "c": np.datetime64("2000-06-30"),
    }

    restart_dir = tmp_path / "restarts"

    def run_init_end(
        init_time: np.datetime64,
        end_time: np.datetime64,
        restart_write: bool = False,
        restart_read: bool = False,
    ) -> PRMSRunoffAg:
        """Helper to run PRMSRunoffAg for a time period."""
        control = get_control(simulation, init_time, end_time)
        input_variables = get_input_variables(simulation, control, parameters)

        intcp_changeover_in_net_rain = control.options.get(
            "intcp_changeover_in_net_rain", False
        )

        run_args: dict[str, Any] = {
            "control": control,
            "discretization": discretization,
            "parameters": parameters,
            **input_variables,
            "imbalance_behavior": imbalance_behavior,
        }
        run_args["intcp_changeover_in_net_rain"] = intcp_changeover_in_net_rain

        if restart_write:
            run_args["restart_write"] = restart_dir
            run_args["restart_write_freq"] = "f"

        if restart_read:
            run_args["restart_read"] = restart_dir

        proc = PRMSRunoffAg(**run_args)

        for istep in range(control.n_times):
            control.advance()
            proc.advance()
            proc.calculate(float(istep))
            proc.output()

        proc.finalize()
        return proc

    # Run ac: continuous from a to c
    proc_ac = run_init_end(init_times["a"], init_times["c"])

    # Run ab: from a to b, write restart at end
    _ = run_init_end(init_times["a"], init_times["b"], restart_write=True)

    # Run bc: from b to c, reading restart
    proc_bc = run_init_end(init_times["b"], init_times["c"], restart_read=True)

    # Verify both runs ended at the same time
    assert proc_ac.control.current_time == proc_bc.control.current_time, (
        f"End times don't match: {proc_ac.control.current_time} vs "
        f"{proc_bc.control.current_time}"
    )

    # Compare all variables - should be bit-for-bit identical
    for vv in proc_ac.variables:
        ac_result = proc_ac[vv]
        bc_result = proc_bc[vv]

        if isinstance(ac_result, TimeseriesArray):
            ac_result = ac_result.current
            bc_result = bc_result.current

        np.testing.assert_equal(
            ac_result,
            bc_result,
            err_msg=f"Variable {vv} differs between continuous and "
            f"restarted runs at time {proc_bc.control.current_time}",
        )
