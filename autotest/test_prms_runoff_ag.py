import pathlib as pl

import pytest
from utils_compare import compare_in_memory, compare_netcdfs

from pywatershed.base.adapter import adapter_factory
from pywatershed.base.control import Control
from pywatershed.hydrology.prms_runoff_ag import PRMSRunoffAg
from pywatershed.parameters import Parameters, PrmsParameters

# compare in memory (faster) or full output files? or both!
do_compare_output_files = True
do_compare_in_memory = True

# Default tolerances for most variables
default_rtol = 1.0e-5
default_atol = 1.0e-5

# Variable-specific tolerance exceptions
var_tolerance_exceptions = {
    "sroff_vol": {"atol": 1.0e-4, "rtol": 5.0e-5},
    "dprst_vol_open": {"atol": 3.0e-4, "rtol": 3.0e-4},
}

calc_methods = ("numba", "numpy")
params = ("params_sep", "params_one")


@pytest.fixture(scope="function")
def control(simulation):
    control = Control.load_prms(
        simulation["control_file"],
        warn_unused_options=False,
        keep_unused_options=True,
    )

    return control


@pytest.fixture(scope="function")
def discretization(simulation):
    dis_hru_file = simulation["dir"] / "parameters_dis_hru.nc"
    return Parameters.from_netcdf(dis_hru_file, encoding=False)


@pytest.fixture(scope="function", params=params)
def parameters(simulation, control, request):
    if request.param == "params_one":
        param_file = simulation["dir"] / control.options["parameter_file"]
        params = PrmsParameters.load(param_file)
        sat_threshold = params.parameters["sat_threshold"]

    else:
        # Load runoff parameters
        param_file = simulation["dir"] / "parameters_PRMSRunoffAg.nc"
        params = PrmsParameters.from_netcdf(param_file)

    if abs(params.parameters["sat_threshold"]).min() < 999.0:
        pytest.skip(
            "test_prms_runoff_ag only valid when sat_threshold >= 999 (or some "
            "amount) which causes zero dunnian_flow"
        )

    return params


@pytest.mark.domain
@pytest.mark.parametrize("calc_method", calc_methods)
def test_compare_prms(
    simulation,
    control,
    discretization,
    parameters,
    tmp_path,
    calc_method,
):
    tmp_path = pl.Path(tmp_path)

    # Skip if not an ag domain
    if "ag" not in simulation["name"].lower():
        pytest.skip("test_prms_runoff_ag only valid for ag domains")

    comparison_var_names = set(PRMSRunoffAg.get_variables()) - {
        "dprst_vol_thres_open",  # not output by fortran nor post-processed
        "infil_ag_hru",  # currently not post-processed but infil_ag is
        "hru_sroff_ag",  # PRMS does can not write this variable
        "sroff_vol",  # errors for large HRUs, rely on sroff
        "intcp_changeover_budget",  # not a PRMS/GSFLOW variable
    }

    control.options["netcdf_output_var_names"] = comparison_var_names
    intcp_changeover_in_net_rain = (
        "gsflow" in control.options["executable_desc"][0].lower()
    )

    output_dir = simulation["output_dir"]

    input_variables = {}
    for key in PRMSRunoffAg.get_inputs():
        if key in ["ag_frac"]:
            dyn_ag_frac_flag = control.options.get("dyn_ag_frac_flag", False)
            # Check for dynamic parameter file first
            if dyn_ag_frac_flag:
                nc_pth = (
                    simulation["dir"] / control.options["ag_frac_dynamic"][0]
                )
            else:
                nc_pth = None

        else:
            nc_pth = output_dir / f"{key}.nc"

        # <
        input_variables[key] = nc_pth

    if do_compare_output_files:
        nc_parent = tmp_path / simulation["name"]
        control.options["netcdf_output_dir"] = nc_parent

    runoff = PRMSRunoffAg(
        control=control,
        discretization=discretization,
        parameters=parameters,
        **input_variables,
        imbalance_behavior="error",
        calc_method=calc_method,
        intcp_changeover_in_net_rain=intcp_changeover_in_net_rain,
    )

    if do_compare_output_files:
        runoff.initialize_netcdf()
        # test that init netcdf twice raises a warning
        with pytest.warns(UserWarning):
            runoff.initialize_netcdf()

    if do_compare_in_memory:
        answers = {}
        for var in comparison_var_names:
            var_pth = output_dir / f"{var}.nc"
            answers[var] = adapter_factory(
                var_pth, variable_name=var, control=control
            )

    for istep in range(control.n_times):
        control.advance()
        runoff.advance()
        runoff.calculate(1.0)
        runoff.output()

        if do_compare_in_memory:
            for var in answers.values():
                var.advance()

            # Build variable-specific tolerances: default for all,
            # then apply exceptions
            var_tolerances = {
                var: {"rtol": default_rtol, "atol": default_atol}
                for var in answers.keys()
            }
            for var, tols in var_tolerance_exceptions.items():
                if var in var_tolerances:
                    var_tolerances[var] = tols

            compare_in_memory(
                runoff,
                answers,
                atol=default_atol,
                rtol=default_rtol,
                var_tolerances=var_tolerances,
                skip_missing_ans=True,
                fail_after_all_vars=True,
            )

    runoff.finalize()

    if do_compare_output_files:
        # Filter out variables without answer files
        vars_with_answers = []
        for var in comparison_var_names:
            var_pth = output_dir / f"{var}.nc"
            if var_pth.exists():
                vars_with_answers.append(var)

        # Build variable-specific tolerances
        var_tolerances = {
            var: {"rtol": default_rtol, "atol": default_atol}
            for var in vars_with_answers
        }
        for var, tols in var_tolerance_exceptions.items():
            if var in var_tolerances:
                var_tolerances[var] = tols

        compare_netcdfs(
            vars_with_answers,
            tmp_path / simulation["name"],
            output_dir,
            atol=default_atol,
            rtol=default_rtol,
            var_tolerances=var_tolerances,
            # fail_after_all_vars=False,
            verbose=True,
        )

    return
