import pathlib as pl

import pytest
from utils_compare import compare_in_memory, compare_netcdfs

from pywatershed.base.adapter import adapter_factory
from pywatershed.base.control import Control
from pywatershed.hydrology.prms_runoff import PRMSRunoffAg
from pywatershed.parameters import Parameters, PrmsParameters

# compare in memory (faster) or full output files? or both!
do_compare_output_files = False
do_compare_in_memory = True

# Default tolerances for most variables
default_rtol = 1.0e-5
default_atol = 1.0e-5

# Variable-specific tolerance exceptions
var_tolerance_exceptions = {
    # "dprst_area_open": {"atol": 1.0e-4, "rtol": 1.0e-5},
    # "dprst_stor_hru": {"atol": 1.0e-4, "rtol": 1.0e-5},
    # "dprst_stor_hru_change": {"atol": 1.0e-4, "rtol": 1.0e-5},
    # "dprst_vol_frac": {"atol": 1.0e-4, "rtol": 1.0e-5},
    # "dprst_vol_open": {"atol": 1.0e-4, "rtol": 1.0e-5},
    # "dprst_vol_open_frac": {"atol": 1.0e-4, "rtol": 1.0e-5},
}

calc_methods = ("numpy",)  # numba not yet implemented
params = ("params_sep", "params_one")[1:]  # TODO: fixim


@pytest.fixture(scope="function")
def control(simulation):
    control = Control.load_prms(
        simulation["control_file"], warn_unused_options=False
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
        param_file = simulation["dir"] / "parameters_PRMSRunoff.nc"
        params = PrmsParameters.from_netcdf(param_file)

        # Load ag soil parameters from soilzone_ag
        sz_param_file = simulation["dir"] / "parameters_PRMSSoilzoneAg.nc"
        sz_params = PrmsParameters.from_netcdf(sz_param_file)

        # Merge in ag_soil_moist_max and ag_soil_rechr_max_frac
        params.parameters["ag_soil_moist_max"] = sz_params.parameters[
            "ag_soil_moist_max"
        ]
        params.parameters["ag_soil_rechr_max_frac"] = sz_params.parameters[
            "ag_soil_rechr_max_frac"
        ]

        sat_threshold = sz_params.parameters["sat_threshold"]

    if abs(sat_threshold).min() < 999.0:
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

    # infil_hru is currently post-processed in a way that does not
    # accomodate ag_frac.
    comparison_var_names = set(PRMSRunoffAg.get_variables()) - {
        "dprst_vol_thres_open",
        "infil_ag_hru",
        # "infil_hru",
        "hru_sroff_ag",
    }
    control.options["netcdf_output_var_names"] = comparison_var_names

    output_dir = simulation["output_dir"]

    input_variables = {}
    for key in PRMSRunoffAg.get_inputs():
        nc_pth = output_dir / f"{key}.nc"
        input_variables[key] = nc_pth

    if do_compare_output_files:
        nc_parent = tmp_path / simulation["name"]
        control.options["netcdf_output_dir"] = nc_parent

    runoff = PRMSRunoffAg(
        control=control,
        discretization=discretization,
        parameters=parameters,
        **input_variables,
        imbalance_behavior=None,  # TODO: "error",
        calc_method=calc_method,
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
        compare_netcdfs(
            comparison_var_names,
            tmp_path / simulation["name"],
            output_dir,
            atol=atol,
            rtol=rtol,
        )

    return
