import pathlib as pl

import pytest
from utils_compare import compare_in_memory, compare_netcdfs

from pywatershed.base.adapter import adapter_factory
from pywatershed.base.control import Control
from pywatershed.base.parameters import Parameters
from pywatershed.hydrology.prms_channel import PRMSChannel
from pywatershed.parameters import PrmsParameters

# compare in memory (faster) or full output files? or both!
do_compare_output_files = True
do_compare_in_memory = True
default_rtol = default_atol = 1.0e-7
# Default tolerances for most variables (depth-based)


calc_methods = ("numpy", "numba")
params = ("params_sep", "params_one")

var_tolerance_exceptions = {}


@pytest.fixture(scope="function")
def control(simulation):
    if "obsin" in simulation["name"]:
        pytest.skip("Not testing passthrough flow graph for drb_2yr:nhm_obsin")

    ctl = Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )
    if ctl.options["streamflow_module"] == "strmflow":
        pytest.skip(
            f"PRMSChannel not present in simulation {simulation['name']}"
        )
    del ctl.options["netcdf_output_dir"]
    del ctl.options["netcdf_output_var_names"]

    return ctl


@pytest.fixture(scope="function")
def discretization(simulation):
    dis_hru_file = simulation["dir"] / "parameters_dis_hru.nc"
    dis_seg_file = simulation["dir"] / "parameters_dis_seg.nc"
    dis = Parameters.merge(
        Parameters.from_netcdf(dis_hru_file, encoding=False),
        Parameters.from_netcdf(dis_seg_file, encoding=False),
    )
    return dis


@pytest.fixture(scope="function", params=params)
def parameters(simulation, control, request):
    if request.param == "params_one":
        param_file = simulation["dir"] / control.options["parameter_file"]
        params = PrmsParameters.load(param_file)
    else:
        param_file = simulation["dir"] / "parameters_PRMSChannel.nc"
        params = PrmsParameters.from_netcdf(param_file)

    return params


@pytest.mark.parametrize("calc_method", calc_methods)
def test_compare_prms(
    simulation, control, discretization, parameters, tmp_path, calc_method
):
    tmp_path = pl.Path(tmp_path)
    output_dir = simulation["output_dir"]

    input_variables = {}
    for key in PRMSChannel.get_inputs():
        nc_path = output_dir / f"{key}.nc"
        input_variables[key] = nc_path

    channel = PRMSChannel(
        control,
        discretization,
        parameters,
        **input_variables,
        imbalance_behavior="error",
        calc_method=calc_method,
    )

    compare_vars = set(PRMSChannel.get_variables()) - {
        "inflow_ts_prev",
        "outflow_ts",
    }
    if do_compare_output_files:
        nc_parent = tmp_path / simulation["name"].replace(":", "_")
        channel.initialize_netcdf(nc_parent)
        # test that init netcdf twice raises a warning
        with pytest.warns(UserWarning):
            channel.initialize_netcdf(nc_parent)

    if do_compare_in_memory:
        answers = {}
        for var in compare_vars:
            var_pth = output_dir / f"{var}.nc"
            answers[var] = adapter_factory(
                var_pth, variable_name=var, control=control
            )

    for istep in range(control.n_times):
        control.advance()
        channel.advance()
        channel.calculate(float(istep))
        channel.output()
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
                channel,
                answers,
                atol=default_atol,
                rtol=default_rtol,
                skip_missing_ans=True,
                var_tolerances=var_tolerances,
            )

    channel.finalize()

    if do_compare_output_files:
        # Filter out variables without answer files
        vars_with_answers = []
        for var in compare_vars:
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
            compare_vars,
            tmp_path / simulation["name"].replace(":", "_"),
            output_dir,
            atol=default_atol,
            rtol=default_rtol,
            var_tolerances=var_tolerances,
            # fail_after_all_vars=False,
            verbose=True,
        )

    return
