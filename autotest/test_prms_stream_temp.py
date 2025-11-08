import pathlib as pl

import pytest
from utils_compare import compare_in_memory, compare_netcdfs

from pywatershed.base.adapter import adapter_factory
from pywatershed.base.control import Control
from pywatershed.base.parameters import Parameters
from pywatershed.hydrology.prms_stream_temp import PRMSStreamTemp
from pywatershed.parameters import PrmsParameters

# compare in memory (faster) or full output files? or both!
do_compare_output_files = True
do_compare_in_memory = True
rtol = atol = 1.0e-5  # Temperature calculations may need looser tolerance

params = ("params_sep", "params_one")


@pytest.fixture(scope="function")
def control(simulation):
    if "stream_temp" in simulation["name"]:
        pytest.skip(
            f"Domain not configured for stream temp: {simulation['name']}"
        )

    ctl = Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )

    # Check if stream temperature module is present
    if "stream_temp" not in ctl.options.get("streamflow_module", ""):
        pytest.skip(
            f"PRMSStreamTemp not present in simulation {simulation['name']}"
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
        param_file = simulation["dir"] / "parameters_PRMSStreamTemp.nc"
        params = PrmsParameters.from_netcdf(param_file)

    return params


def test_compare_prms(
    simulation, control, discretization, parameters, tmp_path
):
    tmp_path = pl.Path(tmp_path)
    output_dir = simulation["output_dir"]

    input_variables = {}
    for key in PRMSStreamTemp.get_inputs():
        nc_path = output_dir / f"{key}.nc"
        input_variables[key] = nc_path

    stream_temp = PRMSStreamTemp(
        control,
        discretization,
        parameters,
        **input_variables,
        budget_type="warn",  # Temperature doesn't have mass budget
    )

    compare_vars = set(PRMSStreamTemp.get_variables()) - {
        "seg_inflow",  # May be computed differently
    }

    if do_compare_output_files:
        nc_parent = tmp_path / simulation["name"].replace(":", "_")
        stream_temp.initialize_netcdf(nc_parent)
        # test that init netcdf twice raises a warning
        with pytest.warns(UserWarning):
            stream_temp.initialize_netcdf(nc_parent)

    if do_compare_in_memory:
        answers = {}
        for var in compare_vars:
            var_pth = output_dir / f"{var}.nc"
            if var_pth.exists():
                answers[var] = adapter_factory(
                    var_pth, variable_name=var, control=control
                )

    for istep in range(control.n_times):
        control.advance()
        stream_temp.advance()
        stream_temp.calculate(float(istep))
        stream_temp.output()
        if do_compare_in_memory:
            for var in answers.values():
                var.advance()
            compare_in_memory(
                stream_temp,
                answers,
                atol=atol,
                rtol=rtol,
                skip_missing_ans=True,
            )

    stream_temp.finalize()

    if do_compare_output_files:
        compare_netcdfs(
            compare_vars,
            tmp_path / simulation["name"].replace(":", "_"),
            output_dir,
            atol=atol,
            rtol=rtol,
        )

    return


def test_init_values():
    """Test that initial values are set correctly."""
    init_vals = PRMSStreamTemp.get_init_values()

    assert "seg_tave_water" in init_vals
    assert "seg_tave_upstream" in init_vals
    assert "seg_tave_gw" in init_vals
    assert "seg_tave_ss" in init_vals
    assert "seg_tave_lat" in init_vals
    assert "seg_shade" in init_vals
    assert "seg_inflow" in init_vals


def test_dimensions():
    """Test that dimensions are correctly defined."""
    dims = PRMSStreamTemp.get_dimensions()

    assert "nhru" in dims
    assert "nsegment" in dims
    assert "nmonths" in dims


def test_parameters():
    """Test that all required parameters are listed."""
    params = PRMSStreamTemp.get_parameters()

    # Check key parameters exist
    assert "hru_segment" in params
    assert "tosegment" in params
    assert "seg_length" in params
    assert "seg_slope" in params
    assert "seg_lat" in params
    assert "seg_elev" in params
    assert "ss_tau" in params
    assert "gw_tau" in params
    assert "melt_temp" in params
    assert "albedo" in params
    assert "lat_temp_adj" in params
    assert "stream_tave_init" in params
    assert "stream_temp_shade_flag" in params


def test_inputs():
    """Test that all required inputs are listed."""
    inputs = PRMSStreamTemp.get_inputs()

    assert "seg_outflow" in inputs
    assert "seg_lateral_inflow" in inputs
    assert "seginc_sroff" in inputs
    assert "seginc_ssflow" in inputs
    assert "seginc_gwflow" in inputs
    assert "seginc_swrad" in inputs
    assert "seg_humid" in inputs
    assert "seg_potet" in inputs
    assert "seg_ccov" in inputs
    assert "seg_melt" in inputs
    assert "seg_rain" in inputs
    assert "seg_tave_air" in inputs
    assert "seg_width" in inputs
