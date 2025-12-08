import pathlib as pl
import warnings

import numpy as np
import pytest
from utils_compare import compare_in_memory, compare_netcdfs

from pywatershed.base.adapter import adapter_factory
from pywatershed.base.control import Control
from pywatershed.base.parameters import Parameters
from pywatershed.hydrology.prms_hydraulic_geometry import (
    PRMSHydraulicGeometry,
    PRMSHydraulicGeometryDefault,
)
from pywatershed.parameters import PrmsParameters

# compare in memory (faster) or full output files? or both!
do_compare_output_files = True
do_compare_in_memory = True
rtol = atol = 1.0e-5

params = ("params_sep", "params_one")  # TODO: use both again


@pytest.fixture(scope="function")
def control(simulation):
    with warnings.catch_warnings():
        # This is the only way to silence "invalid" options.
        warnings.simplefilter("ignore")
        ctl = Control.load_prms(
            simulation["control_file"],
            warn_unused_options=False,
            keep_unused_options=True,
        )

    # Check if stream temperature module is present
    stream_temp_flag = ctl.options.get("stream_temp_flag", np.array([0]))[0]

    if stream_temp_flag != 1:
        pytest.skip(
            f"'stream_temp_flag' not 1/on simulation {simulation['name']}"
        )

    del ctl.options["netcdf_output_dir"]
    del ctl.options["netcdf_output_var_names"]

    return ctl


@pytest.fixture(scope="function")
def discretization(simulation):
    dis_seg_file = simulation["dir"] / "parameters_dis_seg.nc"
    dis = Parameters.from_netcdf(dis_seg_file, encoding=False)
    return dis


@pytest.fixture(scope="function", params=params)
def parameters(simulation, control, request):
    if request.param == "params_one":
        param_file = simulation["dir"] / control.options["parameter_file"]
        params = PrmsParameters.load(param_file)
    else:
        param_file = simulation["dir"] / "parameters_PRMSHydraulicGeometry.nc"
        params = PrmsParameters.from_netcdf(param_file)

    return params


def test_compare_default_depth(
    simulation, control, discretization, parameters, tmp_path
):
    """Test PRMSHydraulicGeometryDefault (uses default depth parameters)."""
    tmp_path = pl.Path(tmp_path)
    output_dir = simulation["output_dir"]

    input_variables = {}
    for key in PRMSHydraulicGeometryDefault.get_inputs():
        nc_path = output_dir / f"{key}.nc"
        input_variables[key] = nc_path

    hydraulic_geom = PRMSHydraulicGeometryDefault(
        control,
        discretization,
        parameters,
        **input_variables,
    )

    compare_vars = set(PRMSHydraulicGeometryDefault.get_variables())

    if do_compare_output_files:
        nc_parent = tmp_path / simulation["name"].replace(":", "_")
        hydraulic_geom.initialize_netcdf(nc_parent)

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
        hydraulic_geom.advance()
        hydraulic_geom.calculate(float(istep))
        hydraulic_geom.output()
        if do_compare_in_memory:
            for var in answers.values():
                var.advance()
            compare_in_memory(
                hydraulic_geom,
                answers,
                atol=atol,
                rtol=rtol,
                skip_missing_ans=True,
            )

    hydraulic_geom.finalize()

    if do_compare_output_files:
        compare_netcdfs(
            compare_vars,
            tmp_path / simulation["name"].replace(":", "_"),
            output_dir,
            atol=atol,
            rtol=rtol,
        )

    return


def test_compare_full(
    simulation, control, discretization, parameters, tmp_path
):
    """Test PRMSHydraulicGeometry.
    Currently manually adding in default depth_alpha and depth_m parameters to
    test against default results from PRMS.
    """
    tmp_path = pl.Path(tmp_path)
    output_dir = simulation["output_dir"]

    # Manually add depth parameters to test the custom code path
    # We need to add depth_alpha and depth_m to the parameters
    import xarray as xr

    # Get the parameters as xarray dataset
    params_ds = parameters.to_xr_ds()
    nsegment = params_ds.dims["nsegment"]

    # Add depth parameters with PRMS default values
    params_ds["depth_alpha"] = xr.DataArray(
        np.full(nsegment, 0.27, dtype=np.float64),
        dims=["nsegment"],
        attrs={"units": "meters"},
    )
    params_ds["depth_m"] = xr.DataArray(
        np.full(nsegment, 0.39, dtype=np.float64),
        dims=["nsegment"],
        attrs={"units": "none"},
    )

    # Create new Parameters object with added depth parameters
    from pywatershed.base.parameters import Parameters as pws_Parameters

    parameters_with_depth = pws_Parameters.from_ds(params_ds)

    input_variables = {}
    for key in PRMSHydraulicGeometry.get_inputs():
        nc_path = output_dir / f"{key}.nc"
        input_variables[key] = nc_path

    hydraulic_geom = PRMSHydraulicGeometry(
        control,
        discretization,
        parameters_with_depth,
        **input_variables,
    )

    compare_vars = set(PRMSHydraulicGeometry.get_variables())

    if do_compare_output_files:
        nc_parent = tmp_path / simulation["name"].replace(":", "_")
        hydraulic_geom.initialize_netcdf(nc_parent)

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
        hydraulic_geom.advance()
        hydraulic_geom.calculate(float(istep))
        hydraulic_geom.output()
        if do_compare_in_memory:
            for var in answers.values():
                var.advance()
            compare_in_memory(
                hydraulic_geom,
                answers,
                atol=atol,
                rtol=rtol,
                skip_missing_ans=True,
            )

    hydraulic_geom.finalize()

    if do_compare_output_files:
        compare_netcdfs(
            compare_vars,
            tmp_path / simulation["name"].replace(":", "_"),
            output_dir,
            atol=atol,
            rtol=rtol,
        )

    return
