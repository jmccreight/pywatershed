import pathlib as pl
import warnings

import numpy as np
import pytest
from utils_compare import compare_in_memory, compare_netcdfs

from pywatershed.base.adapter import adapter_factory
from pywatershed.base.control import Control
from pywatershed.base.parameters import Parameters
from pywatershed.hydrology.prms_hydraulic_geometry import (
    PRMSHydraulicGeometryDefault,
)
from pywatershed.hydrology.prms_stream_shade import (
    PRMSStreamShadeConstant,
    PRMSStreamShadeDynamic,
)
from pywatershed.hydrology.prms_stream_temp import PRMSStreamTemp
from pywatershed.parameters import PrmsParameters

# compare in memory (faster) or full output files? or both!
do_compare_output_files = False
do_compare_in_memory = True
rtol = atol = 1.0e-3  # Temperature calculations may need looser tolerance

params = ("params_sep", "params_one")[1:]  # TODO: use both again


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

    # TODO: replace with inputs from netcdf.
    # Step 1: Instantiate hydraulic geometry (upstream process)
    hydraulic_geom_inputs = {}
    for key in PRMSHydraulicGeometryDefault.get_inputs():
        nc_path = output_dir / f"{key}.nc"
        hydraulic_geom_inputs[key] = nc_path

    hydraulic_geom = PRMSHydraulicGeometryDefault(
        control,
        discretization,
        parameters,
        **hydraulic_geom_inputs,
        budget_type=None,
    )

    # Step 2: Instantiate shade computer (composed component)
    stream_temp_shade_flag = control.options.get(
        "stream_temp_shade_flag", np.array([0])
    )[0]

    if stream_temp_shade_flag == 0:
        # Dynamic shade computation
        shade_computer = PRMSStreamShadeDynamic(
            parameters, discretization.dims["nsegment"]
        )
    else:
        # Constant shade parameters
        shade_computer = PRMSStreamShadeConstant(
            parameters, discretization.dims["nsegment"]
        )

    # Step 3: Prepare inputs for PRMSStreamTemp
    # Most inputs come from output files, but seg_flow_* come from hydraulic_geom
    stream_temp_inputs = {}
    for key in PRMSStreamTemp.get_inputs():
        if key in [
            "seg_flow_width",
            "seg_flow_depth",
            "seg_flow_area",
            "seg_flow_velocity",
        ]:
            # These come from hydraulic_geom process, not files
            continue
        nc_path = output_dir / f"{key}.nc"
        stream_temp_inputs[key] = nc_path

    # Step 4: Instantiate PRMSStreamTemp with composed shade_computer
    stream_temp = PRMSStreamTemp(
        control,
        discretization,
        parameters,
        **stream_temp_inputs,
        seg_flow_width=hydraulic_geom.seg_flow_width,
        seg_flow_depth=hydraulic_geom.seg_flow_depth,
        seg_flow_area=hydraulic_geom.seg_flow_area,
        seg_flow_velocity=hydraulic_geom.seg_flow_velocity,
        shade_computer=shade_computer,
        budget_type=None,
    )

    compare_vars = set(PRMSStreamTemp.get_variables()) - {
        "seginc_sroff",  # Computed internally
        "seginc_ssflow",  # Computed internally
        "seginc_gwflow",  # Computed internally
        "seginc_swrad",  # Computed internally
        "seginc_potet",  # Computed internally
    }

    if do_compare_output_files:
        nc_parent = tmp_path / simulation["name"].replace(":", "_")
        hydraulic_geom.initialize_netcdf(nc_parent / "hydraulic_geom")
        stream_temp.initialize_netcdf(nc_parent)

    if do_compare_in_memory:
        answers = {}
        for var in compare_vars:
            var_pth = output_dir / f"{var}.nc"
            if var_pth.exists():
                answers[var] = adapter_factory(
                    var_pth, variable_name=var, control=control
                )

    # Time loop - run both processes
    for istep in range(control.n_times):
        control.advance()

        # Run hydraulic geometry first (upstream)
        hydraulic_geom.advance()
        hydraulic_geom.calculate(float(istep))
        hydraulic_geom.output()

        # Then run stream temperature
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

    hydraulic_geom.finalize()
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
