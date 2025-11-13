import pathlib as pl
import warnings

import numpy as np
import pytest
from utils_compare import compare_in_memory, compare_netcdfs

from pywatershed.base.adapter import adapter_factory
from pywatershed.base.control import Control
from pywatershed.base.parameters import Parameters
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

    # Step 1: Instantiate shade computer (composed component)
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

    # Step 2: Prepare inputs for PRMSStreamTemp
    # All inputs come from PRMS output files
    stream_temp_inputs = {}
    for key in PRMSStreamTemp.get_inputs():
        nc_path = output_dir / f"{key}.nc"
        stream_temp_inputs[key] = nc_path

    # Step 3: Instantiate PRMSStreamTemp with composed shade_computer
    stream_temp = PRMSStreamTemp(
        control,
        discretization,
        parameters,
        **stream_temp_inputs,
        shade_computer=shade_computer,
        budget_type=None,
    )

    # Compare all PRMSStreamTemp variables
    compare_vars = set(PRMSStreamTemp.get_variables())
    if do_compare_output_files:
        nc_parent = tmp_path / simulation["name"].replace(":", "_")
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

        # Run stream temperature
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
        # Compute statistics from output files
        import xarray as xr

        print("\n" + "=" * 80)
        print("STATISTICS SUMMARY (all timesteps, all segments)")
        print("=" * 80)

        sim_dir = tmp_path / simulation["name"].replace(":", "_")

        for var in sorted(compare_vars):
            sim_file = sim_dir / f"{var}.nc"
            obs_file = output_dir / f"{var}.nc"

            if not sim_file.exists() or not obs_file.exists():
                continue

            # Load data
            sim_ds = xr.open_dataset(sim_file)
            obs_ds = xr.open_dataset(obs_file)

            sim_data = sim_ds[var].values
            obs_data = obs_ds[var].values

            sim_ds.close()
            obs_ds.close()

            # Flatten to 1D for statistics
            sim_flat = sim_data.flatten()
            obs_flat = obs_data.flatten()

            # Create mask for valid (non-NaN, non-sentinel) values
            valid_mask = (
                ~np.isnan(sim_flat)
                & ~np.isnan(obs_flat)
                & (obs_flat > -90)  # Exclude sentinel values like -98.9, -99.9
            )

            if np.sum(valid_mask) == 0:
                print(f"\n{var}: No valid data points")
                continue

            sim_valid = sim_flat[valid_mask]
            obs_valid = obs_flat[valid_mask]

            # Compute statistics
            bias = np.mean(sim_valid - obs_valid)
            rmse = np.sqrt(np.mean((sim_valid - obs_valid) ** 2))

            # R² calculation
            ss_res = np.sum((obs_valid - sim_valid) ** 2)
            ss_tot = np.sum((obs_valid - np.mean(obs_valid)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

            # Correlation coefficient
            corr = np.corrcoef(sim_valid, obs_valid)[0, 1]

            print(f"\n{var}:")
            print(f"  N valid points: {np.sum(valid_mask):,}")
            print(f"  R²:             {r2:.6f}")
            print(f"  Correlation:    {corr:.6f}")
            print(f"  RMSE:           {rmse:.6f}")
            print(f"  Bias:           {bias:.6f}")
            print(f"  Mean observed:  {np.mean(obs_valid):.6f}")
            print(f"  Mean simulated: {np.mean(sim_valid):.6f}")

        print("\n" + "=" * 80)

        compare_netcdfs(
            compare_vars,
            tmp_path / simulation["name"].replace(":", "_"),
            output_dir,
            atol=atol,
            rtol=rtol,
        )

    return
