import pathlib as pl
import warnings

import numpy as np
import pytest
from utils_compare import compare_in_memory, compare_netcdfs

from pywatershed.base.adapter import AdapterNetcdf, adapter_factory
from pywatershed.base.control import Control
from pywatershed.base.parameters import Parameters
from pywatershed.hydrology.prms_stream_shade import (
    PRMSStreamShadeConstant,
    PRMSStreamShadeDynamic,
)
from pywatershed.hydrology.prms_stream_temp import PRMSStreamTempHumidityCBH
from pywatershed.parameters import PrmsParameters

# Define sentinel values that should be treated as NaN for
# comparison. PRMS uses -99.9 for missing data, pywatershed
# uses NaN
var_sentinel_to_nan = {"seg_tave_water": -99.9}


# compare in memory (faster) or full output files? or both!
do_compare_output_files = True
do_compare_in_memory = False
# seg_tave_water and seg_tave_water dont match better rtol=atol=5e-3
# while the rest of the variables are better than 1e-3.
# It appears that small numerical differences in the iteration loop
# and in the trig results for seg_shade drive discrepencies just above
# 32-bit precision. In testing so far, errors dont grow with time but longer
# runs in more diverse locations may turn up otherwise.
rtol = atol = 5.0e-3

# TODO: use both parameter schemes again
params = ("params_sep", "params_one")

# Parametrize calc_method: test both numpy and numba implementations
calc_methods = ("numba", "numpy")

# Parametrize energy flux tracking: (track_energy_fluxes, imbalance_behavior)
energy_flux_options = (
    (True, "error"),  # Track fluxes with strict budget checking
    (False, None),  # Don't track fluxes, no budget checking
)

# Parametrize stream shade initialization styles:
# - "instance": Pass a pre-instantiated PRMSStreamShade object
# - "class_params": Pass stream_shade_class & stream_shade_parameters (loaded)
# - "class_params_path": Pass stream_shade_class and stream_shade_parameters
#   as a path (like the notebooks do)
# - "default": Pass None for all, defaulting to PRMSStreamShadeConstant
shade_init_styles = (
    "instance",
    "class_params",
    "class_params_path",
    "default",
)
shade_init_ids = [f"shade_{ss}" for ss in shade_init_styles]


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
def parameter_style(request):
    return request.param


@pytest.fixture(scope="function")
def parameters_shade(parameter_style, simulation, control, request):
    if parameter_style == "params_one":
        param_file = simulation["dir"] / control.options["parameter_file"]
        params = PrmsParameters.load(param_file)
    else:
        param_file = simulation["dir"] / "parameters_PRMSStreamShadeDynamic.nc"
        params = PrmsParameters.from_netcdf(param_file)

    return params


@pytest.fixture(
    scope="function",
    params=energy_flux_options,
    ids=["track_fluxes", "no_track_fluxes"],
)
def energy_flux_config(request):
    return request.param


@pytest.fixture(scope="function")
def parameters(parameter_style, simulation, control, request):
    if parameter_style == "params_one":
        param_file = simulation["dir"] / control.options["parameter_file"]
        params = PrmsParameters.load(param_file)
    else:
        param_file = simulation["dir"] / "parameters_PRMSStreamTemp.nc"
        params = Parameters.from_netcdf(param_file)

    return params


@pytest.fixture(
    scope="function",
    params=shade_init_styles,
    ids=shade_init_ids,
)
def shade_init_style(request):
    return request.param


@pytest.fixture(
    scope="function",
    params=calc_methods,
    ids=calc_methods,
)
def calc_method(request):
    return request.param


def test_compare_prms(
    simulation,
    control,
    discretization,
    parameters,
    parameters_shade,
    energy_flux_config,
    shade_init_style,
    calc_method,
    tmp_path,
):
    tmp_path = pl.Path(tmp_path)
    output_dir = simulation["output_dir"]

    # Unpack energy flux configuration
    track_energy_fluxes, imbalance_behavior = energy_flux_config

    # Step 1: Prepare stream shade based on initialization style
    # This tests the three ways to provide shade to PRMSStreamTemp:
    # - "instance": Pre-instantiated PRMSStreamShade object
    # - "class_params": Pass class and parameters separately
    # - "default": Pass None, defaulting to PRMSStreamShadeConstant
    stream_temp_shade_flag = control.options.get(
        "stream_temp_shade_flag", np.array([0])
    )[0]

    if shade_init_style == "instance":
        # Case 1: Pass a pre-instantiated stream_shade object
        if stream_temp_shade_flag == 0:
            stream_shade = PRMSStreamShadeDynamic(
                parameters_shade,
                discretization,
            )
        else:
            stream_shade = PRMSStreamShadeConstant(
                parameters_shade,
                discretization,
            )
        stream_shade_class = None
        stream_shade_parameters = None
    elif shade_init_style == "class_params":
        # Case 2: Pass stream_shade_class and stream_shade_parameters (loaded)
        stream_shade = None
        if stream_temp_shade_flag == 0:
            stream_shade_class = PRMSStreamShadeDynamic
        else:
            stream_shade_class = PRMSStreamShadeConstant
        stream_shade_parameters = parameters_shade
    elif shade_init_style == "class_params_path":
        # Case 3: Pass stream_shade_class and stream_shade_parameters as path
        # This matches how the notebooks initialize PRMSStreamTemp
        stream_shade = None
        if stream_temp_shade_flag == 0:
            stream_shade_class = PRMSStreamShadeDynamic
        else:
            stream_shade_class = PRMSStreamShadeConstant
        stream_shade_parameters = (
            simulation["dir"] / "parameters_PRMSStreamShadeDynamic.nc"
        )
    else:
        # Case 4: Pass None for all, defaulting to PRMSStreamShadeConstant
        # This requires the main parameters to contain shade parameters
        # We need to merge shade parameters into the main parameters
        stream_shade = None
        stream_shade_class = None
        stream_shade_parameters = None
        # For this case to work, parameters must contain shade params
        # Merge shade parameters into main parameters if not already there

        if not isinstance(parameters, PrmsParameters):
            # In the case of params_one, we get a PrmsParameters object replete
            # with the necessary parameters. In the case of params_sep, we get
            # a Parameters object to which we need to add shade parameters.
            parameters = Parameters.merge(parameters, parameters_shade)

    # Step 2: Prepare inputs for PRMSStreamTempHumidityCBH
    # Most inputs come from PRMS output files, but some need special handling
    stream_temp_inputs = {}
    for key in PRMSStreamTempHumidityCBH.get_inputs():
        if key == "humidity_hru":
            # humidity_hru comes from rhavg.nc in the simulation directory.
            # The file variable is named "rhavg" (percent, 0-100); the *0.01
            # scaling to decimal fraction is applied inside the class.
            stream_temp_inputs[key] = AdapterNetcdf(
                simulation["dir"] / "rhavg.nc",
                "rhavg",
                control,
            )
        else:
            # Most inputs come from PRMS output files
            nc_path = output_dir / f"{key}.nc"
            stream_temp_inputs[key] = nc_path

    # Step 3: Instantiate PRMSStreamTempHumidityCBH with the appropriate shade init style
    stream_temp = PRMSStreamTempHumidityCBH(
        control,
        discretization,
        parameters,
        **stream_temp_inputs,
        stream_shade=stream_shade,
        stream_shade_class=stream_shade_class,
        stream_shade_parameters=stream_shade_parameters,
        calc_method=calc_method,
        imbalance_behavior=imbalance_behavior,
        track_energy_fluxes=track_energy_fluxes,
    )

    # Compare all PRMSStreamTempHumidityCBH variables
    compare_vars = set(PRMSStreamTempHumidityCBH.get_variables()) - set(
        [
            "heat_upstream",
            "heat_lateral",
            "solar_radiation",
            "atmospheric_longwave",
            "friction_heat",
            "groundwater_conduction",
            "heat_outflow",
            "longwave_emission",
            "longwave_vegetation",
            "evaporative_cooling",
            "convective_exchange",
        ]
    )
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
                var_sentinel_to_nan=var_sentinel_to_nan,
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
            # Handle case where variance is zero (all values identical)
            if np.std(sim_valid) == 0 or np.std(obs_valid) == 0:
                corr = np.nan
            else:
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
            var_sentinel_to_nan=var_sentinel_to_nan,
        )

    # The following is currently part of the test, nowhere else is the repr of
    # the energy budget exercised at this time.
    if track_energy_fluxes:
        print(stream_temp.energy_budget)
    stream_temp.finalize()
    return
