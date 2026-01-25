import pathlib as pl

import numpy as np
import pytest
import xarray as xr

import pywatershed as pws
from pywatershed.base.control import Control
from pywatershed.parameters import PrmsParameters

# Test only the nhm configuration
test_configs = ["nhm"]


@pytest.fixture(scope="function")
def control(simulation):
    """Load and configure control object for testing."""
    sim_name = simulation["name"]
    config_name = sim_name.split(":")[1]
    if config_name not in test_configs:
        pytest.skip(
            "The configuration is not tested by test_custom_output: "
            "{config_name}"
        )

    control = Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )
    control.edit_end_time(np.datetime64("1979-07-01T00:00:00"))
    control.options["verbosity"] = 10
    control.options["budget_type"] = "warn"
    control.options["calc_method"] = "numba"
    control.options["input_dir"] = simulation["dir"]

    if "netcdf_output_dir" in control.options:
        del control.options["netcdf_output_dir"]
    if "netcdf_output_var_names" in control.options:
        del control.options["netcdf_output_var_names"]

    return control


@pytest.fixture(scope="function")
def parameters(simulation):
    """Load parameters for testing."""
    param_file = simulation["dir"] / "myparam.param"
    return PrmsParameters.load(param_file)


@pytest.fixture(scope="function")
def nhm_processes():
    """Return the list of NHM processes for testing."""
    return [
        pws.PRMSSolarGeometry,
        pws.PRMSAtmosphere,
        pws.PRMSCanopy,
        pws.PRMSSnow,
        pws.PRMSRunoff,
        pws.PRMSSoilzone,
        pws.PRMSGroundwater,
        pws.PRMSChannel,
    ]


@pytest.fixture(scope="function")
def poi_info(parameters):
    """Calculate POI information from parameters."""
    nhm_seg = parameters.parameters["nhm_seg"]
    poi_gage_segment = parameters.parameters["poi_gage_segment"]
    poi_nhm_seg = nhm_seg[poi_gage_segment - 1]  # fortran indexing

    return {
        "nhm_seg": nhm_seg,
        "poi_gage_segment": poi_gage_segment,
        "poi_nhm_seg": poi_nhm_seg,
        "poi_id_nhm_seg": dict(
            zip(parameters.parameters["poi_gage_id"], poi_nhm_seg.tolist())
        ),
        "poi_nhm_seg_id": dict(
            zip(poi_nhm_seg.tolist(), parameters.parameters["poi_gage_id"])
        ),
    }


def max_stat(
    da: xr.DataArray, dim=None, *, skipna=None, keep_attrs=None, **kwargs
):
    """Custom max statistic function for testing."""
    return da.max(dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)


def test_custom_output_monthly_accumulations(
    simulation, control, parameters, nhm_processes, tmp_path
):
    """Test monthly accumulations functionality."""
    tmp_path = pl.Path(tmp_path)

    # Variables to track
    var_list = ["sroff", "hru_actet", "seg_outflow"]

    # Setup netcdf output for comparison
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    control.options["netcdf_output_dir"] = output_dir
    control.options["netcdf_output_var_names"] = var_list

    model = pws.Model(
        nhm_processes,
        control=control,
        parameters=parameters,
    )

    output = pws.base.CustomOutput(
        control=control,
        model=model,
        monthly_accum_var_list=var_list,
    )

    model.run(finalize=True, output=output)

    # Check that monthly accumulations were created
    assert output.monthly_accumulations is not None
    assert "sroff" in output.monthly_accumulations
    assert "hru_actet" in output.monthly_accumulations
    assert "seg_outflow" in output.monthly_accumulations

    # Check that n_days_per_month was tracked as a DataArray
    assert output.n_days_per_month is not None
    assert isinstance(output.n_days_per_month, xr.DataArray)
    assert "month" in output.n_days_per_month.dims
    assert len(output.n_days_per_month) > 0
    assert output.n_days_per_month.sum() == control.n_times
    # Check it has same coordinates as monthly accumulations
    assert (
        output.n_days_per_month.coords["month"] == output.time_months
    ).all()

    # Check dimensions and coordinates
    for var_name, data_array in output.monthly_accumulations.items():
        assert isinstance(data_array, xr.DataArray)
        assert "month" in data_array.dims
        assert len(data_array.coords["month"]) == len(output.time_months)

    # Validate against netcdf output by post-processing
    for var_name in var_list:
        nc_file = output_dir / f"{var_name}.nc"
        assert nc_file.exists(), f"NetCDF file {nc_file} not created"

        # Load netcdf data
        ds = xr.open_dataset(nc_file)

        # Calculate monthly means from netcdf
        monthly_mean_nc = ds[var_name].resample(time="1MS").mean()

        # Calculate monthly means from CustomOutput accumulations
        # This division now works because n_days_per_month is a DataArray
        # with proper month dimension
        custom_monthly_mean = (
            output.monthly_accumulations[var_name] / output.n_days_per_month
        )

        # Check shapes match
        assert monthly_mean_nc.shape == custom_monthly_mean.shape, (
            f"{var_name}: shapes don't match"
        )

        # Check values match (allowing for floating point tolerance)
        np.testing.assert_allclose(
            monthly_mean_nc.values,
            custom_monthly_mean.values,
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"{var_name}: monthly means don't match netcdf",
        )

        ds.close()


def test_custom_output_poi_data(
    simulation, control, parameters, nhm_processes, poi_info, tmp_path
):
    """Test POI data collection and statistics."""
    tmp_path = pl.Path(tmp_path)

    # Variables to track
    var_list = ["seg_outflow"]

    # Setup netcdf output for comparison
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    control.options["netcdf_output_dir"] = output_dir
    control.options["netcdf_output_var_names"] = var_list

    model = pws.Model(
        nhm_processes,
        control=control,
        parameters=parameters,
    )

    output = pws.base.CustomOutput(
        control=control,
        model=model,
        poi_var_list=var_list,
        poi_nhm_seg=poi_info["poi_nhm_seg"],
        poi_gage_segment=poi_info["poi_gage_segment"] - 1,
        poi_stats=["mean", "median", max_stat],
        poi_stats_groupby={"median": "month"},
        poi_stats_resample={"median": "1MS", "max_stat": "5D"},
    )

    model.run(finalize=True, output=output)

    # Check that POI arrays were created
    assert output.poi_arrays is not None
    assert "seg_outflow" in output.poi_arrays

    # Verify that the last timestep matches the model state
    # (from notebook assertion)
    assert (
        model.processes["PRMSChannel"]["seg_outflow"][
            poi_info["poi_gage_segment"] - 1
        ]
        == output.poi_arrays["seg_outflow"][-1, :]
    ).all()

    # Check POI statistics
    assert output.poi_stats is not None
    assert "seg_outflow_mean" in output.poi_stats
    assert "seg_outflow_median_month" in output.poi_stats
    assert "seg_outflow_median_1MS" in output.poi_stats
    assert "seg_outflow_max_stat_5D" in output.poi_stats

    # Check dimensions
    poi_array = output.poi_arrays["seg_outflow"]
    assert isinstance(poi_array, xr.DataArray)
    assert "time" in poi_array.dims
    assert len(poi_array.coords["time"]) == control.n_times

    # Validate against netcdf output by post-processing
    for var_name in var_list:
        nc_file = output_dir / f"{var_name}.nc"
        assert nc_file.exists(), f"NetCDF file {nc_file} not created"

        # Load netcdf data
        ds = xr.open_dataset(nc_file)

        # Extract POI data from full netcdf output
        poi_indices = poi_info["poi_gage_segment"] - 1
        poi_data_nc = ds[var_name].isel(nhm_seg=poi_indices)

        # Check that full time series matches
        np.testing.assert_allclose(
            poi_data_nc.values,
            output.poi_arrays[var_name].values,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"{var_name}: POI arrays don't match netcdf",
        )

        # Calculate and validate statistics from netcdf
        # Mean over all time
        mean_nc = poi_data_nc.mean(dim="time")
        np.testing.assert_allclose(
            mean_nc.values,
            output.poi_stats[f"{var_name}_mean"].values,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"{var_name}: mean statistic doesn't match",
        )

        # Median grouped by month
        median_by_month_nc = poi_data_nc.groupby("time.month").median(
            dim="time"
        )
        np.testing.assert_allclose(
            median_by_month_nc.values,
            output.poi_stats[f"{var_name}_median_month"].values,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"{var_name}: median_month statistic doesn't match",
        )

        # Median resampled to monthly
        median_resample_nc = poi_data_nc.resample(time="1MS").median(
            dim="time"
        )
        np.testing.assert_allclose(
            median_resample_nc.values,
            output.poi_stats[f"{var_name}_median_1MS"].values,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"{var_name}: median_1MS statistic doesn't match",
        )

        # Max resampled to 5 day periods
        max_resample_nc = poi_data_nc.resample(time="5D").max(dim="time")
        np.testing.assert_allclose(
            max_resample_nc.values,
            output.poi_stats[f"{var_name}_max_stat_5D"].values,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"{var_name}: max_stat_5D statistic doesn't match",
        )

        ds.close()


def test_custom_output_hru_subset(
    simulation, control, parameters, nhm_processes, tmp_path
):
    """Test HRU subset data collection and statistics."""
    tmp_path = pl.Path(tmp_path)

    # Variables to track
    var_list = ["hru_actet", "pkwater_equiv"]

    # Setup netcdf output for comparison
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    control.options["netcdf_output_dir"] = output_dir
    control.options["netcdf_output_var_names"] = var_list

    model = pws.Model(
        nhm_processes,
        control=control,
        parameters=parameters,
    )

    # Select first HRU for testing
    hru_id = parameters.parameters["nhm_id"][0].tolist()
    # Find the index of this HRU
    hru_index = np.where(parameters.parameters["nhm_id"] == hru_id)[0][0]

    output = pws.base.CustomOutput(
        control=control,
        model=model,
        hru_sub_var_list=var_list,
        hru_sub_ids=[hru_id],
        hru_sub_stats=["mean", max_stat],
        hru_sub_stats_resample={"mean": "1MS", "max_stat": "1YS"},
    )

    model.run(finalize=True, output=output)

    # Check that HRU subset arrays were created
    assert output.hru_sub_arrays is not None
    assert "hru_actet" in output.hru_sub_arrays
    assert "pkwater_equiv" in output.hru_sub_arrays

    # Check HRU subset statistics
    assert output.hru_sub_stats is not None
    assert "hru_actet_mean_1MS" in output.hru_sub_stats
    assert "hru_actet_max_stat_1YS" in output.hru_sub_stats
    assert "pkwater_equiv_mean_1MS" in output.hru_sub_stats
    assert "pkwater_equiv_max_stat_1YS" in output.hru_sub_stats

    # Validate against netcdf output by post-processing
    for var_name in var_list:
        nc_file = output_dir / f"{var_name}.nc"
        assert nc_file.exists(), f"NetCDF file {nc_file} not created"

        # Load netcdf data
        ds = xr.open_dataset(nc_file)

        # Extract HRU subset data from full netcdf output
        hru_data_nc = ds[var_name].isel(nhm_id=hru_index)

        # Check that full time series matches
        np.testing.assert_allclose(
            hru_data_nc.values,
            output.hru_sub_arrays[var_name].values.squeeze(),
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"{var_name}: HRU subset arrays don't match netcdf",
        )

        # Calculate and validate statistics from netcdf
        # Mean resampled to monthly
        mean_resample_nc = hru_data_nc.resample(time="1MS").mean(dim="time")
        np.testing.assert_allclose(
            mean_resample_nc.values,
            output.hru_sub_stats[f"{var_name}_mean_1MS"].values.squeeze(),
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"{var_name}: mean_1MS statistic doesn't match",
        )

        # Max resampled to yearly
        max_resample_nc = hru_data_nc.resample(time="1YS").max(dim="time")
        np.testing.assert_allclose(
            max_resample_nc.values,
            output.hru_sub_stats[f"{var_name}_max_stat_1YS"].values.squeeze(),
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"{var_name}: max_stat_1YS statistic doesn't match",
        )

        ds.close()


def test_custom_output_combined(
    simulation, control, parameters, nhm_processes, poi_info, tmp_path
):
    """Test all CustomOutput features together (comprehensive notebook test)."""
    tmp_path = pl.Path(tmp_path)

    # All variables to track
    all_vars = ["sroff", "hru_actet", "seg_outflow", "pkwater_equiv"]

    # Setup netcdf output for comparison
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    control.options["netcdf_output_dir"] = output_dir
    control.options["netcdf_output_var_names"] = all_vars

    model = pws.Model(
        nhm_processes,
        control=control,
        parameters=parameters,
    )

    hru_id = parameters.parameters["nhm_id"][0].tolist()

    output = pws.base.CustomOutput(
        control=control,
        model=model,
        monthly_accum_var_list=["sroff", "hru_actet", "seg_outflow"],
        poi_var_list=["seg_outflow"],
        poi_nhm_seg=poi_info["poi_nhm_seg"],
        poi_gage_segment=poi_info["poi_gage_segment"] - 1,
        poi_stats=["mean", "median", max_stat],
        poi_stats_groupby={"median": "month"},
        poi_stats_resample={"median": "1MS", "max_stat": "5D"},
        hru_sub_var_list=["hru_actet", "pkwater_equiv"],
        hru_sub_ids=[hru_id],
        hru_sub_stats=["mean", max_stat],
        hru_sub_stats_resample={"mean": "1MS", "max_stat": "1YS"},
    )

    model.run(finalize=True, output=output)

    # Verify all outputs are available
    assert output.monthly_accumulations is not None
    assert output.poi_arrays is not None
    assert output.poi_stats is not None
    assert output.hru_sub_arrays is not None
    assert output.hru_sub_stats is not None

    # Check time coordinates
    assert output.time is not None
    assert output.time_months is not None
    assert len(output.time) == control.n_times

    # Spot check: validate one variable from each output type against netcdf
    # (Full validation done in individual tests above)

    # Monthly accumulation check
    ds_sroff = xr.open_dataset(output_dir / "sroff.nc")
    monthly_nc = ds_sroff["sroff"].resample(time="1MS").sum()
    np.testing.assert_allclose(
        monthly_nc.values,
        output.monthly_accumulations["sroff"].values,
        rtol=1e-10,
        atol=1e-10,
    )
    ds_sroff.close()

    # POI check
    ds_seg = xr.open_dataset(output_dir / "seg_outflow.nc")
    poi_indices = poi_info["poi_gage_segment"] - 1
    poi_data_nc = ds_seg["seg_outflow"].isel(nhm_seg=poi_indices)
    mean_nc = poi_data_nc.mean(dim="time")
    np.testing.assert_allclose(
        mean_nc.values,
        output.poi_stats["seg_outflow_mean"].values,
        rtol=1e-10,
        atol=1e-10,
    )
    ds_seg.close()

    # HRU subset check
    ds_actet = xr.open_dataset(output_dir / "hru_actet.nc")
    hru_index = np.where(parameters.parameters["nhm_id"] == hru_id)[0][0]
    hru_data_nc = ds_actet["hru_actet"].isel(nhm_id=hru_index)
    mean_resample_nc = hru_data_nc.resample(time="1MS").mean(dim="time")
    np.testing.assert_allclose(
        mean_resample_nc.values,
        output.hru_sub_stats["hru_actet_mean_1MS"].values.squeeze(),
        rtol=1e-10,
        atol=1e-10,
    )
    ds_actet.close()


def test_custom_output_properties_before_finalization(
    simulation, control, parameters, nhm_processes, poi_info
):
    """Test that properties return None before finalization."""
    model = pws.Model(
        nhm_processes,
        control=control,
        parameters=parameters,
    )

    output = pws.base.CustomOutput(
        control=control,
        model=model,
        monthly_accum_var_list=["sroff"],
        poi_var_list=["seg_outflow"],
        poi_nhm_seg=poi_info["poi_nhm_seg"],
        poi_stats=["mean"],
    )

    # Properties should return None before finalization
    assert output.monthly_accumulations is None
    assert output.poi_arrays is None
    assert output.poi_stats is None
    assert output.n_days_per_month is None


def test_custom_output_poi_requires_segments(
    simulation, control, parameters, nhm_processes
):
    """Test that POI variables require segment specification."""
    model = pws.Model(
        nhm_processes,
        control=control,
        parameters=parameters,
    )

    # Should raise ValueError if poi_var_list provided without segments
    with pytest.raises(ValueError, match="poi_nhm_seg or poi_gage_segment"):
        output = pws.base.CustomOutput(
            control=control,
            model=model,
            poi_var_list=["seg_outflow"],
        )


def test_custom_output_string_stats(
    simulation, control, parameters, nhm_processes, poi_info
):
    """Test that built-in string statistics work correctly."""
    model = pws.Model(
        nhm_processes,
        control=control,
        parameters=parameters,
    )

    output = pws.base.CustomOutput(
        control=control,
        model=model,
        poi_var_list=["seg_outflow"],
        poi_nhm_seg=poi_info["poi_nhm_seg"],
        poi_stats=["mean", "median", "std"],  # All built-in string stats
    )

    model.run(finalize=True, output=output)

    # Verify all built-in stats were calculated
    assert "seg_outflow_mean" in output.poi_stats
    assert "seg_outflow_median" in output.poi_stats
    assert "seg_outflow_std" in output.poi_stats
