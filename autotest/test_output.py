import pathlib as pl

import numpy as np
import pytest
import xarray as xr

import pywatershed as pws
from pywatershed import (
    PassThroughFlowNodeMaker,
    prms_channel_flow_graph_to_model_dict,
)
from pywatershed.analysis.time_stats import (
    max_5day,
    max_yearly,
    mean_monthly,
    median_by_month,
    median_monthly,
)
from pywatershed.base.control import Control
from pywatershed.base.model import Model
from pywatershed.parameters import Parameters, PrmsParameters

# Test only drb_2yr domain with nhm configuration
test_sim_names = ["drb_2yr:nhm"]


@pytest.fixture(scope="function")
def control(simulation):
    """Control object for drb_2yr domain."""
    sim_name = simulation["name"]
    if sim_name not in test_sim_names:
        pytest.skip(
            f"The configuration is not tested by test_output: {sim_name}"
        )

    control = Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )
    control.edit_end_time(np.datetime64("1979-07-01T00:00:00"))
    control.options["verbosity"] = 10
    control.options["imbalance_behavior"] = "warn"
    control.options["calc_method"] = "numba"
    control.options["input_dir"] = simulation["dir"]

    if "netcdf_output_dir" in control.options:
        del control.options["netcdf_output_dir"]
    if "netcdf_output_var_names" in control.options:
        del control.options["netcdf_output_var_names"]

    return control


@pytest.fixture(scope="function")
def parameters(simulation):
    """PRMS parameters."""
    param_file = simulation["dir"] / "myparam.param"
    return PrmsParameters.load(param_file)


@pytest.fixture(scope="function")
def nhm_processes():
    """NHM process list."""
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
    """POI information with nhm_seg IDs and crosswalks (from POI params)."""
    nhm_seg = parameters.parameters["nhm_seg"]
    poi_gage_segment = parameters.parameters["poi_gage_segment"]
    poi_nhm_seg = nhm_seg[poi_gage_segment - 1]  # fortran indexing
    return {
        # "nhm_seg": nhm_seg,
        "poi_ids": poi_nhm_seg,
        # "poi_id_nhm_seg": dict(
        #     zip(parameters.parameters["poi_gage_id"], poi_nhm_seg.tolist())
        # ),
        # "poi_nhm_seg_id": dict(
        #     zip(poi_nhm_seg.tolist(), parameters.parameters["poi_gage_id"])
        # ),
    }


def test_output_monthly_accumulations(
    simulation, control, parameters, nhm_processes, tmp_path
):
    """Test monthly accumulations."""
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

    output = pws.base.Output(
        control=control,
        model=model,
        monthly_accum_var_list=var_list,
        netcdf_output_action="allow",
    )

    model.run(finalize=True, output_obj=output)

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

    # Validate against netcdf output by post-processing the mean
    # and comparing to accumulations/n_days_per_month
    for var_name in var_list:
        nc_file = output_dir / f"{var_name}.nc"
        assert nc_file.exists(), f"NetCDF file {nc_file} not created"

        # Load netcdf data
        ds = xr.open_dataset(nc_file)

        # Calculate monthly means from netcdf
        monthly_mean_nc = ds[var_name].resample(time="1MS").mean()

        # Calculate monthly means from Output accumulations
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


def test_output_noi_data(
    simulation, control, parameters, nhm_processes, poi_info, tmp_path
):
    """Test NOI collection and hierarchical statistics."""
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

    def mean_stat(da: xr.DataArray):
        return da.mean(dim="time")

    output = pws.base.Output(
        control=control,
        model=model,
        noi_var_list=var_list,
        noi_ids=poi_info["poi_ids"],
        noi_stats={
            mean_stat: var_list,
            median_by_month: var_list,
            median_monthly: var_list,
            max_5day: var_list,
        },
        netcdf_output_action="allow",
    )

    model.run(finalize=True, output_obj=output)

    # Check that NOI arrays were created
    assert output.noi_arrays is not None
    assert "seg_outflow" in output.noi_arrays

    # Verify that the last timestep matches the model state
    # (from notebook assertion)
    # Get 0-based indices for checking model state
    noi_indices = np.where(
        np.isin(
            parameters.parameters["nhm_seg"],
            poi_info["poi_ids"],
        )
    )[0]
    assert (
        model.processes["PRMSChannel"]["seg_outflow"][noi_indices]
        == output.noi_arrays["seg_outflow"][-1, :]
    ).all()

    # Check NOI statistics (hierarchical structure)
    assert output.noi_stats is not None
    assert "seg_outflow" in output.noi_stats
    assert "mean_stat" in output.noi_stats["seg_outflow"]
    assert "median_by_month" in output.noi_stats["seg_outflow"]
    assert "median_monthly" in output.noi_stats["seg_outflow"]
    assert "max_5day" in output.noi_stats["seg_outflow"]

    # Check DataArray metadata (name and attrs)
    mean_stat_da = output.noi_stats["seg_outflow"]["mean_stat"]
    assert mean_stat_da.name == "seg_outflow"
    assert mean_stat_da.attrs["variable"] == "seg_outflow"
    assert mean_stat_da.attrs["statistic"] == "mean_stat"
    assert "period_of_record" in mean_stat_da.attrs
    assert " to " in mean_stat_da.attrs["period_of_record"]

    # Check dimensions
    noi_array = output.noi_arrays["seg_outflow"]
    assert isinstance(noi_array, xr.DataArray)
    assert "time" in noi_array.dims
    assert len(noi_array.coords["time"]) == control.n_times

    # Validate against netcdf output by post-processing
    for var_name in var_list:
        nc_file = output_dir / f"{var_name}.nc"
        assert nc_file.exists(), f"NetCDF file {nc_file} not created"

        # Load netcdf data
        ds = xr.open_dataset(nc_file)

        # Extract NOI data from full netcdf output (by nhm_seg ID)
        noi_indices = np.where(
            np.isin(
                parameters.parameters["nhm_seg"],
                poi_info["poi_ids"],
            )
        )[0]
        noi_data_nc = ds[var_name].isel(nhm_seg=noi_indices)

        # Check that full time series matches
        np.testing.assert_allclose(
            noi_data_nc.values,
            output.noi_arrays[var_name].values,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"{var_name}: NOI arrays don't match netcdf",
        )

        # Calculate and validate statistics from netcdf
        # Mean over all time
        mean_nc = noi_data_nc.mean(dim="time")
        np.testing.assert_allclose(
            mean_nc.values,
            output.noi_stats[var_name]["mean_stat"].values,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"{var_name}: mean statistic doesn't match",
        )

        # Median grouped by month
        median_by_month_nc = noi_data_nc.groupby("time.month").median(
            dim="time"
        )
        np.testing.assert_allclose(
            median_by_month_nc.values,
            output.noi_stats[var_name]["median_by_month"].values,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"{var_name}: median_by_month statistic doesn't match",
        )

        # Median resampled to monthly
        median_resample_nc = noi_data_nc.resample(time="1MS").median(
            dim="time"
        )
        np.testing.assert_allclose(
            median_resample_nc.values,
            output.noi_stats[var_name]["median_monthly"].values,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"{var_name}: median_monthly statistic doesn't match",
        )

        # Max resampled to 5 day periods
        max_resample_nc = noi_data_nc.resample(time="5D").max(dim="time")
        np.testing.assert_allclose(
            max_resample_nc.values,
            output.noi_stats[var_name]["max_5day"].values,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"{var_name}: max_5day statistic doesn't match",
        )

        ds.close()


def test_output_hoi_subset(
    simulation, control, parameters, nhm_processes, tmp_path
):
    """Test HOI collection and hierarchical statistics."""
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

    output = pws.base.Output(
        control=control,
        model=model,
        hoi_var_list=var_list,
        hoi_ids=[hru_id],
        hoi_stats={
            mean_monthly: var_list,
            max_yearly: var_list,
        },
        netcdf_output_action="allow",
    )

    model.run(finalize=True, output_obj=output)

    # Check that HRU subset arrays were created
    assert output.hoi_arrays is not None
    assert "hru_actet" in output.hoi_arrays
    assert "pkwater_equiv" in output.hoi_arrays

    # Check HRU subset statistics (hierarchical structure)
    assert output.hoi_stats is not None
    assert "hru_actet" in output.hoi_stats
    assert "mean_monthly" in output.hoi_stats["hru_actet"]
    assert "max_yearly" in output.hoi_stats["hru_actet"]

    assert "pkwater_equiv" in output.hoi_stats
    assert "mean_monthly" in output.hoi_stats["pkwater_equiv"]
    assert "max_yearly" in output.hoi_stats["pkwater_equiv"]

    # Check DataArray metadata (name and attrs)
    actet_mean_da = output.hoi_stats["hru_actet"]["mean_monthly"]
    assert actet_mean_da.name == "hru_actet"
    assert actet_mean_da.attrs["variable"] == "hru_actet"
    assert actet_mean_da.attrs["statistic"] == "mean_monthly"
    assert "period_of_record" in actet_mean_da.attrs
    assert " to " in actet_mean_da.attrs["period_of_record"]

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
            output.hoi_arrays[var_name].values.squeeze(),
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"{var_name}: HRU subset arrays don't match netcdf",
        )

        # Calculate and validate statistics from netcdf
        # Mean resampled to monthly
        mean_resample_nc = hru_data_nc.resample(time="1MS").mean(dim="time")
        np.testing.assert_allclose(
            mean_resample_nc.values,
            output.hoi_stats[var_name]["mean_monthly"].values.squeeze(),
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"{var_name}: mean_monthly statistic doesn't match",
        )

        # Max resampled to yearly
        max_resample_nc = hru_data_nc.resample(time="1YS").max(dim="time")
        np.testing.assert_allclose(
            max_resample_nc.values,
            output.hoi_stats[var_name]["max_yearly"].values.squeeze(),
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"{var_name}: max_yearly statistic doesn't match",
        )

        ds.close()


def test_output_combined(
    simulation, control, parameters, nhm_processes, poi_info, tmp_path
):
    """Test combined monthly/NOI/HOI output."""
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

    def mean(da: xr.DataArray):
        return da.mean(dim="time")

    output = pws.base.Output(
        control=control,
        model=model,
        monthly_accum_var_list=["sroff", "hru_actet", "seg_outflow"],
        noi_var_list=["seg_outflow"],
        noi_ids=poi_info["poi_ids"],
        noi_stats={
            mean: ["seg_outflow"],
            median_by_month: ["seg_outflow"],
            median_monthly: ["seg_outflow"],
            max_5day: ["seg_outflow"],
        },
        hoi_var_list=["hru_actet", "pkwater_equiv"],
        hoi_ids=[hru_id],
        hoi_stats={
            mean_monthly: ["hru_actet", "pkwater_equiv"],
            max_yearly: ["hru_actet", "pkwater_equiv"],
        },
        netcdf_output_action="allow",
    )

    model.run(finalize=True, output_obj=output)

    # Verify all outputs are available
    assert output.monthly_accumulations is not None
    assert output.noi_arrays is not None
    assert output.noi_stats is not None
    assert output.hoi_arrays is not None
    assert output.hoi_stats is not None

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

    # NOI check
    ds_seg = xr.open_dataset(output_dir / "seg_outflow.nc")
    noi_indices = np.where(
        np.isin(
            parameters.parameters["nhm_seg"],
            poi_info["poi_ids"],
        )
    )[0]
    noi_data_nc = ds_seg["seg_outflow"].isel(nhm_seg=noi_indices)
    mean_nc = noi_data_nc.mean(dim="time")
    np.testing.assert_allclose(
        mean_nc.values,
        output.noi_stats["seg_outflow"]["mean"].values,
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
        output.hoi_stats["hru_actet"]["mean_monthly"].values.squeeze(),
        rtol=1e-10,
        atol=1e-10,
    )
    ds_actet.close()


def test_output_properties_before_finalization(
    simulation, control, parameters, nhm_processes, poi_info
):
    """Test properties return None before finalization."""
    model = pws.Model(
        nhm_processes,
        control=control,
        parameters=parameters,
    )

    def mean(da: xr.DataArray):
        return da.mean(dim="time")

    output = pws.base.Output(
        control=control,
        model=model,
        monthly_accum_var_list=["sroff"],
        noi_var_list=["seg_outflow"],
        noi_ids=poi_info["poi_ids"],
        noi_stats={mean: ["seg_outflow"]},
        netcdf_output_action="allow",
    )

    # Properties should return None before finalization
    assert output.monthly_accumulations is None
    assert output.noi_arrays is None
    assert output.noi_stats is None
    assert output.n_days_per_month is None


def test_output_noi_requires_segments(
    simulation, control, parameters, nhm_processes
):
    """Test NOI requires noi_ids."""
    model = pws.Model(
        nhm_processes,
        control=control,
        parameters=parameters,
    )

    # Should raise ValueError if noi_var_list provided without IDs
    with pytest.raises(ValueError, match="noi_ids must be passed"):
        output = pws.base.Output(  # noqa:F841
            control=control,
            model=model,
            noi_var_list=["seg_outflow"],
            netcdf_output_action="allow",
        )


def test_output_string_stats(
    simulation, control, parameters, nhm_processes, poi_info
):
    """Test built-in statistics from time_stats module."""
    from pywatershed.analysis.time_stats import mean, median, std

    model = pws.Model(
        nhm_processes,
        control=control,
        parameters=parameters,
    )

    output = pws.base.Output(
        control=control,
        model=model,
        noi_var_list=["seg_outflow"],
        noi_ids=poi_info["poi_ids"],
        noi_stats={
            mean: ["seg_outflow"],
            median: ["seg_outflow"],
            std: ["seg_outflow"],
        },
        netcdf_output_action="allow",
    )

    model.run(finalize=True, output_obj=output)

    # Verify all built-in stats were calculated (hierarchical structure)
    assert "seg_outflow" in output.noi_stats
    assert "mean" in output.noi_stats["seg_outflow"]
    assert "median" in output.noi_stats["seg_outflow"]
    assert "std" in output.noi_stats["seg_outflow"]


def test_output_validation_invalid_noi_stats(
    simulation, control, parameters, nhm_processes, poi_info
):
    """Test noi_stats validation: keys must be callable."""
    model = pws.Model(
        nhm_processes,
        control=control,
        parameters=parameters,
    )

    with pytest.raises(ValueError, match="noi_stats keys must be callable"):
        output = pws.base.Output(  # noqa:F841
            control=control,
            model=model,
            noi_var_list=["seg_outflow"],
            noi_ids=poi_info["poi_ids"],
            noi_stats={"not_a_function": ["seg_outflow"]},
            netcdf_output_action="allow",
        )


def test_output_validation_invalid_hoi_stats(
    simulation, control, parameters, nhm_processes, poi_info
):
    """Test hoi_stats validation: keys must be callable."""
    model = pws.Model(
        nhm_processes,
        control=control,
        parameters=parameters,
    )

    with pytest.raises(ValueError, match="hoi_stats keys must be callable"):
        output = pws.base.Output(  # noqa:F841
            control=control,
            model=model,
            hoi_var_list=["hru_actet"],
            hoi_ids=[1],
            hoi_stats={"not_a_function": ["hru_actet"]},
            netcdf_output_action="allow",
        )


def test_output_property_warning_before_finalization(
    simulation, control, parameters, nhm_processes, poi_info
):
    """Test property access warns before finalization."""
    model = pws.Model(
        nhm_processes,
        control=control,
        parameters=parameters,
    )

    def mean_stat(da: xr.DataArray):
        return da.mean(dim="time")

    output = pws.base.Output(
        control=control,
        model=model,
        monthly_accum_var_list=["sroff"],
        noi_var_list=["seg_outflow"],
        noi_ids=poi_info["poi_ids"],
        noi_stats={mean_stat: ["seg_outflow"]},
        netcdf_output_action="allow",
    )

    # Access properties before finalization should warn
    with pytest.warns(UserWarning, match="only available after finalization"):
        result = output.monthly_accumulations
        assert result is None

    with pytest.warns(UserWarning, match="only available after finalization"):
        result = output.noi_stats
        assert result is None


def test_output_dict_mode_per_variable_ids(
    simulation, control, parameters, nhm_processes, poi_info, tmp_path
):
    """Test dict mode with per-variable IDs for NOI and HOI."""
    tmp_path = pl.Path(tmp_path)

    model = pws.Model(
        nhm_processes,
        control=control,
        parameters=parameters,
    )

    # Define stats
    def mean_stat(da: xr.DataArray):
        return da.mean(dim="time")

    # Dict mode: different segments/HRUs for different variables
    # Include variables without stats to test arrays but not stats
    output_kwargs = {
        "control": control,
        "model": model,
        "noi_ids": {
            "seg_outflow": poi_info["poi_ids"][:2].tolist(),
            "seg_upstream_inflow": poi_info["poi_ids"][1:3].tolist(),
            "seg_lateral_inflow": poi_info["poi_ids"][
                2:4
            ].tolist(),  # No stats
        },
        "noi_stats": {mean_stat: ["seg_outflow", "seg_upstream_inflow"]},
        "hoi_ids": {
            "hru_actet": [parameters.parameters["nhm_id"][0].tolist()],
            "pkwater_equiv": [
                parameters.parameters["nhm_id"][1].tolist(),
                parameters.parameters["nhm_id"][2].tolist(),
            ],
            "soil_moist": [
                parameters.parameters["nhm_id"][3].tolist()
            ],  # No stats
        },
        "hoi_stats": {mean_stat: ["hru_actet", "pkwater_equiv"]},
        "netcdf_output_action": "allow",
    }

    # from pprint import pprint
    # pprint("output_kwargs: ")
    # pprint(output_kwargs, indent=4, width=120)

    output = pws.base.Output(**output_kwargs)

    model.run(finalize=True, output_obj=output)

    # Check NOI arrays have correct shapes (per-variable segments)
    assert output.noi_arrays["seg_outflow"].shape[1] == 2
    assert output.noi_arrays["seg_upstream_inflow"].shape[1] == 2
    assert output.noi_arrays["seg_lateral_inflow"].shape[1] == 2

    # Check HOI arrays have correct shapes (per-variable HRUs)
    assert output.hoi_arrays["hru_actet"].shape[1] == 1
    assert output.hoi_arrays["pkwater_equiv"].shape[1] == 2
    assert output.hoi_arrays["soil_moist"].shape[1] == 1

    # Check variables without stats exist in arrays but NOT in stats
    assert "seg_lateral_inflow" in output.noi_arrays
    assert "seg_lateral_inflow" not in output.noi_stats
    assert "soil_moist" in output.hoi_arrays
    assert "soil_moist" not in output.hoi_stats

    # Check stats calculated correctly for variables with stats
    assert "seg_outflow" in output.noi_stats
    assert "mean_stat" in output.noi_stats["seg_outflow"]
    assert "seg_upstream_inflow" in output.noi_stats
    assert "mean_stat" in output.noi_stats["seg_upstream_inflow"]

    assert "hru_actet" in output.hoi_stats
    assert "mean_stat" in output.hoi_stats["hru_actet"]
    assert "pkwater_equiv" in output.hoi_stats
    assert "mean_stat" in output.hoi_stats["pkwater_equiv"]

    # Verify coordinates match requested IDs
    # NOI coordinates
    np.testing.assert_array_equal(
        output.noi_arrays["seg_outflow"].coords["nhm_seg"].values,
        poi_info["poi_ids"][:2],
    )
    np.testing.assert_array_equal(
        output.noi_arrays["seg_upstream_inflow"].coords["nhm_seg"].values,
        poi_info["poi_ids"][1:3],
    )
    np.testing.assert_array_equal(
        output.noi_arrays["seg_lateral_inflow"].coords["nhm_seg"].values,
        poi_info["poi_ids"][2:4],
    )

    # HOI coordinates
    np.testing.assert_array_equal(
        output.hoi_arrays["hru_actet"].coords["nhm_id"].values,
        [parameters.parameters["nhm_id"][0]],
    )
    np.testing.assert_array_equal(
        output.hoi_arrays["pkwater_equiv"].coords["nhm_id"].values,
        parameters.parameters["nhm_id"][1:3],
    )
    np.testing.assert_array_equal(
        output.hoi_arrays["soil_moist"].coords["nhm_id"].values,
        [parameters.parameters["nhm_id"][3]],
    )

    # Verify stat coordinates also match (stats inherit from arrays)
    np.testing.assert_array_equal(
        output.noi_stats["seg_outflow"]["mean_stat"].coords["nhm_seg"].values,
        poi_info["poi_ids"][:2],
    )
    np.testing.assert_array_equal(
        output.hoi_stats["hru_actet"]["mean_stat"].coords["nhm_id"].values,
        [parameters.parameters["nhm_id"][0]],
    )


def test_output_with_flow_graph(
    simulation, control, parameters, poi_info, tmp_path
):
    """Test Output with FlowGraph variables (node_outflows, etc.)."""
    tmp_path = pl.Path(tmp_path)
    domain_dir = simulation["dir"]
    input_dir = simulation["output_dir"]

    # this test requires pws style invokation with params and dis
    dis_file = domain_dir / "parameters_dis_hru.nc"
    dis_hru = pws.Parameters.from_netcdf(dis_file, encoding=False)

    dis_both_file = domain_dir / "parameters_dis_both.nc"
    dis_both = pws.Parameters.from_netcdf(dis_both_file, encoding=False)

    # Setup control for FlowGraph
    control.options = control.options | {
        "input_dir": input_dir,
        "imbalance_behavior": "warn",
        "calc_method": "numba",
    }

    # Variables to track - FlowGraph equivalents
    flow_graph_vars = ["node_outflows", "node_upstream_inflows"]

    # Setup netcdf output for comparison
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    control.options["netcdf_output_dir"] = output_dir
    control.options["netcdf_output_var_names"] = flow_graph_vars

    # Build model with FlowGraph instead of PRMSChannel
    nhm_processes = [
        pws.PRMSRunoff,
        pws.PRMSSoilzone,
        pws.PRMSGroundwater,
    ]

    model_dict = {
        "control": control,
        "dis_both": dis_hru,
        "dis_hru": dis_both,
        "model_order": [],
    }

    # Add PRMS processes
    for proc in nhm_processes:
        proc_name = proc.__name__
        proc_rename = "prms_" + proc_name[4:].lower()
        model_dict["model_order"] += [proc_rename]
        model_dict[proc_rename] = {}
        proc_dict = model_dict[proc_rename]
        proc_dict["class"] = proc
        proc_param_file = domain_dir / f"parameters_{proc_name}.nc"
        proc_dict["parameters"] = Parameters.from_netcdf(proc_param_file)
        proc_dict["dis"] = "dis_hru"

    # Add FlowGraph with some pass-through nodes
    nsegs = len(parameters.parameters["nhm_seg"])
    rando = ((0, int(nsegs / 2), -1),)
    random_seg_ids = parameters.parameters["nhm_seg"][rando]
    n_new_nodes = len(random_seg_ids)

    check_names = ["pass"] * n_new_nodes
    check_indices = list(range(n_new_nodes))
    check_ids = list(range(n_new_nodes))
    prms_channel_node_maker_name = "PRMS-CHANNEL"

    model_dict = prms_channel_flow_graph_to_model_dict(
        model_dict=model_dict,
        prms_channel_dis=parameters,
        prms_channel_dis_name="dis_both",
        prms_channel_params=parameters,
        new_nodes_maker_dict={"pass": PassThroughFlowNodeMaker()},
        new_nodes_maker_names=check_names,
        new_nodes_maker_indices=check_indices,
        new_nodes_maker_ids=check_ids,
        new_nodes_flow_to_nhm_seg=random_seg_ids,
        graph_imbalance_behavior="warn",
        addtl_output_vars=["outflow_substep"],
        prms_channel_node_maker_name=prms_channel_node_maker_name,
    )

    model = Model(model_dict)

    # Create Output with FlowGraph variables
    monthly_accum_var_list = ["node_outflows", "outflow_substep"]
    noi_ids = {
        "node_outflows": list(
            zip(
                [prms_channel_node_maker_name] * len(poi_info["poi_ids"]),
                poi_info["poi_ids"],
            )
        ),
        "outflow_substep": list(zip(["pass"] * len(check_ids), check_ids)),
    }

    def mean(da: xr.DataArray):
        return da.mean(dim="time")

    output = pws.base.Output(
        control=control,
        model=model,
        monthly_accum_var_list=monthly_accum_var_list,
        noi_ids=noi_ids,
        noi_stats={mean: list(noi_ids.keys())},
        netcdf_output_action="allow",
    )

    # Run model
    model.run(finalize=True, output_obj=output)

    # Verify Output collected data from FlowGraph
    assert output.monthly_accumulations is not None
    for vv in monthly_accum_var_list:
        assert vv in output.monthly_accumulations
        assert isinstance(output.monthly_accumulations[vv], xr.DataArray)

    assert output.noi_arrays is not None
    for vv in noi_ids.keys():
        assert vv in output.noi_arrays
        assert isinstance(output.noi_arrays[vv], xr.DataArray)

    assert output.noi_stats is not None
    for vv in noi_ids.keys():
        assert vv in output.noi_stats
        assert "mean" in output.noi_stats[vv].keys()

    # Validate against netcdf output
    # Check monthly accumulation
    for vv in monthly_accum_var_list:
        nc_file = output_dir / f"{vv}.nc"
        assert nc_file.exists(), "NetCDF file not found: {nc_file}"
        ds = xr.load_dataarray(nc_file)
        # test n_days_per_month
        monthly_nc = ds.resample(time="1MS").mean()
        custom_monthly = (
            output.monthly_accumulations[vv] / output.n_days_per_month
        )
        assert "month" in custom_monthly.dims
        assert custom_monthly.shape[0] == len(output.time_months)
        np.testing.assert_allclose(
            monthly_nc.values,
            custom_monthly.values,
            rtol=1e-10,
            atol=1e-10,
            err_msg="FlowGraph mean statistic doesn't match",
        )

    for vv, ids in noi_ids.items():
        nc_file = output_dir / f"{vv}.nc"
        assert nc_file.exists(), "NetCDF file not found: {nc_file}"
        ds = xr.load_dataarray(nc_file)
        noi_data_nc = ds.sel(node_coord=output.noi_arrays[vv].node_coord)

        # Check that NOI arrays match
        np.testing.assert_allclose(
            noi_data_nc.values,
            output.noi_arrays[vv].values,
            rtol=1e-10,
            atol=1e-10,
            err_msg="FlowGraph NOI arrays don't match netcdf",
        )


def test_output_obj_kwargs_dict_basic(
    simulation, control, parameters, nhm_processes, poi_info, tmp_path
):
    """Test basic usage of output_obj_kwargs_dict in Model.__init__."""
    tmp_path = pl.Path(tmp_path)

    # Variables to track
    mon_var_list = ["sroff", "hru_actet"]
    noi_ids = poi_info["poi_ids"]

    def mean_stat(da: xr.DataArray):
        return da.mean(dim="time")

    # Create Model with output_obj_kwargs_dict instead of separate Output
    model = pws.Model(
        nhm_processes,
        control=control,
        parameters=parameters,
        output_obj_kwargs_dict={
            "monthly_accum_var_list": mon_var_list,
            "noi_ids": {"seg_outflow": noi_ids},
            "noi_stats": {mean_stat: ["seg_outflow"]},
        },
    )

    # Verify output_obj was created and is accessible
    assert model.output_obj is not None
    assert isinstance(model.output_obj, pws.base.Output)

    # Verify the Output was configured with the provided kwargs
    assert model.output_obj._monthly_accum_var_list == mon_var_list

    # TODO these are inconsistently stored but they are private
    # assert model.output_obj._noi_var_list == noi_var_list
    # assert (
    #     model.output_obj._noi_ids["seg_outflow"] == poi_info["poi_ids"]
    # ).all()

    # Run the model with the auto-created output object
    model.run(finalize=True)

    assert model.output_obj.monthly_accumulations is not None
    assert "sroff" in model.output_obj.monthly_accumulations
    assert "hru_actet" in model.output_obj.monthly_accumulations
    assert model.output_obj.noi_arrays is not None
    assert "seg_outflow" in model.output_obj.noi_arrays
    assert model.output_obj.noi_stats is not None
    assert "seg_outflow" in model.output_obj.noi_stats
    assert "mean_stat" in model.output_obj.noi_stats["seg_outflow"]


def test_output_obj_kwargs_dict_wrong_control_raises(
    simulation, control, parameters, nhm_processes, tmp_path
):
    """Test wrong control in output_obj_kwargs_dict raises ValueError."""
    tmp_path = pl.Path(tmp_path)

    # Create a different control object
    wrong_control = Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )
    wrong_control.edit_end_time(np.datetime64("1979-06-01T00:00:00"))

    # Attempt to create Model with wrong control should raise ValueError
    with pytest.raises(ValueError, match="inappropriate.*control"):
        pws.Model(
            nhm_processes,
            control=control,
            parameters=parameters,
            output_obj_kwargs_dict={
                "control": wrong_control,  # Wrong control!
                "monthly_accum_var_list": ["sroff"],
            },
        )


def test_output_obj_kwargs_dict_wrong_model_raises(
    simulation, control, parameters, nhm_processes, tmp_path
):
    """Test passing a model in output_obj_kwargs_dict raises ValueError."""
    tmp_path = pl.Path(tmp_path)

    # Create a dummy model object
    class DummyModel:
        pass

    wrong_model = DummyModel()

    # Attempt to create Model with wrong model should raise ValueError
    with pytest.raises(ValueError, match="inappropriate.*model"):
        pws.Model(
            nhm_processes,
            control=control,
            parameters=parameters,
            output_obj_kwargs_dict={
                "model": wrong_model,  # Wrong model!
                "monthly_accum_var_list": ["sroff"],
            },
        )


def test_output_zarr_chunked(
    simulation, control, parameters, nhm_processes, tmp_path
):
    """Test zarr chunked output and compare to netcdf output."""
    tmp_path = pl.Path(tmp_path)

    # Variables to write with both methods
    var_list = ["sroff", "seg_outflow"]

    # Setup netcdf output for comparison
    netcdf_dir = tmp_path / "netcdf"
    netcdf_dir.mkdir()
    control.options["netcdf_output_dir"] = netcdf_dir
    control.options["netcdf_output_var_names"] = var_list

    # Setup zarr output
    zarr_file = tmp_path / "output.zarr"
    chunk_sizes = {
        "time": 30,  # 1 month chunks for testing
        "nhru": 500,
        "nsegment": 500,
    }

    # Create model
    model = pws.Model(
        nhm_processes,
        control=control,
        parameters=parameters,
    )

    # Create Output object with zarr chunked output
    output = pws.base.Output(
        control=control,
        model=model,
        chunked_var_list=var_list,
        chunked_output_file=zarr_file,
        chunk_sizes=chunk_sizes,
        chunk_size_auto_warn=False,
        netcdf_output_action="allow",
    )

    # Run model once with both netcdf and zarr output
    model.run(finalize=True, output_obj=output)

    # Verify zarr file was created
    assert zarr_file.exists(), f"Zarr file not created: {zarr_file}"

    # Load zarr data
    ds_zarr = xr.open_zarr(zarr_file)

    # Compare each variable
    for vv in var_list:
        # Load netcdf data
        nc_file = netcdf_dir / f"{vv}.nc"
        assert nc_file.exists(), f"NetCDF file not created: {nc_file}"
        ds_netcdf = xr.open_dataset(nc_file)

        # Check shapes match
        assert ds_zarr[vv].shape == ds_netcdf[vv].shape, (
            f"{vv}: shapes don't match - "
            f"zarr: {ds_zarr[vv].shape}, netcdf: {ds_netcdf[vv].shape}"
        )

        # Check values match (allowing for floating point tolerance)
        np.testing.assert_allclose(
            ds_zarr[vv].values,
            ds_netcdf[vv].values,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"{vv}: values don't match between zarr and netcdf",
        )

        # Check time coordinate matches
        np.testing.assert_array_equal(
            ds_zarr.time.values,
            ds_netcdf.time.values,
            err_msg=f"{vv}: time coordinates don't match",
        )

        ds_netcdf.close()

    ds_zarr.close()


def test_output_zarr_auto_chunk_sizes(
    simulation, control, parameters, nhm_processes, tmp_path
):
    """Test zarr chunked output with auto-determined chunk sizes."""
    tmp_path = pl.Path(tmp_path)

    var_list = ["sroff"]
    zarr_file = tmp_path / "output_auto.zarr"

    model = pws.Model(
        nhm_processes,
        control=control,
        parameters=parameters,
    )

    # Should warn about auto-determined chunk sizes
    with pytest.warns(UserWarning, match="Chunk sizes not specified"):
        output = pws.base.Output(
            control=control,
            model=model,
            chunked_var_list=var_list,
            chunked_output_file=zarr_file,
            chunk_sizes=None,  # Will auto-determine
            chunk_size_auto_warn=True,
        )

    model.run(finalize=True, output_obj=output)

    # Verify zarr file was created
    assert zarr_file.exists()

    # Load and verify data
    ds_zarr = xr.open_zarr(zarr_file)
    assert "sroff" in ds_zarr
    assert ds_zarr["sroff"].shape[0] == control.n_times
    ds_zarr.close()


def test_output_zarr_no_warn(
    simulation, control, parameters, nhm_processes, tmp_path
):
    """Test zarr with chunk_size_auto_warn=False doesn't warn."""
    tmp_path = pl.Path(tmp_path)

    var_list = ["sroff"]
    zarr_file = tmp_path / "output_nowarn.zarr"

    model = pws.Model(
        nhm_processes,
        control=control,
        parameters=parameters,
    )

    # Should not warn when chunk_size_auto_warn=False
    output = pws.base.Output(
        control=control,
        model=model,
        chunked_var_list=var_list,
        chunked_output_file=zarr_file,
        chunk_sizes=None,
        chunk_size_auto_warn=False,
    )

    model.run(finalize=True, output_obj=output)
    assert zarr_file.exists()


def test_output_zarr_requires_filename(
    simulation, control, parameters, nhm_processes, tmp_path
):
    """Test that zarr output requires chunked_output_file."""
    tmp_path = pl.Path(tmp_path)

    model = pws.Model(
        nhm_processes,
        control=control,
        parameters=parameters,
    )

    # Should raise ValueError when file not provided
    with pytest.raises(
        ValueError, match="chunked_output_file must be provided"
    ):
        pws.base.Output(
            control=control,
            model=model,
            chunked_var_list=["sroff"],
            chunked_output_file=None,  # Missing!
        )


def test_output_zarr_chunked_flow_graph(
    simulation, control, parameters, poi_info, tmp_path
):
    """Test zarr chunked output with FlowGraph variables."""
    tmp_path = pl.Path(tmp_path)
    domain_dir = simulation["dir"]
    input_dir = simulation["output_dir"]

    # this test requires pws style invokation with params and dis
    dis_file = domain_dir / "parameters_dis_hru.nc"
    dis_hru = pws.Parameters.from_netcdf(dis_file, encoding=False)

    dis_both_file = domain_dir / "parameters_dis_both.nc"
    dis_both = pws.Parameters.from_netcdf(dis_both_file, encoding=False)

    # Setup control for FlowGraph
    control.options = control.options | {
        "input_dir": input_dir,
        "imbalance_behavior": "warn",
        "calc_method": "numba",
    }

    # Variables to track - FlowGraph equivalents
    flow_graph_vars = ["node_outflows", "node_upstream_inflows"]

    # Setup netcdf output for comparison
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    control.options["netcdf_output_dir"] = output_dir
    control.options["netcdf_output_var_names"] = flow_graph_vars

    # Build model with FlowGraph instead of PRMSChannel
    nhm_processes = [
        pws.PRMSRunoff,
        pws.PRMSSoilzone,
        pws.PRMSGroundwater,
    ]

    model_dict = {
        "control": control,
        "dis_both": dis_hru,
        "dis_hru": dis_both,
        "model_order": [],
    }

    # Add PRMS processes
    for proc in nhm_processes:
        proc_name = proc.__name__
        proc_rename = "prms_" + proc_name[4:].lower()
        model_dict["model_order"] += [proc_rename]
        model_dict[proc_rename] = {}
        proc_dict = model_dict[proc_rename]
        proc_dict["class"] = proc
        proc_param_file = domain_dir / f"parameters_{proc_name}.nc"
        proc_dict["parameters"] = Parameters.from_netcdf(proc_param_file)
        proc_dict["dis"] = "dis_hru"

    # Add FlowGraph with some pass-through nodes
    nsegs = len(parameters.parameters["nhm_seg"])
    rando = ((0, int(nsegs / 2), -1),)
    random_seg_ids = parameters.parameters["nhm_seg"][rando]
    n_new_nodes = len(random_seg_ids)

    check_names = ["pass"] * n_new_nodes
    check_indices = list(range(n_new_nodes))
    check_ids = list(range(n_new_nodes))
    prms_channel_node_maker_name = "PRMS-CHANNEL"

    model_dict = prms_channel_flow_graph_to_model_dict(
        model_dict=model_dict,
        prms_channel_dis=parameters,
        prms_channel_dis_name="dis_both",
        prms_channel_params=parameters,
        new_nodes_maker_dict={"pass": PassThroughFlowNodeMaker()},
        new_nodes_maker_names=check_names,
        new_nodes_maker_indices=check_indices,
        new_nodes_maker_ids=check_ids,
        new_nodes_flow_to_nhm_seg=random_seg_ids,
        graph_imbalance_behavior="warn",
        addtl_output_vars=["outflow_substep"],
        prms_channel_node_maker_name=prms_channel_node_maker_name,
    )

    model = Model(model_dict)

    # Setup zarr output
    zarr_file = tmp_path / "flowgraph_output.zarr"
    chunk_sizes = {
        "time": 30,  # 1 month chunks for testing
        "nnode": 500,
    }

    # Create Output with FlowGraph variables and zarr output
    output = pws.base.Output(
        control=control,
        model=model,
        chunked_var_list=flow_graph_vars,
        chunked_output_file=zarr_file,
        chunk_sizes=chunk_sizes,
        chunk_size_auto_warn=False,
        netcdf_output_action="allow",
    )

    # Run model
    model.run(finalize=True, output_obj=output)

    # Verify zarr file was created
    assert zarr_file.exists(), f"Zarr file not created: {zarr_file}"

    # Load zarr data
    ds_zarr = xr.open_zarr(zarr_file)

    # Compare each variable with netcdf
    for vv in flow_graph_vars:
        # Load netcdf data
        nc_file = output_dir / f"{vv}.nc"
        assert nc_file.exists(), f"NetCDF file not created: {nc_file}"
        ds_netcdf = xr.open_dataset(nc_file)

        # Check shapes match
        assert ds_zarr[vv].shape == ds_netcdf[vv].shape, (
            f"{vv}: shapes don't match - "
            f"zarr: {ds_zarr[vv].shape}, netcdf: {ds_netcdf[vv].shape}"
        )

        # Check values match (allowing for floating point tolerance)
        np.testing.assert_allclose(
            ds_zarr[vv].values,
            ds_netcdf[vv].values,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"{vv}: values don't match between zarr and netcdf",
        )

        # Check time coordinate matches
        np.testing.assert_array_equal(
            ds_zarr.time.values,
            ds_netcdf.time.values,
            err_msg=f"{vv}: time coordinates don't match",
        )

        ds_netcdf.close()

    ds_zarr.close()
