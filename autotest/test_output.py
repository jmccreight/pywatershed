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
    """POI information with nhm_seg IDs and crosswalks."""
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


def test_output_poi_data(
    simulation, control, parameters, nhm_processes, poi_info, tmp_path
):
    """Test POI collection and hierarchical statistics."""
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
        poi_var_list=var_list,
        poi_nhm_seg=poi_info["poi_nhm_seg"],
        poi_gage_segment=poi_info["poi_gage_segment"] - 1,
        poi_stats={
            mean_stat: var_list,
            median_by_month: var_list,
            median_monthly: var_list,
            max_5day: var_list,
        },
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

    # Check POI statistics (hierarchical structure)
    assert output.poi_stats is not None
    assert "seg_outflow" in output.poi_stats
    assert "mean_stat" in output.poi_stats["seg_outflow"]
    assert "median_by_month" in output.poi_stats["seg_outflow"]
    assert "median_monthly" in output.poi_stats["seg_outflow"]
    assert "max_5day" in output.poi_stats["seg_outflow"]

    # Check DataArray metadata (name and attrs)
    mean_stat_da = output.poi_stats["seg_outflow"]["mean_stat"]
    assert mean_stat_da.name == "seg_outflow"
    assert mean_stat_da.attrs["variable"] == "seg_outflow"
    assert mean_stat_da.attrs["statistic"] == "mean_stat"
    assert "period_of_record" in mean_stat_da.attrs
    assert " to " in mean_stat_da.attrs["period_of_record"]

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
            output.poi_stats[var_name]["mean_stat"].values,
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
            output.poi_stats[var_name]["median_by_month"].values,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"{var_name}: median_by_month statistic doesn't match",
        )

        # Median resampled to monthly
        median_resample_nc = poi_data_nc.resample(time="1MS").median(
            dim="time"
        )
        np.testing.assert_allclose(
            median_resample_nc.values,
            output.poi_stats[var_name]["median_monthly"].values,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"{var_name}: median_monthly statistic doesn't match",
        )

        # Max resampled to 5 day periods
        max_resample_nc = poi_data_nc.resample(time="5D").max(dim="time")
        np.testing.assert_allclose(
            max_resample_nc.values,
            output.poi_stats[var_name]["max_5day"].values,
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
    )

    model.run(finalize=True, output=output)

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
    """Test combined monthly/POI/HOI output."""
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
        poi_var_list=["seg_outflow"],
        poi_nhm_seg=poi_info["poi_nhm_seg"],
        poi_gage_segment=poi_info["poi_gage_segment"] - 1,
        poi_stats={
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
    )

    model.run(finalize=True, output=output)

    # Verify all outputs are available
    assert output.monthly_accumulations is not None
    assert output.poi_arrays is not None
    assert output.poi_stats is not None
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

    # POI check
    ds_seg = xr.open_dataset(output_dir / "seg_outflow.nc")
    poi_indices = poi_info["poi_gage_segment"] - 1
    poi_data_nc = ds_seg["seg_outflow"].isel(nhm_seg=poi_indices)
    mean_nc = poi_data_nc.mean(dim="time")
    np.testing.assert_allclose(
        mean_nc.values,
        output.poi_stats["seg_outflow"]["mean"].values,
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
        poi_var_list=["seg_outflow"],
        poi_nhm_seg=poi_info["poi_nhm_seg"],
        poi_stats={mean: ["seg_outflow"]},
    )

    # Properties should return None before finalization
    assert output.monthly_accumulations is None
    assert output.poi_arrays is None
    assert output.poi_stats is None
    assert output.n_days_per_month is None


def test_output_poi_requires_segments(
    simulation, control, parameters, nhm_processes
):
    """Test POI requires poi_nhm_seg or poi_gage_segment."""
    model = pws.Model(
        nhm_processes,
        control=control,
        parameters=parameters,
    )

    # Should raise ValueError if poi_var_list provided without segments
    with pytest.raises(ValueError, match="poi_nhm_seg or poi_gage_segment"):
        output = pws.base.Output(  # noqa:F841
            control=control,
            model=model,
            poi_var_list=["seg_outflow"],
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
        poi_var_list=["seg_outflow"],
        poi_nhm_seg=poi_info["poi_nhm_seg"],
        poi_stats={
            mean: ["seg_outflow"],
            median: ["seg_outflow"],
            std: ["seg_outflow"],
        },
    )

    model.run(finalize=True, output=output)

    # Verify all built-in stats were calculated (hierarchical structure)
    assert "seg_outflow" in output.poi_stats
    assert "mean" in output.poi_stats["seg_outflow"]
    assert "median" in output.poi_stats["seg_outflow"]
    assert "std" in output.poi_stats["seg_outflow"]


def test_output_validation_invalid_poi_stats(
    simulation, control, parameters, nhm_processes, poi_info
):
    """Test poi_stats validation: keys must be callable."""
    model = pws.Model(
        nhm_processes,
        control=control,
        parameters=parameters,
    )

    with pytest.raises(ValueError, match="poi_stats keys must be callable"):
        output = pws.base.Output(  # noqa:F841
            control=control,
            model=model,
            poi_var_list=["seg_outflow"],
            poi_nhm_seg=poi_info["poi_nhm_seg"],
            poi_stats={"not_a_function": ["seg_outflow"]},
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
        poi_var_list=["seg_outflow"],
        poi_nhm_seg=poi_info["poi_nhm_seg"],
        poi_stats={mean_stat: ["seg_outflow"]},
    )

    # Access properties before finalization should warn
    with pytest.warns(UserWarning, match="only available after finalization"):
        result = output.monthly_accumulations
        assert result is None

    with pytest.warns(UserWarning, match="only available after finalization"):
        result = output.poi_stats
        assert result is None


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
        "budget_type": "warn",
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
        graph_budget_type="warn",
        addtl_output_vars=["outflow_substep"],
    )

    model = Model(model_dict)

    # Get POI indices (only original segments, not new nodes)
    poi_indices = poi_info["poi_gage_segment"] - 1  # 0-based

    # Create Output with FlowGraph variables
    monthly_accum_var_list = ["node_outflows", "outflow_substep"]
    poi_var_list = ["node_outflows", "outflow_substep"]

    def mean(da: xr.DataArray):
        return da.mean(dim="time")

    output = pws.base.Output(
        control=control,
        model=model,
        monthly_accum_var_list=monthly_accum_var_list,
        poi_var_list=poi_var_list,
        poi_gage_segment=poi_indices,
        poi_stats={mean: poi_var_list},
    )

    # Run model
    model.run(finalize=True, output=output)

    # Verify Output collected data from FlowGraph
    assert output.monthly_accumulations is not None
    for vv in monthly_accum_var_list:
        assert vv in output.monthly_accumulations
        assert isinstance(output.monthly_accumulations[vv], xr.DataArray)

    assert output.poi_arrays is not None
    for vv in poi_var_list:
        assert vv in output.poi_arrays
        assert isinstance(output.poi_arrays[vv], xr.DataArray)

    assert output.poi_stats is not None
    for vv in poi_var_list:
        assert vv in output.poi_stats
        assert "mean" in output.poi_stats[vv].keys()

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

    for vv in poi_var_list:
        nc_file = output_dir / f"{vv}.nc"
        assert nc_file.exists(), "NetCDF file not found: {nc_file}"
        ds = xr.load_dataarray(nc_file)
        poi_data_nc = ds.isel(node_coord=poi_indices)

        # Check that POI arrays match
        np.testing.assert_allclose(
            poi_data_nc.values,
            output.poi_arrays[vv].values,
            rtol=1e-10,
            atol=1e-10,
            err_msg="FlowGraph POI arrays don't match netcdf",
        )
