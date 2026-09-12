import pathlib as pl

import numpy as np
import pytest
import xarray as xr

from pywatershed.base.parameters import Parameters

# Three-reach synthetic network: reaches 0 and 1 are headwaters that
# flow into reach 2, which is the outlet.
NSEG = 3
NHRU = 3


def _meta(dims: tuple, units: str) -> dict:
    return {"dims": dims, "attrs": {"units": units}}


@pytest.fixture
def synthetic_params() -> Parameters:
    dims = {"nsegment": NSEG, "nhru": NHRU}
    coords = {
        "nhm_seg": np.array([101, 102, 103], dtype=np.int64),
        "nhm_id": np.array([1, 2, 3], dtype=np.int64),
    }
    data_vars = {
        "tosegment": np.array([3, 3, 0], dtype=np.int64),
        "tosegment_nhm": np.array([103, 103, 0], dtype=np.int64),
        "seg_length": np.array([1000.0, 2000.0, 1500.0]),
        "seg_slope": np.array([0.01, 0.005, 0.002]),
        "mann_n": np.array([0.04, 0.035, 0.03]),
        "seg_width": np.array([5.0, 8.0, 12.0]),
        "seg_depth": np.array([0.5, 0.8, 1.2]),
        "hru_segment": np.array([1, 2, 3], dtype=np.int64),
        "hru_elev": np.array([120.0, 110.0, 100.0]),
    }
    metadata = {
        "global": {},
        "nhm_seg": _meta(("nsegment",), "none"),
        "nhm_id": _meta(("nhru",), "none"),
        "tosegment": _meta(("nsegment",), "none"),
        "tosegment_nhm": _meta(("nsegment",), "none"),
        "seg_length": _meta(("nsegment",), "meters"),
        "seg_slope": _meta(("nsegment",), "decimal fraction"),
        "mann_n": _meta(("nsegment",), "seconds / meter ** (1/3)"),
        "seg_width": _meta(("nsegment",), "meter"),
        "seg_depth": _meta(("nsegment",), "meter"),
        "hru_segment": _meta(("nhru",), "none"),
        "hru_elev": _meta(("nhru",), "meters"),
    }
    return Parameters(
        dims=dims, coords=coords, data_vars=data_vars, metadata=metadata
    )


@pytest.mark.domainless
def test_shear_velocity_values():
    from pywatershed.utils.network_hydraulics import G, shear_velocity

    depth = np.array([1.0, 2.0, 0.0])
    slope = np.array([0.001, 0.0, 0.01])
    result = shear_velocity(depth, slope)
    expected = np.array(
        [np.sqrt(G * 1.0 * 0.001), np.sqrt(G * 2.0 * 1.0e-7), 0.0]
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.domainless
def test_calculate_seg_mid_elevations(synthetic_params):
    from pywatershed.utils.network_hydraulics import (
        calculate_seg_mid_elevations,
    )

    mid, outlet_mid = calculate_seg_mid_elevations(synthetic_params)
    np.testing.assert_allclose(mid, np.array([108.0, 108.0, 101.5]))
    assert outlet_mid == {2: 101.5}


NTIME = 4
TIMES = np.arange(
    np.datetime64("1979-01-01"), np.datetime64("1979-01-05")
).astype("datetime64[ns]")


def _write_run_var(run_dir, name, values, units, nhm_seg):
    da = xr.DataArray(
        values,
        dims=("time", "nhm_seg"),
        coords={"time": TIMES, "nhm_seg": nhm_seg},
        name=name,
        attrs={"units": units},
    )
    da.to_netcdf(run_dir / f"{name}.nc")


@pytest.fixture
def synthetic_run_dir(tmp_path, synthetic_params) -> pl.Path:
    """A fake pywatershed output directory for the three-reach network."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    nhm_seg = synthetic_params.parameters["nhm_seg"]
    base = np.array([[10.0, 20.0, 35.0]])  # cfs, outlet sums the two
    ramp = np.arange(1, NTIME + 1)[:, None]  # 1..4
    outflow = base * ramp
    inflow = outflow * 0.9
    width = np.array([[4.0, 6.0, 10.0]]) * np.ones((NTIME, 1))
    depth = np.array([[0.3, 0.5, 0.9]]) * ramp * 0.5
    velocity = outflow * 0.028316847 / (width * depth)
    res_time = (
        width
        * depth
        * synthetic_params.parameters["seg_length"]
        / (outflow * 0.028316847)
    )
    _write_run_var(run_dir, "seg_outflow", outflow, "cfs", nhm_seg)
    _write_run_var(run_dir, "seg_inflow", inflow, "cfs", nhm_seg)
    _write_run_var(run_dir, "seg_flow_width", width, "meters", nhm_seg)
    _write_run_var(run_dir, "seg_flow_depth", depth, "meters", nhm_seg)
    _write_run_var(
        run_dir, "seg_flow_velocity", velocity, "meters per second", nhm_seg
    )
    _write_run_var(run_dir, "seg_res_time", res_time, "seconds", nhm_seg)
    return run_dir


@pytest.mark.domainless
def test_export_static_fields(synthetic_params, synthetic_run_dir, tmp_path):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    out = export_network_hydraulics(
        synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
    )
    assert out == tmp_path / "net.nc"
    ds = xr.open_dataset(out)
    p = synthetic_params.parameters
    assert ds.sizes["reach"] == NSEG
    assert ds.sizes["time"] == NTIME
    np.testing.assert_array_equal(ds["reach_id"], p["nhm_seg"])
    np.testing.assert_array_equal(ds["to_id"], p["tosegment_nhm"])
    np.testing.assert_array_equal(ds["to_index"], np.array([2, 2, -1]))
    np.testing.assert_array_equal(ds["is_outlet"], np.array([0, 0, 1]))
    np.testing.assert_array_equal(ds["length"], p["seg_length"])
    np.testing.assert_array_equal(ds["slope"], p["seg_slope"])
    np.testing.assert_array_equal(ds["mann_n"], p["mann_n"])
    np.testing.assert_array_equal(ds["bankfull_width"], p["seg_width"])
    np.testing.assert_array_equal(ds["bankfull_depth"], p["seg_depth"])
    np.testing.assert_allclose(
        ds["elevation_mid"], np.array([108.0, 108.0, 101.5])
    )
    assert "vertex" not in ds.dims
    assert "x_mid" not in ds
    assert ds["length"].attrs["units"] == "m"
    assert ds["to_index"].attrs["source_name"] == "tosegment"
    assert ds.attrs["source_model"] == "pywatershed PRMS"
    assert "pywatershed_version" in ds.attrs
    assert ds.attrs["n_unconnected"] == -1  # no polyline supplied
    ds.close()


@pytest.mark.domainless
def test_export_time_varying_fields(
    synthetic_params, synthetic_run_dir, tmp_path
):
    from pywatershed.hydrology.prms_hydraulic_geometry import CFS_TO_CMS
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
        shear_velocity,
    )

    out = export_network_hydraulics(
        synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
    )
    ds = xr.open_dataset(out)
    src = {
        nm: xr.open_dataarray(synthetic_run_dir / f"{nm}.nc").load()
        for nm in [
            "seg_outflow",
            "seg_inflow",
            "seg_flow_width",
            "seg_flow_depth",
            "seg_flow_velocity",
            "seg_res_time",
        ]
    }
    np.testing.assert_array_equal(ds["time"], TIMES)
    np.testing.assert_allclose(
        ds["flow_out"], src["seg_outflow"].values * CFS_TO_CMS
    )
    np.testing.assert_allclose(
        ds["flow_in"], src["seg_inflow"].values * CFS_TO_CMS
    )
    np.testing.assert_allclose(ds["width"], src["seg_flow_width"].values)
    np.testing.assert_allclose(ds["depth"], src["seg_flow_depth"].values)
    np.testing.assert_allclose(ds["velocity"], src["seg_flow_velocity"].values)
    np.testing.assert_allclose(
        ds["residence_time"], src["seg_res_time"].values
    )
    expected_ustar = shear_velocity(
        src["seg_flow_depth"].values,
        synthetic_params.parameters["seg_slope"][None, :],
    )
    np.testing.assert_allclose(ds["ustar"], expected_ustar)
    assert ds["flow_out"].attrs["units"] == "m3 s-1"
    assert ds["flow_out"].attrs["source_name"] == "seg_outflow"
    assert ds["ustar"].attrs["method"] == "sqrt(g*depth*slope)"
    assert ds["velocity"].attrs["method"] == "power_law_at_a_station"
    assert ds["flow_out"].dims == ("time", "reach")
    assert "water_temperature" not in ds
    ds.close()


@pytest.mark.domainless
def test_export_optional_temperature(
    synthetic_params, synthetic_run_dir, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    temp = np.full((NTIME, NSEG), 12.5)
    _write_run_var(
        synthetic_run_dir,
        "seg_tave_water",
        temp,
        "degrees Celsius",
        synthetic_params.parameters["nhm_seg"],
    )
    ds = xr.open_dataset(
        export_network_hydraulics(
            synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
        )
    )
    np.testing.assert_allclose(ds["water_temperature"], temp)
    assert ds["water_temperature"].attrs["units"] == "degC"
    ds.close()


@pytest.mark.domainless
def test_export_time_subset(synthetic_params, synthetic_run_dir, tmp_path):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    ds = xr.open_dataset(
        export_network_hydraulics(
            synthetic_params,
            synthetic_run_dir,
            tmp_path / "net.nc",
            start_time=np.datetime64("1979-01-02"),
            end_time=np.datetime64("1979-01-03"),
        )
    )
    assert ds.sizes["time"] == 2
    np.testing.assert_array_equal(ds["time"], TIMES[1:3])
    ds.close()


@pytest.mark.domainless
def test_export_missing_files_raise(
    synthetic_params, synthetic_run_dir, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    (synthetic_run_dir / "seg_flow_depth.nc").unlink()
    (synthetic_run_dir / "seg_res_time.nc").unlink()
    with pytest.raises(FileNotFoundError) as excinfo:
        export_network_hydraulics(
            synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
        )
    assert "seg_flow_depth" in str(excinfo.value)
    assert "seg_res_time" in str(excinfo.value)


@pytest.mark.domainless
def test_export_reach_order_mismatch_raises(
    synthetic_params, synthetic_run_dir, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    path = synthetic_run_dir / "seg_outflow.nc"
    with xr.open_dataarray(path) as opened:
        da = opened.load()  # close the file before rewriting it (Windows)
    da = da.assign_coords(nhm_seg=np.array([103, 102, 101]))
    da.to_netcdf(path)
    with pytest.raises(ValueError, match="nhm_seg"):
        export_network_hydraulics(
            synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
        )
