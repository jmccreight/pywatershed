import numpy as np
import pytest

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
