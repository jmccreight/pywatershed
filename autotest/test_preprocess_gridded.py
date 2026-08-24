import numpy as np
import pytest

from pywatershed.base.accessor import Accessor
from pywatershed.base.hru_mixin import HruMixin
from pywatershed.constants import HruType
from pywatershed.parameters import Parameters
from pywatershed.utils.preprocess_gridded import (
    get_active_hru_params,
    preprocess_gridded_params,
)

# HRUs 1 and 4 are INACTIVE, the rest are LAND, LAKE, or SWALE
nhru = 6
hru_type = np.array([1, 0, 1, 2, 0, 3], dtype="int32")
hru_type_mask = np.array([True, False, True, True, False, True])


def make_parameters(active_hru_mask: np.ndarray = None) -> Parameters:
    """A minimal Parameters with hru_type and an optional active_hru_mask."""
    data_vars = {"hru_type": hru_type.copy()}
    if active_hru_mask is not None:
        data_vars["active_hru_mask"] = active_hru_mask

    metadata = {kk: {"dims": ("nhru",), "attrs": {}} for kk in data_vars}
    # nhm_id is the coordinate on the nhru dimension, as in the domain files
    metadata["nhm_id"] = {"dims": ("nhru",), "attrs": {}}

    return Parameters(
        dims={"nhru": nhru},
        coords={"nhm_id": np.arange(nhru)},
        data_vars=data_vars,
        metadata=metadata,
    )


class HruThing(Accessor, HruMixin):
    """The smallest consumer of HruMixin, for testing the mixin alone."""

    def __init__(self, parameters: Parameters):
        self._params = parameters
        self.hru_ppt = np.arange(nhru, dtype="float64")
        self._set_active_hrus()

    def get_variables(self) -> tuple:
        return ("hru_ppt",)


@pytest.mark.domainless
def test_get_active_hru_params():
    result = get_active_hru_params(hru_type)

    np.testing.assert_array_equal(result["active_hru_mask"], hru_type_mask)
    # the indices are a bare array, not the tuple np.where returns
    assert isinstance(result["wh_active_hrus"], np.ndarray)
    np.testing.assert_array_equal(result["wh_active_hrus"], [0, 2, 3, 5])
    assert result["nactive_hrus"] == 4


@pytest.mark.domainless
def test_preprocess_gridded_params():
    params = make_parameters()
    assert "active_hru_mask" not in params.parameters.keys()

    new_params = preprocess_gridded_params(params)

    np.testing.assert_array_equal(
        new_params.parameters["active_hru_mask"], hru_type_mask
    )
    np.testing.assert_array_equal(new_params.parameters["hru_type"], hru_type)
    # the input is not modified
    assert "active_hru_mask" not in params.parameters.keys()


@pytest.mark.domainless
def test_preprocess_gridded_params_netcdf_round_trip(tmp_path):
    new_params = preprocess_gridded_params(make_parameters())
    param_file = tmp_path / "params_gridded.nc"
    new_params.to_netcdf(param_file)

    from_file = Parameters.from_netcdf(param_file)
    mask = from_file.parameters["active_hru_mask"]

    # netcdf has no bool type: the mask must come back as bool, not int8,
    # or ~mask in _mask_inactive_hrus is a bitwise not
    assert mask.dtype == np.dtype("bool")
    np.testing.assert_array_equal(mask, hru_type_mask)


@pytest.mark.domainless
def test_set_active_hrus_from_hru_type():
    thing = HruThing(make_parameters())

    np.testing.assert_array_equal(thing._active_hru_mask, hru_type_mask)
    np.testing.assert_array_equal(thing._wh_active_hrus, [0, 2, 3, 5])
    assert thing._nactive_hrus == 4


@pytest.mark.domainless
@pytest.mark.parametrize("mask_dtype", ["bool", "int8"])
def test_set_active_hrus_uses_supplied_mask(mask_dtype):
    # a mask which disagrees with hru_type: HRU 3 is deactivated and
    # HRU 4 is activated
    supplied_mask = np.array([True, False, True, False, True, True])
    thing = HruThing(
        make_parameters(active_hru_mask=supplied_mask.astype(mask_dtype))
    )

    np.testing.assert_array_equal(thing._active_hru_mask, supplied_mask)
    np.testing.assert_array_equal(thing._wh_active_hrus, [0, 2, 4, 5])
    assert thing._nactive_hrus == 4
    # the indices and the count follow the mask, not hru_type
    assert not (thing._active_hru_mask == hru_type_mask).all()


@pytest.mark.domainless
def test_mask_inactive_hrus():
    thing = HruThing(make_parameters())
    thing._mask_inactive_hrus()

    assert np.isnan(thing.hru_ppt[~hru_type_mask]).all()
    np.testing.assert_array_equal(
        thing.hru_ppt[hru_type_mask], np.arange(nhru)[hru_type_mask]
    )


@pytest.mark.domainless
def test_mask_inactive_hrus_all_inactive():
    all_inactive = np.zeros(nhru, dtype="bool")
    thing = HruThing(make_parameters(active_hru_mask=all_inactive))
    assert thing._nactive_hrus == 0

    thing._mask_inactive_hrus()

    # nothing is active: everything is masked, the shortcut is for the
    # nothing-to-mask case
    assert np.isnan(thing.hru_ppt).all()


@pytest.mark.domainless
def test_mask_inactive_hrus_all_active():
    all_active = np.ones(nhru, dtype="bool")
    thing = HruThing(make_parameters(active_hru_mask=all_active))

    thing._mask_inactive_hrus()

    np.testing.assert_array_equal(thing.hru_ppt, np.arange(nhru))


@pytest.mark.domainless
def test_hru_type_inactive_value():
    # the mask convention above depends on this
    assert HruType.INACTIVE.value == 0
