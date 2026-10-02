import numpy as np
import pytest

from pywatershed.constants import HruType
from pywatershed.parameters import Parameters
from pywatershed.utils.preprocess_gridded import (
    get_active_hru_params,
    preprocess_gridded_params,
)

# These tests are domainless: they construct minimal Parameters objects
# rather than requiring generated domain test data.

INACTIVE = HruType.INACTIVE.value

hru_type_cases = {
    "all_active": np.array([1, 1, 1, 1, 1], dtype="int32"),
    "some_inactive": np.array(
        [1, 1, INACTIVE, 1, INACTIVE],
        dtype="int32",
    ),
    "all_hru_types": np.array([1, 2, INACTIVE, 3, 4], dtype="int32"),
}


def make_parameters(hru_type: np.ndarray) -> Parameters:
    """A minimal Parameters object with hru_type and one other parameter."""
    nhru = len(hru_type)
    return Parameters(
        dims={"nhru": nhru},
        coords={"nhru": np.arange(nhru)},
        data_vars={
            "hru_type": hru_type,
            "hru_area": np.arange(nhru, dtype="float64") + 1.0,
        },
        metadata={
            "nhru": {"dims": ["nhru"]},
            "hru_type": {"dims": ["nhru"]},
            "hru_area": {"dims": ["nhru"]},
        },
        validate=True,
    )


@pytest.fixture(params=list(hru_type_cases.keys()))
def hru_type(request):
    return hru_type_cases[request.param].copy()


@pytest.mark.domainless
def test_get_active_hru_params(hru_type):
    result = get_active_hru_params(hru_type)
    wh_active_hrus = result["wh_active_hrus"]

    # the canonical shape is a 1-D integer index array, not a tuple
    assert not isinstance(wh_active_hrus, tuple)
    assert isinstance(wh_active_hrus, np.ndarray)
    assert wh_active_hrus.ndim == 1
    assert np.issubdtype(wh_active_hrus.dtype, np.integer)

    assert (result["active_hru_mask"] == (hru_type != INACTIVE)).all()
    assert (wh_active_hrus == np.where(hru_type != INACTIVE)[0]).all()
    assert result["nactive_hrus"] == len(wh_active_hrus)
    assert isinstance(result["nactive_hrus"], int)


@pytest.mark.domainless
def test_get_active_hru_params_all_inactive_raises():
    """An hru_type with no active HRU is an input error."""
    with pytest.raises(ValueError, match="no HRU active"):
        get_active_hru_params(np.array([INACTIVE] * 4, dtype="int32"))


@pytest.mark.domainless
def test_preprocess_gridded_params(hru_type):
    """The three quantities land in the Parameters, values and structure.

    Answers are computed here from hru_type so they follow hru_type_cases.
    Structure: the index array is a data variable rather than an index
    coordinate, on its own nactive_hru dimension.
    """
    params = make_parameters(hru_type)
    result = preprocess_gridded_params(params)
    assert isinstance(result, Parameters)

    active = hru_type != INACTIVE
    assert (result.parameters["active_hru_mask"] == active).all()
    assert (result.parameters["wh_active_hrus"] == np.where(active)[0]).all()
    assert result.parameters["nactive_hrus"] == active.sum()

    assert "wh_active_hrus" in result.data_vars.keys()
    assert "wh_active_hrus" not in result.coords.keys()
    assert result.metadata["wh_active_hrus"]["dims"] == ("nactive_hru",)
    assert result.dims["nactive_hru"] == active.sum()

    assert "active_hru_mask" in result.data_vars.keys()
    assert result.metadata["active_hru_mask"]["dims"] == ("nhru",)
    assert "nactive_hrus" in result.data_vars.keys()
    assert result.metadata["nactive_hrus"]["dims"] == ("scalar",)


@pytest.mark.domainless
def test_preprocess_gridded_params_inputs_unchanged(hru_type):
    params = make_parameters(hru_type)
    result = preprocess_gridded_params(params)

    for kk, vv in params.parameters.items():
        assert (result.parameters[kk] == vv).all()

    for kk, vv in params.dims.items():
        assert result.dims[kk] == vv


@pytest.mark.domainless
def test_preprocess_gridded_params_all_inactive_raises():
    """The preprocessor refuses an hru_type with no active HRU."""
    hru_type = np.array([INACTIVE] * 4, dtype="int32")
    with pytest.raises(ValueError, match="no HRU active"):
        preprocess_gridded_params(make_parameters(hru_type))
