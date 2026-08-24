import numpy as np

from ..constants import HruType
from ..parameters import Parameters


def preprocess_gridded_params(parameters: Parameters) -> Parameters:
    """Add the active HRU mask to gridded parameters.

    The mask is derived from ``hru_type``, HRUs with
    ``hru_type == HruType.INACTIVE`` are inactive, and is stored as the
    ``active_hru_mask`` parameter. Processes using
    :class:`~pywatershed.base.HruMixin` prefer a supplied
    ``active_hru_mask`` over deriving one from ``hru_type``, so editing the
    mask on the returned object deactivates HRUs without editing
    ``hru_type``.

    Args:
        parameters (Parameters): The parameters to preprocess.

    Returns:
        Parameters: A new Parameters object with ``active_hru_mask`` added.
    """
    new_params = parameters.to_xr_ds()

    hru_type = new_params["hru_type"]
    active_results = get_active_hru_params(hru_type.values)
    new_params["active_hru_mask"] = (
        hru_type.dims,
        active_results["active_hru_mask"],
    )

    ret_params = Parameters.from_ds(new_params)

    return ret_params


def get_active_hru_params(hru_type: np.ndarray) -> dict:
    """Get active HRU parameters from hru_type.

    Args:
        hru_type (np.ndarray): The HRU type array.

    Returns:
        dict: A dictionary containing the active HRU mask, indices, and
            count, see
            :func:`~pywatershed.utils.preprocess_gridded.active_hru_params_from_mask`.
    """
    active_hru_mask = hru_type != HruType.INACTIVE.value

    return active_hru_params_from_mask(active_hru_mask)


def active_hru_params_from_mask(active_hru_mask: np.ndarray) -> dict:
    """Get active HRU parameters from an active HRU mask.

    Args:
        active_hru_mask (np.ndarray): Boolean array, True on active HRUs.

    Returns:
        dict: A dictionary containing the active HRU mask
            (``active_hru_mask``), the indices of the active HRUs
            (``wh_active_hrus``), and their count (``nactive_hrus``).
    """
    wh_active_hrus = np.where(active_hru_mask)[0]

    return {
        "active_hru_mask": active_hru_mask,
        "wh_active_hrus": wh_active_hrus,
        "nactive_hrus": len(wh_active_hrus),
    }
