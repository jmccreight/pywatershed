import numpy as np

from ..constants import HruType
from ..parameters import Parameters


def preprocess_gridded_params(parameters: Parameters):
    """Preprocess gridded parameters for active HRU masks, indices, and counts.

    Args:
        parameters (Parameters): The parameters to preprocess.

    Returns:
        Parameters: The preprocessed Parameters object
    """
    new_params = parameters.to_xr_ds()

    active_results = get_active_hru_params(new_params["hru_type"])
    new_params["active_hru_mask"] = active_results["active_hru_mask"]
    new_params["wh_active_hrus"] = active_results["wh_active_hrus"]
    new_params["nactive_hrus"] = active_results["nactive_hrus"]

    ret_params = Parameters.from_ds(new_params)

    return ret_params


def get_active_hru_params(hru_type):
    """Get active HRU parameters.

    Args:
        hru_type (np.ndarray): The HRU type array.

    Returns:
        dict: A dictionary containing the active HRU mask, indices, and count.
    """
    active_hru_mask = hru_type != HruType.INACTIVE.value
    wh_active_hrus = np.where(active_hru_mask)
    nactive_hrus = len(wh_active_hrus[0])

    return {
        "active_hru_mask": active_hru_mask,
        "wh_active_hrus": wh_active_hrus,
        "nactive_hrus": nactive_hrus,
    }
