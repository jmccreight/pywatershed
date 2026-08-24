import numpy as np

from pywatershed.base.timeseries import TimeseriesArray

from ..base import meta
from ..constants import fill_values_dict
from ..utils.preprocess_gridded import (
    active_hru_params_from_mask,
    get_active_hru_params,
)


class HruMixin:
    """Mixin for HRU functionalities."""

    def _set_active_hrus(self):
        """Set _active_hru_mask, _wh_active_hrus, and _nactive_hrus.

        These are set on self as derived parameters and not tracked in the
        Parameter object. The mask is taken from the "active_hru_mask"
        parameter when it is supplied (see
        :func:`~pywatershed.utils.preprocess_gridded.preprocess_gridded_params`)
        and is derived from "hru_type" otherwise. The indices and the count
        of active HRUs are always derived from the mask in use.

        Returns:
            None
        """
        parameters = self._params.parameters
        if "active_hru_mask" in parameters.keys():
            # a mask round-tripped through file may come back as int8
            result = active_hru_params_from_mask(
                np.asarray(parameters["active_hru_mask"]).astype("bool")
            )
        else:
            result = get_active_hru_params(parameters["hru_type"])

        for kk, vv in result.items():
            self[f"_{kk}"] = vv

        return

    def _mask_inactive_hrus(self):
        """Set all variables to missing values outside of _active_hru_mask."""
        if self._active_hru_mask.all():
            return

        # TODO: use constants.fill_values_dict here for different types.
        for var_name in self.get_variables():
            var_dims = list(meta.get_dimensions(var_name).values())[0]
            if "nhru" not in var_dims:
                continue

            var = self[var_name]
            if isinstance(var, TimeseriesArray):
                # data are (ntimes, nhru)
                var.data[:, ~self._active_hru_mask] = fill_values_dict[
                    var.data.dtype
                ]
            else:
                axis = var_dims.index("nhru")
                index = [slice(None)] * var.ndim
                index[axis] = ~self._active_hru_mask
                var[tuple(index)] = fill_values_dict[var.dtype]

        return
