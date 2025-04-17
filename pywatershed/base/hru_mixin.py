from pywatershed.base.timeseries import TimeseriesArray

from ..base import meta
from ..constants import fill_values_dict
from ..utils.preprocess_gridded import get_active_hru_params


class HruMixin:
    """Mixin for HRU functionalities."""

    def _set_active_hrus(self):
        """Set _active_hru_mask _wh_active_hrus, and _nactive_hrus.

        These are set on self as derived parameters and not tracked in the
        Parameter object. However, the variables may be supplied in the
        Parameter object and will be used if present..

        Returns:
            None
        """
        param_keys = self._params.parameters.keys()
        var_keys = ["active_hru_mask", "wh_active_hrus", "nactive_hrus"]
        for kk in var_keys:
            if kk in param_keys:
                self[f"_{kk}"] = self._params.parameters[kk]

        have_all_vars = [hasattr(self, kk) for kk in var_keys]
        if not all(have_all_vars):
            result = get_active_hru_params(self._params.parameters["hru_type"])
            for kk in var_keys:
                self[f"_{kk}"] = result[kk]

        if isinstance(self._wh_active_hrus, tuple):
            self._wh_active_hrus = self._wh_active_hrus[0]

    def _mask_inactive_hrus(self):
        """Set all variables to missing values outside of _active_hru_mask."""
        if self._nactive_hrus == 0:
            return

        # TODO: use constants.fill_values_dict here for different types.
        for var_name in self.get_variables():
            var_dim_name = list(meta.get_dimensions(var_name).values())[0][0]
            if "nhru" not in var_dim_name:
                continue

            var = self[var_name]
            if isinstance(var, TimeseriesArray):
                var.data[:, ~self._active_hru_mask] = fill_values_dict[
                    var.data.dtype
                ]
            else:
                var[~self._active_hru_mask] = fill_values_dict[var.dtype]

        return
