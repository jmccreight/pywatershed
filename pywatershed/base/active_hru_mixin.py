import numpy as np

from pywatershed.base.timeseries import TimeseriesArray

from ..constants import mask_fill_values_dict
from ..utils.preprocess_gridded import get_active_hru_params


class ActiveHruMixin:
    """Derive the active-HRU mask from hru_type; mask inactive HRUs."""

    def _set_active_hrus(self) -> None:
        """Set _active_hru_mask, _wh_active_hrus, _nactive_hrus and, for a
        class that does not declare it as a parameter, hru_route_order.

        All three are derived from the hru_type parameter, via
        :func:`~pywatershed.utils.preprocess_gridded.get_active_hru_params`,
        every time this method is called. They are set on self as private,
        derived quantities; they are neither read from nor tracked in the
        Parameters object. hru_type is the single source of truth for which
        HRUs are active: an active_hru_mask present in the Parameters
        object (a discretization written by preprocess_gridded_params) is
        not read, only checked against the recomputed mask.

        Raises:
            KeyError: if hru_type is not among the host's parameters, i.e.
                the host class does not list it in get_parameters().
            ValueError: if a supplied active_hru_mask disagrees with
                hru_type, i.e. the discretization is stale; or if a
                supplied hru_route_order (written by
                :func:`~utils.preprocess_cascades.preprocess_cascade_params`)
                does not name exactly the active HRUs, i.e. the cascade
                parameters are stale.

        Returns:
            None
        """
        if "hru_type" not in self.parameters:
            raise KeyError(
                f"{self.__class__.__name__} uses ActiveHruMixin, so "
                "'hru_type' must be in its get_parameters()"
            )
        result = get_active_hru_params(self._params.parameters["hru_type"])
        supplied = self._params.parameters.get("active_hru_mask")
        if (
            supplied is not None
            and not (supplied == result["active_hru_mask"]).all()
        ):
            raise ValueError(
                "active_hru_mask in the discretization disagrees with "
                "hru_type; rerun preprocess_gridded_params"
            )
        route_order = self._params.parameters.get("hru_route_order")
        if route_order is not None:
            # 1-based, the active HRUs first, zeros after. The kernels loop
            # over the first _nactive_hrus entries of it without bounds
            # checks, so it must name exactly the active HRUs.
            routed = np.sort(route_order[route_order > 0]) - 1
            if not np.array_equal(routed, result["wh_active_hrus"]):
                raise ValueError(
                    "hru_route_order in the parameters disagrees with "
                    "hru_type; rerun preprocess_cascade_params"
                )
        for kk in ("active_hru_mask", "wh_active_hrus", "nactive_hrus"):
            self[f"_{kk}"] = result[kk]

        if "hru_route_order" not in self.parameters:
            # The kernels loop over hru_route_order (1-based, as in PRMS);
            # a class that does not declare it (no cascades) gets the
            # active HRUs in index order. The cascade classes declare it
            # and _set_params has already set it from the parameters.
            self.hru_route_order = result["wh_active_hrus"] + 1

        return

    def _mask_inactive_hrus(self) -> None:
        """Set nhru-dimensioned variables to missing outside _active_hru_mask.

        Variables without an nhru dimension (e.g. on nsegment) are left
        untouched.
        """
        if self._active_hru_mask.all():
            # nothing to mask
            return

        for var_name in self.get_variables():
            var_dims = self.meta[var_name]["dims"]
            if "nhru" not in var_dims:
                continue

            var = self[var_name]
            if isinstance(var, TimeseriesArray):
                # data are (ntimes, nhru)
                var.data[:, ~self._active_hru_mask] = mask_fill_values_dict[
                    var.data.dtype
                ]
            else:
                axis = var_dims.index("nhru")
                index = [slice(None)] * var.ndim
                index[axis] = ~self._active_hru_mask
                var[tuple(index)] = mask_fill_values_dict[var.dtype]

        return
