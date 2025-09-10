import pathlib as pl
import warnings
from typing import Literal, Union

import numpy as np

import pywatershed as pws

# outline
# - option to write PRMS files or pywatershed files
# - get full domain parameters, cbh/forcings, control
# - identify subdomain nhm_id and nhm_seg:
#   - from a single nhm_seg of interest: get upstream (option to add one
#     downstream) segs, solve nhm_ids
#   - supplied ids
# - subset cbh
# - subset parameters (allow multiple parameter files)
# - sub control
# - sub restart files
# - write desired formats


class DomainSubset:
    def __init__(
        self,
        output_format: Literal["pywatershed", "PRMS"],
        full_cbh_dir: pl.Path,
        full_control_file: pl.Path,
        sub_ids: Union[np.ndarray, None],
        sub_segs: Union[np.ndarray, np.int64],
    ) -> None:
        self._output_format = output_format
        self._full_cbh_dir = full_cbh_dir
        self._full_control_file = full_control_file
        self._sub_nhm_ids = sub_ids
        self._sub_nhm_segs = sub_segs

        self._full_control = pws.Control.load_prms(
            self._full_control_file, warn_unused_options=False
        )

        self._solve_subset_ids_segs()

        return None

    def _solve_subset_ids_segs(self):
        if isinstance(self._sub_nhm_ids, np.ndarray) and isinstance(
            self._sub_nhm_segs, np.ndarray
        ):
            self._sub_domain_ids()

        elif self._sub_nhm_ids is None:
            if isinstance(self._sub_nhm_segs, np.ndarray):
                assert len(self._sub_nhm_segs) == 1
                self._sub_nhm_segs = self._sub_nhm_segs[0]
            # <
            self._sub_domain_from_nhm_seg()

        else:
            raise ValueError(
                "Invalid input types for sub_nhm_ids and sub_nhm_segs"
            )

        return None

    def _sub_inds_segs_set(self):
        if self._sub_nhm_ids is not None and self._sub_nhm_segs is not None:
            msg = (
                "self._sub_nhm_ids and self._sub_nhm_segs have already been "
                "set. Set both to None to recalculate."
            )
            warnings.warn(msg)
            return True
        else:
            return False

    def _sub_domain_ids(self):
        if self._sub_inds_segs_set():
            return None

        # <
        self._sub_nhm_ids = sub_nhm_ids
        self._sub_nhm_segs = sub_nhm_segs
        return None

    def _sub_domain_from_nhm_seg(self):
        if self._sub_inds_segs_set():
            return None

        raise NotImplementedError("Not implemented")
        # <
        return None

    @property
    def sub_nhm_ids(self):
        return self._sub_nhm_ids

    @property
    def sub_nhm_segs(self):
        return self._sub_nhm_segs
