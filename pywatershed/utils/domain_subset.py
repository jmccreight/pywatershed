import pathlib as pl
from typing import Literal, Union

import numpy as np
import xarray as xr

import pywatershed as pws

from .segment_from_tracing import (
    get_from_segment_params,
    get_nhm_segs_ids_above_seg,
)

# TODO: externalize functions for creating from_segment parameters and tracing
#       with them.

# outline
# - option to write PRMS files or pywatershed files
# - get full domain parameters
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
    """Subset a domain.

    Args:
        full_cbh_files_list: A list of pathlib.Path objects describing the
          paths of the NetCDF CBH files to subset.
        full_control_file: A pathlib.Path to the full domain control file.
        sub_nhm_ids: Either an np.ndarray of the nhm_ids or None if subsetting
          above a single nhm_seg.
        sub_nhm_segs: Either an np.ndarray of the nhm_segs or a np.int64 if
          subsetting above a single nhm_seg.
        output_format: Either "pywatershed" (default) or "PRMS".
    """

    def __init__(
        self,
        full_cbh_files_list: list,
        full_control_file: pl.Path,
        sub_nhm_ids: Union[np.ndarray, None],
        sub_nhm_segs: Union[np.ndarray, np.int64],
        start_one_below_nhm_seg: bool = True,
        output_format: Literal["pywatershed", "PRMS"] = "pywatershed",
        from_seg_calc_parallel: bool = False,
        from_seg_calc_check: bool = False,
    ) -> None:
        self._full_cbh_files_list = full_cbh_files_list
        self._full_control_file = full_control_file

        if not isinstance(sub_nhm_ids, (np.ndarray, type(None))):
            msg = f"The type of sub_nhm_ids, '{type(sub_nhm_ids)}' is invald."
            raise ValueError(msg)
        # <
        if not isinstance(sub_nhm_segs, (np.ndarray, np.int64)):
            msg = f"The type of sub_nhm_segs, '{type(sub_nhm_segs)}' is invald"
            raise ValueError(msg)

        self._sub_nhm_ids = sub_nhm_ids
        if sub_nhm_ids is None:
            self._start_seg = sub_nhm_segs
            self._sub_nhm_segs = None
        else:
            self._start_seg = None
            self._sub_nhm_segs = sub_nhm_segs

        self._start_one_below_nhm_seg = start_one_below_nhm_seg
        self._output_format = output_format
        self._from_seg_calc_parallel = from_seg_calc_parallel
        self._from_seg_calc_check = from_seg_calc_check

        self._full_control = pws.Control.load_prms(
            self._full_control_file, warn_unused_options=False
        )

        self._get_full_params()
        if self._sub_nhm_ids is None:
            self._sub_domain_from_nhm_seg()
        # <
        self._set_subset_masks()
        self._subset_cbh()

        return None

    @property
    def sub_nhm_ids(self) -> np.ndarray:
        return self._sub_nhm_ids

    @property
    def sub_nhm_segs(self) -> np.ndarray:
        return self._sub_nhm_segs

    def _sub_domain_from_nhm_seg(self) -> None:
        # This is the case where a single user-selected nhm_seg
        # is used to identify subset nhm_ids and nhm_segs

        # The user can pass a len 1 array or a scalar integer for the "outlet."
        if isinstance(self._start_seg, np.ndarray):
            assert len(self._start_seg) == 1
            self._start_seg = self._start_seg[0]

        # <
        # In most cases, when the passed seg id is a segment of interest, it is
        # likely better to take an additional segment below it before
        # subsetting
        if self._start_one_below_nhm_seg:
            wh_in_seg = np.where(self._full_params.nhm_seg == self._start_seg)[
                0
            ]
            if len(wh_in_seg):
                assert len(wh_in_seg) == 1
                self._start_seg = self._full_params.tosegment_nhm[
                    wh_in_seg[0]
                ].values
            else:
                raise ValueError(
                    "There appears to be no downstream segment of "
                    f"nhm_seg={self._start_seg}. You can disable the "
                    "start_one_below_nhm_seg option."
                )

        # Need upstream tracing parameters
        self._get_from_segment_params(
            parallel=self._from_seg_calc_parallel,
            check=self._from_seg_calc_check,
        )

        # Trace upstream
        nhm_ids_segs_above = get_nhm_segs_ids_above_seg(
            start_seg=self._start_seg,
            nhm_segs=self._full_params.nhm_seg.values,
            nhm_ids=self._full_params.nhm_id.values,
            hru_segment_nhm=self._full_params.hru_segment_nhm.values,
            from_segment_starts=self._full_params.from_segment_starts.values,
            from_segment_ends=self._full_params.from_segment_ends.values,
            from_segment=self._full_params.from_segment.values,
        )

        self._sub_nhm_ids = nhm_ids_segs_above["nhm_ids_above"]
        self._sub_nhm_segs = nhm_ids_segs_above["nhm_segs_above"]

        return None

    def _get_from_segment_params(self, parallel=True, check=True) -> None:
        from_dict = get_from_segment_params(
            tosegment=self._full_params.tosegment.values,
            parallel=parallel,
            check=check,
        )

        new_vars_kv = {
            "from_segment_starts": from_dict["from_segment_starts"],
            "from_segment_ends": from_dict["from_segment_ends"],
        }

        for kk, vv in new_vars_kv.items():
            self._full_params[kk] = self._full_params.tosegment.copy()
            self._full_params[kk][:] = vv
            self._full_params[kk].attrs = pws.meta.find_variables(kk)[kk]

        # there is no useful coordinate because "from" (down stream to
        # upstream) is not unique
        froms = from_dict["from_segment"]
        from_seg_da = xr.DataArray(
            froms,
            coords={"from_nhm_seg_unstruct": np.arange(len(froms))},
            dims=["from_nhm_seg_unstruct"],
            name="from_segment",
        )
        kk = "from_segment"
        self._full_params[kk] = from_seg_da
        self._full_params[kk].attrs = pws.meta.find_variables(kk)[kk]

        return None

    def _get_full_params(self) -> None:
        full_param_files = (
            self._full_control_file.parent
            / self._full_control.options["parameter_file"]
        )
        self._full_params = pws.parameters.PrmsParameters.load(
            full_param_files
        ).to_xr_ds()
        return None

    def _set_subset_masks(self) -> None:
        # need to check that the order of the ids is preserved WRT to the
        # full domain.
        # OR DO WE DO THIS AFTER SUBSETTING?

        self._sub_nhm_id_mask = self._full_params.nhm_id.isin(
            self._sub_nhm_ids
        )

        # # need to re-order the data in some cases, use pandas DF.merge
        # df_full_ids = pd.DataFrame(
        #     {"nhm_id": self._full_params.nhm_id}
        # ).set_index("nhm_id")

        # df_sub_ids = pd.DataFrame({"nhm_id": self._sub_nhm_ids}).set_index(
        #     "nhm_id"
        # )
        # df_sub_ids_ordered = df_full_ids.merge(
        #     df_sub_ids, left_index=True, right_index=True
        # )
        # sadf
        # # print(df_target)
        # wu_sub_data = df_target.to_xarray()
        # assert (wu_sub_data.nhm_id.values == nhm_ids).all()

        return None

    def _subset_cbh(self) -> None:
        for ff in self._full_cbh_files_list:
            assert ff.exists()

        self._sub_cbh_files_dict = {
            ff.name: xr.open_dataarray(ff) for ff in self._full_cbh_files_list
        }

        asdf
        for kk, da in self._sub_cbh_files_dict.items():
            sub_da = da.where(self._sub_nhm_ids, drop=True)
            # self._sub_cbh_files_dict

        asdf
        pass
