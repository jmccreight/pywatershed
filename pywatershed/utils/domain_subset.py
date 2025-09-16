import pathlib as pl
from typing import Literal, Union

import numpy as np
import pyPRMS as pp
import xarray as xr

import pywatershed as pws

from .segment_from_tracing import (
    get_from_segment_params,
    get_nhm_segs_ids_above_seg,
)

# TODO: subset restarts

"""Notes:
At the moment the assumption is that the full domain is provided PRMS input
files with only the option for NetCDF files.
"""

cbh_control_names = [
    "albedo_day",
    "cloud_cover_day",
    "humidity_day",
    "potet_day",
    "precip_day",
    "swrad_day",
    "tmax_day",
    "tmin_day",
    "transp_day",
    "windspeed_day",
    "AET_cbh_file",
    "PET_cbh_file",
    "rhavg",
]
# it's vague to me what these are in the headers of the files for
# https://github.com/DOI-USGS/pyPRMS/blob/49cbb8cd46b6760b1be67c106b8074688abaab39/tests/func/test_Control/ctl_metadata_default.csv#L96
cbh_var_names = [
    "albedo_hru",
    "cloud_cover_cbh",
    "humidity_hru",
    "potet",
    "prcp",
    "swrad",  # ?
    "tmax",
    "tmin",
    "transp_on",  # ?
    "windspeed_hru",
    "actet",
    "potet",
    "rhavg",
]
cbh_ctl_var_map = dict(zip(cbh_control_names, cbh_var_names))


class DomainSubset:
    """Subset a domain.

    Args:
        full_control_file: A pathlib.Path to the full domain control file.
        sub_nhm_ids: Either an np.ndarray of the nhm_ids or None if subsetting
          above a single nhm_seg.
        sub_nhm_segs: Either an np.ndarray of the nhm_segs or a np.int64 if
          subsetting above a single nhm_seg.
        full_cbh_nc_files_dict: Only used when NetCDF CBH files are being
          used, which should not be specified in a PRMS control file. If the
          CBH files in the control file are to be used, this argument is None.
          The argument is to be a dict of name:pathlib.Path where the names are
          in the list:
          ["albedo_day", "cloud_cover_day", "humidity_day", "potet_day",
           "precip_day", "swrad_day", "tmax_day", "tmin_day", "transp_day",
           "windspeed_day", "AET_cbh_file", "PET_cbh_file"]
         output_format: Either "pywatershed" (default) or "PRMS".
    """

    def __init__(
        self,
        full_control_file: pl.Path,
        sub_nhm_ids: Union[np.ndarray, None],
        sub_nhm_segs: Union[np.ndarray, np.int64],
        full_cbh_nc_files_dict: dict[str, pl.Path],
        start_one_below_nhm_seg: bool = False,
        output_format: Literal["pywatershed", "PRMS", None] = None,
        from_seg_calc_parallel: bool = False,
        from_seg_calc_check: bool = False,
    ) -> None:
        # Bring in arguments with vetting.
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

        # Get the full domain parameters
        self._full_control = pws.Control.load_prms(
            self._full_control_file, warn_unused_options=False
        )

        if full_cbh_nc_files_dict is None:
            raise NotImplementedError(
                "Currently only Netcdf CBH files accepted."
            )
            self._full_cbh_nc_files_dict = {}
            for kk in cbh_control_names:
                if kk in self._full_control.control_variables.keys():
                    self._full_cbh_nc_files_dict[kk] = (
                        self._full_control.control_variables[kk]
                    )
        else:
            self._full_cbh_nc_files_dict = full_cbh_nc_files_dict

        self._get_full_params()
        # Solve upstream segs and hrus if only a single segment is specified.
        if self._sub_nhm_ids is None:
            self._sub_domain_from_nhm_seg()
        # <
        self._set_subset_masks_order()
        self._subset_cbh()
        self._subset_params()
        # self._subset_restarts()

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
            self_attr_key="_full_params",
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

    def _get_from_segment_params(
        self, self_attr_key: str, parallel=True, check=True
    ) -> None:
        params_edit = getattr(self, self_attr_key)
        from_dict = get_from_segment_params(
            tosegment=params_edit.tosegment.values,
            parallel=parallel,
            check=check,
        )

        new_vars_kv = {
            "from_segment_starts": from_dict["from_segment_starts"],
            "from_segment_ends": from_dict["from_segment_ends"],
        }

        for kk, vv in new_vars_kv.items():
            params_edit[kk] = params_edit.tosegment.copy()
            params_edit[kk][:] = vv
            params_edit[kk].attrs = pws.meta.find_variables(kk)[kk]

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
        params_edit[kk] = from_seg_da
        params_edit[kk].attrs = pws.meta.find_variables(kk)[kk]

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

    def _set_subset_masks_order(self) -> None:
        self._sub_nhm_ids_mask = self._full_params.nhm_id.isin(
            self._sub_nhm_ids
        )

        self._sub_nhm_segs_mask = self._full_params.nhm_seg.isin(
            self._sub_nhm_segs
        )

        nhm_id_df = self._full_params["nhm_id"].to_pandas().reset_index()
        sub_nhm_id_df = nhm_id_df.iloc[self._sub_nhm_ids_mask.values]
        assert (np.diff(sub_nhm_id_df.nhru.values) > 0).all()
        self._sub_nhm_ids_order = sub_nhm_id_df.nhm_id.values

        nhm_seg_df = self._full_params["nhm_seg"].to_pandas().reset_index()
        sub_nhm_seg_df = nhm_seg_df.iloc[self._sub_nhm_segs_mask.values]
        assert (np.diff(sub_nhm_seg_df.nsegment.values) > 0).all()
        self._sub_nhm_segs_order = sub_nhm_seg_df.nhm_seg.values

        return None

    def _subset_cbh(self) -> None:
        if self._full_cbh_nc_files_dict is not None:
            self._subset_cbh_netcdf_files()
        else:
            self._subset_cbh_files()

    def _subset_cbh_files(self) -> None:
        raise NotImplementedError

    def _subset_cbh_netcdf_files(self):
        for kk, vv in self._full_cbh_nc_files_dict.items():
            if kk not in cbh_control_names:
                raise ValueError(
                    f"Supplied key, '{kk}', for CBH file invalid. It must be "
                    f"in: {cbh_control_names}"
                )
            if not vv.exists():
                raise ValueError("Supplied CBH file path, '{vv}', not found.")

        # bit of a silly pipeline, dont see another way to rename dim
        # independently of the coordinate with a dataarray
        self._sub_cbh_files_dict = {
            kk: xr.open_dataset(vv).rename_dims(
                nhm_id="nhru"
            )  # .to_dataarray()
            for kk, vv in self._full_cbh_nc_files_dict.items()
        }
        for kk in self._sub_cbh_files_dict.keys():
            self._sub_cbh_files_dict[kk] = self._sub_cbh_files_dict[kk].where(
                self._sub_nhm_ids_mask, drop=True
            )
            assert (
                self._sub_cbh_files_dict[kk].nhm_id.values
                == self._sub_nhm_ids_order
            ).all()

        return None

    def _subset_params(self) -> None:
        indexed_vars = [
            "from_segment",
            "from_segment_starts",
            "from_segment_ends",
            "tosegment",
            "from_nhm_seg_unstruct",
        ]

        # where(drop=True) causes dimensions to be expanded, so we have to
        # drop on hrus and segments separately, we separate all these dims that
        # are not time. the we reassemble
        params_sub = self._full_params.copy()
        params_sub = params_sub.drop_vars(indexed_vars)
        all_dims_drop = set(
            [
                "nhru",
                "nsegment",
                "scalar",
                "npoigages",
                "ndeplval",
            ]
        )
        grouped_param_subs = {}
        for gg in all_dims_drop:
            grouped_param_subs[gg] = params_sub.drop_dims(
                all_dims_drop - set([gg])
            )

        grouped_param_subs["nhru"] = grouped_param_subs["nhru"].where(
            self._sub_nhm_ids_mask, drop=True
        )
        grouped_param_subs["nsegment"] = grouped_param_subs["nsegment"].where(
            self._sub_nhm_segs_mask, drop=True
        )

        self._sub_params = xr.merge(grouped_param_subs.values())
        assert len(self._sub_params) == len(params_sub)
        # preserve the types (seems like this should be unnecessary)
        for vv in params_sub.variables:
            self._sub_params[vv] = self._sub_params[vv].astype(
                params_sub[vv].dtype
            )
            if params_sub[vv].dtype != self._sub_params[vv].dtype:
                print(
                    f"{vv}: {params_sub[vv].dtype=} "
                    f"{self._sub_params[vv].dtype=}"
                )
        del grouped_param_subs, params_sub

        assert (
            self._sub_params.nhm_id.values == self._sub_nhm_ids_order
        ).all()
        assert (
            self._sub_params.nhm_seg.values == self._sub_nhm_segs_order
        ).all()

        # re-solve tosegment and hru_segment
        tosegment_sub = []
        for ii in self._sub_params.nsegment.values:
            result = np.where(
                self._sub_params.nhm_seg.values
                == self._sub_params.tosegment_nhm.values[ii]
            )[0]
            len_result = len(result)
            if len_result == 1:
                tosegment_sub += result.tolist()
            elif len_result == 0:
                tosegment_sub += [-1]
            else:
                raise ValueError

        self._sub_params["tosegment"] = self._sub_params["nhm_seg"] * 0
        self._sub_params["tosegment"].values = np.array(tosegment_sub) + 1

        hru_segment_sub = []
        for ii in self._sub_params.nhru.values:
            result = np.where(
                self._sub_params.nhm_seg.values
                == self._sub_params.hru_segment_nhm.values[ii]
            )[0]
            len_result = len(result)
            if len_result == 1:
                hru_segment_sub += result.tolist()
            elif len_result == 0:
                hru_segment_sub += [-1]
            else:
                raise ValueError

        self._sub_params["hru_segment"] = self._sub_params["hru_segment"] * 0
        self._sub_params["hru_segment"].values = np.array(hru_segment_sub) + 1

        # Need upstream tracing parameters
        self._get_from_segment_params(
            self_attr_key="_sub_params",
            parallel=self._from_seg_calc_parallel,
            check=self._from_seg_calc_check,
        )

        return None

    def write(
        self,
        write_dir: pl.Path,
        output_format: Literal["pywatershed", "PRMS", None],
    ) -> None:
        """Write the subset domain to file.

        Args:
            write_dir: An NON-EXISTENT directory into which to write domain
                files. The reason is so that existing domain files are not
                overwritten.
            output_format: Optional choice of format. If not supplied, this
                argument supplied at __init__ is consulted. An error is
                raised if None is found.
        """
        write_dir = write_dir.resolve()
        if write_dir.exists():
            raise ValueError(
                f"The write_dir can not exist beforehand: {write_dir}"
            )
        if not write_dir.parent.exists():
            raise ValueError(
                "The parent of write_dir must exist beforehand: "
                f"{write_dir.parent}"
            )
        write_dir.mkdir()

        if output_format is None:
            output_format = self._output_format

        if output_format is None:
            raise ValueError(
                "output_format not specified on initialization or write."
            )

        pyprms_meta = pp.MetaData(verbose=False).metadata
        self._sub_control = pp.ControlFile(
            self._full_control_file, metadata=pyprms_meta, verbose=False
        )

        if output_format.lower() == "pywatershed":
            self._write_pws(write_dir=write_dir)
        elif output_format.lower() == "prms":
            # ensure
            for kk, vv in self._sub_cbh_files_dict.items():
                # what if the key is not in the control file? let it error
                sub_control.control_variables[kk].values = vv.name

            self._write_prms(write_dir=write_dir)
        pass

    def _write_prms(self, write_dir: pl.Path()) -> None:
        # write ascii cbh files
        # write ascii parameter file
        # write control file
        pass

    def _write_pws(self, write_dir: pl.Path) -> None:
        # Edit the control
        # dont really need to do anything with the cbh or param files for
        # pywatershed, but since the control file really shouldnt be used
        # by PRMS, we'll edit it to point to the netcdf files.
        for kk in self._sub_cbh_files_dict.keys():
            self._sub_control.control_variables[kk].values = f"{kk}.nc"

        self._sub_control.control_variables[
            "param_file"
        ].values = "parameters.nc"

        # write netcdf cbh files
        for kk, vv in self._sub_cbh_files_dict.items():
            # check the order before output
            assert (vv.nhm_id.values == self._sub_nhm_ids_order).all()
            # The naming of cbh files, their control parameters, and their
            # internal variables is INSANE and a mess in PRMS.
            var_name = cbh_ctl_var_map[kk]
            file_name = f"{var_name}.nc"
            vv.to_netcdf(write_dir / file_name)
        # <
        # write netcdf parameter file
        # check order before output
        assert (
            self._sub_params.nhm_id.values == self._sub_nhm_ids_order
        ).all()
        assert (
            self._sub_params.nhm_seg.values == self._sub_nhm_segs_order
        ).all()
        param_file_name = self._sub_control.control_variables[
            "param_file"
        ].values
        self._sub_params.to_netcdf(write_dir / param_file_name)

        # write control file
        ctl_file_name = f"{self._full_control_file.stem}_subset.control"
        self._sub_control.write(write_dir / ctl_file_name)
        return None
