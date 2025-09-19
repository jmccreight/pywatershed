import pathlib as pl
import shutil
from typing import Literal, Union

import numpy as np
import pyPRMS as pp
import xarray as xr

import pywatershed as pws

from .segment_from_tracing import (
    get_from_segment_params,
    get_nhm_segs_ids_above_seg,
)

pyprms_meta = pp.MetaData(verbose=False).metadata

# TODO: revisit subset parameters now that isel is being used, may be much
#       simpler
# TODO: subset sf_data
# TODO: subset restarts
# TODO: maybe something todo, there are no real checks on time or subsetting
#       in time.

additional_cbh_meta = {
    "PET_cbh_file": {
        "datatype": "string",
        "description": (
            "Pathname of the CBH file of pre-processed potential "
            "evapotransipration input data for each HRU to specify "
            "variable potet."
        ),
        "context": "scalar",
        # "default": np.str_(""),
    },
    "AET_cbh_file": {
        "datatype": "string",
        "description": (
            "Pathname of the CBH file of pre-processed actual "
            "evapotransipration input data for each HRU to specify "
            "variable actet."
        ),
        "context": "scalar",
        # "default": np.str_("tmin.day"),
    },
}

"""Notes:
At the moment the assumption is that the full domain is provided PRMS input
files with only the option for NetCDF files.
"""

cbh_ctl_var_map = pws.constants.cbh_ctl_var_map
cbh_control_names = list(cbh_ctl_var_map.keys())
cbh_var_names = list(cbh_ctl_var_map.values())


class DomainSubset:
    """Subset a domain.

    Args:
        full_control_file: A pathlib.Path to the full domain control file.
          Currently, the "param_file" field must a be a scalar (multiple
          parameter files specified will throw an error).
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
        full_cbh_nc_files_dict: Union[dict[str, pl.Path], None],
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
        param_file = self._full_control.options["parameter_file"]
        if not isinstance(param_file, str) and len(param_file) > 1:
            raise NotImplementedError(
                "DomainSubset only works with single parameter files at the "
                "moment. Please consolidate your parameter files or extend "
                "pyPRMS to handle multiple parameter files."
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
        self, self_attr_key: str, parallel=False, check=True
    ) -> None:
        params_edit = getattr(self, self_attr_key)
        print("starting get_from_segment_params")
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
            kk: xr.load_dataset(vv).rename_dims(
                nhm_id="nhru"
            )  # .to_dataarray()
            for kk, vv in self._full_cbh_nc_files_dict.items()
        }
        for kk in self._sub_cbh_files_dict.keys():
            self._sub_cbh_files_dict[kk] = self._sub_cbh_files_dict[kk].isel(
                nhru=self._sub_nhm_ids_mask
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
        indexed_vars_in_ds = set(indexed_vars).intersection(
            set(params_sub.variables)
        )
        params_sub = params_sub.drop_vars(indexed_vars_in_ds)
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

        grouped_param_subs["nhru"] = grouped_param_subs["nhru"].isel(
            nhru=self._sub_nhm_ids_mask
        )
        grouped_param_subs["nsegment"] = grouped_param_subs["nsegment"].isel(
            nsegment=self._sub_nhm_segs_mask
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

        self._sub_control = pws.utils.utils.pyprms_control_no_defaults(
            self._full_control_file, metadata=pyprms_meta, verbose=False
        )
        self._sub_control_file_name = (
            f"{self._full_control_file.stem}_subset.control"
        )

        if output_format.lower() == "pywatershed":
            self._write_pws(write_dir=write_dir)
        elif output_format.lower() == "prms":
            self._write_prms(write_dir=write_dir)

        return None

    def _write_prms(self, write_dir: pl.Path) -> None:
        # these functions set the control, reusing the filename in it
        # but making it relative to the write_dir.
        self._cbh_dataset_to_ascii(write_dir=write_dir)
        self._parameters_to_ascii(write_dir=write_dir)

        # todo: subset the data file
        data_file = pl.Path(self._sub_control.get("data_file").values)
        _ = shutil.copy(data_file, write_dir / data_file.name)

        self._sub_control.write(write_dir / self._sub_control_file_name)

        return None

    def _write_pws(self, write_dir: pl.Path) -> None:
        # Edit the control
        # dont really need to do anything with the cbh or param files for
        # pywatershed, but since the control file really shouldnt be used
        # by PRMS, we'll edit it to point to the netcdf files.

        for kk in self._sub_cbh_files_dict.keys():
            if kk in self._sub_control.control_variables.keys():
                self._sub_control.control_variables[kk].values = f"{kk}.nc"
            else:
                vv = pp.ControlVariable(
                    name=kk, strict=False, meta=additional_cbh_meta[kk]
                )
                vv.__dict__["_ControlVariable__values"] = f"{kk}.nc"
                self._sub_control.control_variables[kk] = vv

        self._sub_control.control_variables[
            "param_file"
        ].values = "parameters.nc"

        # write netcdf cbh files
        for kk, vv in self._sub_cbh_files_dict.items():
            print(f"{kk=}")
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
        self._sub_control.write(write_dir / self._sub_control_file_name)
        return None

    def _cbh_dataset_to_ascii(self, write_dir):
        test_data_dir = pl.Path(
            pws.constants.__pywatershed_root__ / "../test_data"
        )
        # simply load a dummy CBH netcdf file from the pywatershed repo
        # and replace its dataset with ours.
        pp_cbh = pp.Cbh(
            src_path=test_data_dir / "drb_2yr/cbh.nc",
            metadata=pyprms_meta,
            engine="netcdf",
        )
        cbh_ds = xr.merge(self._sub_cbh_files_dict.values())
        pp_cbh._Cbh__dataset = cbh_ds
        for kk, vv in self._sub_cbh_files_dict.items():
            var_name = list(vv.data_vars)[0]
            file_name = pl.Path(self._sub_control[kk].values).name
            self._sub_control[kk].values = file_name
            pp_cbh.write_ascii(
                filename=write_dir / file_name, variable=var_name
            )

    def _parameters_to_ascii(self, write_dir):
        # This is basically implemeting pyPRMS/parameters/ParameterNetCDF.py
        # but just skipping reading from netcdf file
        verbose = False

        # prep the parameters to be acceptable
        sub_params_ds = self._sub_params.rename(
            ndoy="ndays", scalar="one", nmonth="nmonths"
        )
        drop_vars = [
            "doy",
            "from_segment_starts",
            "from_segment_ends",
            "from_nhm_seg_unstruct",
            "from_segment",
            "hru_in_to_cf",
        ]
        sub_params_ds = sub_params_ds.drop(drop_vars)

        pp_params = pp.Parameters(metadata=pyprms_meta, verbose=verbose)
        pp_params.__verbose = verbose

        # leaving this here for the mask_and_scale and decode_timedelta
        # implications
        # xr_df = xr.open_dataset(
        #     self.__filename, mask_and_scale=False, decode_timedelta=False
        # )

        # Populate the dimensions first
        pp_params.dimensions.add(name="one", size=1)
        # self.dimensions.add(name='ndays')

        for dn, ds in dict(sub_params_ds.sizes).items():
            pp_params.dimensions.add(name=str(dn), size=ds)

        # Add ndepl using ndeplval
        pp_params.dimensions.add(
            name="ndepl", size=int(sub_params_ds.sizes["ndeplval"] / 11)
        )

        # Add nobs if needed
        if not pp_params.dimensions.exists("nobs"):
            if pp_params.dimensions.exists("npoigages"):
                pp_params.dimensions.add(
                    name="nobs",
                    size=pp_params.dimensions.get("npoigages").size,
                )
            else:
                pp_params.dimensions.add(name="nobs", size=0)

        if not pp_params.dimensions.exists("ngw"):
            pp_params.dimensions.add(
                name="ngw", size=pp_params.dimensions.get("nhru").size
            )

        if not pp_params.dimensions.exists("nssr"):
            pp_params.dimensions.add(
                name="nssr", size=pp_params.dimensions.get("nhru").size
            )

        # Now add the parameters
        for var in sub_params_ds.variables.keys():
            if pp_params.__verbose:
                print(str(var))

            cparam = sub_params_ds[var].T

            # Add the parameter
            pp_params.add(name=str(var))

            # Add the data
            pp_params.get(str(var)).data = cparam.values

        pp_params.adjust_bounded_parameters()

        file_name = pl.Path(self._sub_control["param_file"].values).name
        self._sub_control["param_file"].values = file_name
        pp_params.write_parameter_file(filename=write_dir / file_name)

        return None
