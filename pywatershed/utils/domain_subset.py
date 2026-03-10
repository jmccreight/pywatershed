import pathlib as pl
from copy import deepcopy
from typing import Literal, Union
from warnings import warn

import numpy as np
import pyPRMS as pp
import xarray as xr

import pywatershed as pws

from ..base import meta as pws_meta
from ..constants import pyprms_meta as _pyprms_meta
from .segment_from_tracing import (
    get_from_segment_params,
    get_nhm_segs_ids_above_seg,
)

cbh_var_ascii_format = {
    "tmax": "%0.2f",
    "tmin": "%0.2f",
    "prcp": "%0.2f",
    "rhavg": "%0.2f",
    "actet": "%0.4f",
    "potet": "%0.4f",
}

cbh_vars_rename = {"actet": "aet_observed", "potet": "pet_observed"}
cbh_vars_rename_inv = {vv: kk for kk, vv in cbh_vars_rename.items()}

# TODO: subset restarts
# TODO: maybe something todo, there are no real checks on time or subsetting
#       in time.

# Only some of these are in the GSFLOW 2.4.0 input instruction word doc.
additional_cbh_meta = {
    "AET_cbh_file": {
        "datatype": "string",
        "description": (
            "Pathname of the CBH file of pre-processed actual "
            "evapotransipration input data for each HRU to specify "
            "variable actet."
        ),
        "context": "scalar",
    },
    "AET_module": {
        "datatype": "string",
        "description": (
            "Module to read actual evapotranspiration CBH File; specify "
            "climate_hru to activate."
        ),
        "context": "scalar",
    },
    "ag_frac_dynamic": {
        "datatype": "string",
        "description": "Pathname to the dynamic ag fraction parameter file.",
        "context": "scalar",
    },
    "dyn_ag_frac_flag": {
        "datatype": "int32",
        "description": "Flag governing the use of ag_frac_dynamic fie data.",
        "context": "scalar",
    },
    "forcing_check_flag": {
        "datatype": "int32",
        "description": (
            "Flag to indicate performance of precipitation and temperature "
            "checks for hru_ppt < 0.0, hru_rain < 0.0, hru_snow < 0.0, "
            "tmax < tmin, tminf < -150.0 and tmaxf > 200.0 (0=no; 1=yes)."
        ),
        "context": "scalar",
    },
    "PET_ag_module": {
        "datatype": "string",
        "description": (
            "Module to read potential evapotranspiration CBH File; specify "
            "climate_hru to activate."
        ),
        "context": "scalar",
    },
    "PET_cbh_file": {
        "datatype": "string",
        "description": (
            "Pathname of the CBH file of pre-processed potential "
            "evapotransipration input data for each HRU to specify "
            "variable potet."
        ),
        "context": "scalar",
    },
    "iter_aet_flag": {
        "datatype": "int32",
        "description": (
            "Flag to indicate to estimate irrigation water based on the "
            "difference between actual evapotranspiration and specified "
            "actual evapotranspiration; computed in soilzone_ag (0=no; 1=yes)"
        ),
        "context": "scalar",
    },
    "springfrost_dynamic": {
        "datatype": "string",
        "description": (
            "Pathname to the dynamic spring frost parameter file."
        ),
        "context": "scalar",
        "force_default": False,
    },
    "dyn_springfrost_flag": {
        "datatype": "int32",
        "description": (
            "Flag governing the use of springfrost_dynamic file data "
            "(0=off; 1=on)."
        ),
        "context": "scalar",
        "force_default": False,
    },
    "fallfrost_dynamic": {
        "datatype": "string",
        "description": ("Pathname to the dynamic fall frost parameter file."),
        "context": "scalar",
        "force_default": False,
    },
    "dyn_fallfrost_flag": {
        "datatype": "int32",
        "description": (
            "Flag governing the use of fallfrost_dynamic file data "
            "(0=off; 1=on)."
        ),
        "context": "scalar",
        "force_default": False,
    },
}

# TODO move this code in to a global function to be called. like init_module.
# Make a local deep copy to avoid modifying the shared constants.pyprms_meta
pyprms_meta = deepcopy(_pyprms_meta)
pyprms_meta["control"] = pyprms_meta["control"] | additional_cbh_meta

ppmp = pyprms_meta["parameters"]
dt_conv = {"I": "int32", "F": "float32"}


class id_dict(dict):
    @staticmethod
    def __missing__(key):
        return key


dim_remap = id_dict(scalar="one", ndoy="ndays")

for kk in pws_meta.parameters.keys():
    if kk not in ppmp.keys():
        kk_pws_meta = pws_meta.parameters[kk].copy()
        kk_pws_meta["dimensions"] = list(kk_pws_meta.pop("dims"))
        kk_pws_meta["datatype"] = dt_conv[kk_pws_meta.pop("type")]
        kk_pws_meta["dimensions"] = [
            dim_remap[ii] for ii in kk_pws_meta["dimensions"]
        ]
        ppmp[kk] = kk_pws_meta

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
        start_time: Union[np.datetime64, None] = None,
        end_time: Union[np.datetime64, None] = None,
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
        self._start_time = start_time
        self._end_time = end_time

        # Get the full domain parameters
        # TODO: probably rename this one _full_control_pws and also track
        #       a _full_control_pp as well?
        self._full_control = pws.Control.load_prms(
            self._full_control_file, warn_unused_options=False
        )

        control = pp.ControlFile(
            filename=self._full_control_file,
            metadata=pyprms_meta,
            verbose=False,
        )
        self._full_control_var_names = control.control_variables.keys()
        del control
        self._cbh_control_names = list(
            set(cbh_control_names).intersection(self._full_control_var_names)
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
            for kk in self._cbh_control_names:
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

        # TODO: make these individual methods available externally?
        #       IE the case of Noah's parameter files.

        print("Subsetting dynamic parameter files.")
        self._subset_dynamic_params()

        print("Subsetting parameters.")
        self._subset_params()

        print("Subsetting CBH files.")
        self._subset_cbh()

        # make this conditional/lazy on output format?
        if (
            self._output_format is not None
            and self._output_format.lower == "prms"
        ):
            print("Subsetting data file.")
            self._subset_data_file()

        # Make this conditional on restart being active in the control file
        # print("Subsetting restarts.")
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
        """Only masks are solved here. Recalculating indexed variables happens
        later in _subset_params, etc...
        """
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

        # poi
        # associate poi with nhm_seg first, since that's actually useful. then
        # solve indices.
        poi_gage_segment_nhm = []
        for ii in self._full_params.npoigages:
            poi_gage_segment_nhm += [
                self._full_params.nhm_seg[
                    self._full_params.poi_gage_segment[ii]
                ].values.tolist()
            ]
        self._full_params["poi_gage_segment_nhm"] = (
            self._full_params["poi_gage_segment"] * 0 - 1
        )
        self._full_params["poi_gage_segment_nhm"].values = np.array(
            poi_gage_segment_nhm, dtype=np.int32
        )
        # now only keep the pois with nhm_seg in nhm_seg_df
        self._sub_pois_mask = self._full_params["poi_gage_segment_nhm"].isin(
            sub_nhm_seg_df.nhm_seg
        )
        nhm_poi_df = self._full_params["poi_gage_id"].to_pandas().reset_index()
        sub_poi_df = nhm_poi_df.iloc[self._sub_pois_mask.values]
        assert (np.diff(sub_poi_df.npoigages.values) > 0).all()
        self._sub_pois_order = sub_poi_df.poi_gage_id.values

        return None

    def _subset_cbh(self) -> None:
        if self._full_cbh_nc_files_dict is not None:
            self._subset_cbh_netcdf_files()
        else:
            self._subset_cbh_files()

    def _subset_cbh_files(self) -> None:
        raise NotImplementedError

    def _subset_cbh_netcdf_files(self):
        keys_drop = []
        for kk, vv in self._full_cbh_nc_files_dict.items():
            if kk not in self._cbh_control_names:
                warn(
                    f"'{kk}' CBH file not present in: "
                    f"{self._cbh_control_names}. Skipping"
                )
                keys_drop.append(kk)
            if not vv.exists():
                raise ValueError("Supplied CBH file path, '{vv}', not found.")

        for kk in keys_drop:
            del self._full_cbh_nc_files_dict[kk]

        # bit of a silly pipeline, dont see another way to rename dim
        # independently of the coordinate with a dataarray
        self._sub_cbh_files_dict = {
            kk: xr.load_dataset(vv).rename_dims(nhm_id="nhru")
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

        if self._start_time is not None and self._end_time is not None:
            if self._start_time is None:
                start_time = self._sub_cbh_files_dict[kk].time[0]
            else:
                start_time = self._start_time
            if self._end_time is None:
                end_time = self._sub_cbh_files_dict[kk].time[-1]
            else:
                end_time = self._end_time
            # <
            self._sub_cbh_files_dict = {
                kk: vv.sel(time=slice(start_time, end_time))
                for kk, vv in self._sub_cbh_files_dict.items()
            }

        return None

    def _subset_params(self) -> None:
        indexed_vars = [
            "from_segment",
            "from_segment_starts",
            "from_segment_ends",
            "tosegment",
            "from_nhm_seg_unstruct",
            "poi_gage_segment",
        ]

        # drop the indexed vars in advance of re-solving them
        self._sub_params = self._full_params.copy()
        indexed_vars_in_ds = set(indexed_vars).intersection(
            set(self._sub_params.variables)
        )
        self._sub_params = self._sub_params.drop_vars(indexed_vars_in_ds)

        # actually subset on npoigages=self._sub_poi_gages_mask
        # need to recalculate poi_gage_segment upon subsetting
        self._sub_params = (
            self._sub_params.isel(nhru=self._sub_nhm_ids_mask)
            .isel(nsegment=self._sub_nhm_segs_mask)
            .isel(npoigages=self._sub_pois_mask)
        )

        assert (
            self._sub_params.nhm_id.values == self._sub_nhm_ids_order
        ).all()
        assert (
            self._sub_params.nhm_seg.values == self._sub_nhm_segs_order
        ).all()
        assert (
            self._sub_params.poi_gage_id.values == self._sub_pois_order
        ).all()

        # re-solve: tosegment, hru_segment, poi_gage_segment
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

        # Add upstream tracing parameters
        self._get_from_segment_params(
            self_attr_key="_sub_params",
            parallel=self._from_seg_calc_parallel,
            check=self._from_seg_calc_check,
        )

        poi_gage_segment = []
        for ii in self._sub_params.npoigages.values:
            result = np.where(
                self._sub_params.poi_gage_segment_nhm.values[ii]
                == self._sub_params.nhm_seg.values
            )[0]
            len_result = len(result)
            if len_result == 1:
                poi_gage_segment += result.tolist()
            elif len_result == 0:
                poi_gage_segment += [-1]
            else:
                raise ValueError

        self._sub_params["poi_gage_segment"] = (
            self._sub_params["poi_gage_segment_nhm"] * 0
        )
        self._sub_params["poi_gage_segment"].values = (
            np.array(poi_gage_segment) + 1
        )

        return None

    def _subset_dynamic_params(self) -> None:
        """Subset dynamic parameter files if they are active in the control.

        This method checks for dynamic parameter files (ag_frac_dynamic,
        springfrost_dynamic, fallfrost_dynamic) in the control file,
        verifies their associated flags are enabled, and subsets them
        based on the HRU mask.
        """
        from .prms_dyn_param import (
            DYNAMIC_PARAM_CONFIG,
            PrmsDynamicParameter,
        )

        self._sub_dynamic_param_files = {}

        # Get the control directory for resolving relative paths
        control_dir = self._full_control_file.parent

        # Use pyPRMS control to access control variables
        control = pp.ControlFile(
            filename=self._full_control_file,
            metadata=pyprms_meta,
            verbose=False,
        )
        control_vars = control.control_variables

        for param_name, config in DYNAMIC_PARAM_CONFIG.items():
            flag_name = config["flag_name"]
            dtype = config["dtype"]

            # Check if the flag variable exists and is enabled
            if flag_name not in control_vars:
                continue

            flag_value = control_vars[flag_name].values
            if isinstance(flag_value, (list, np.ndarray)):
                flag_value = flag_value[0]
            flag_value = int(flag_value)

            if flag_value != 1:
                continue

            # Check if the file path variable exists
            if param_name not in control_vars:
                warn(
                    f"Dynamic parameter flag {flag_name} is set but "
                    f"{param_name} is not specified in control file."
                )
                continue

            file_path = control_vars[param_name].values
            if isinstance(file_path, (list, np.ndarray)):
                file_path = file_path[0]

            # Resolve the path relative to control directory
            file_path = pl.Path(file_path)
            if not file_path.is_absolute():
                file_path = control_dir / file_path

            if not file_path.exists():
                warn(
                    f"Dynamic parameter file not found: {file_path}. "
                    f"Skipping subsetting for {param_name}."
                )
                continue

            print(f"  Subsetting {param_name}: {file_path}")

            # Load and subset the dynamic parameter file
            try:
                dyn_param = PrmsDynamicParameter.load(file_path, dtype=dtype)
                subset_param = dyn_param.subset(self._sub_nhm_ids_mask)

                # Store the subsetted data and original filename for writing
                self._sub_dynamic_param_files[param_name] = {
                    "data": subset_param,
                    "original_path": file_path,
                    "control_var_name": param_name,
                }
            except Exception as e:
                warn(
                    f"Error subsetting dynamic parameter file {file_path}: "
                    f"{e}. Skipping."
                )
                continue

        return None

    def _subset_data_file(self):
        if hasattr(self, "_sub_data_file"):
            return None

        # _full_control is pws control, sub_control is pp
        data_file_name = pl.Path(self._sub_control.get("data_file").values)
        # since this path is likely relative to the control_file
        if not data_file_name.exists():
            data_file_name = self._full_control_file.parent / str(
                data_file_name
            )

        self._data_file_name = data_file_name
        # <
        data_file = pp.DataFile(data_file_name, metadata=pyprms_meta)
        # for now only handle runoff, throw an error if there are other
        # variables, mostly because I'm unsure how that will work for certain.
        runoff_mask = data_file.data.columns.str.contains("runoff")
        if not runoff_mask.all():
            raise NotImplementedError("")
        # <
        file_poi_ids = data_file.data.columns.str.slice(7).values
        file_poi_keep_mask = np.isin(
            file_poi_ids, self._sub_params.poi_gage_id.values
        )
        cols_to_drop = data_file.data.columns[~file_poi_keep_mask].tolist()
        self._sub_data_file = data_file.data.drop(columns=cols_to_drop)

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
        else:
            self._output_format = output_format

        if self._output_format is None:
            raise ValueError(
                "output_format not specified on initialization or write."
            )

        self._sub_control = pp.ControlFile(
            filename=self._full_control_file,
            metadata=pyprms_meta,
            verbose=False,
        )
        self._sub_control_file_name = (
            f"{self._full_control_file.stem}_subset.control"
        )

        if self._start_time is not None:
            self._sub_control["start_time"].values = self._start_time
        if self._end_time is not None:
            self._sub_control["end_time"].values = self._end_time

        if self._output_format.lower() == "pywatershed":
            self._write_pws(write_dir=write_dir)
        elif self._output_format.lower() == "prms":
            self._write_prms(write_dir=write_dir)

        return None

    def _write_prms(self, write_dir: pl.Path) -> None:
        # the following methods set the control, reusing the filename in it
        # but making it relative to the write_dir.
        self._cbh_dataset_to_ascii(write_dir=write_dir)

        # not an approved PRMS parameter, remove before writing
        del self._sub_params["poi_gage_segment_nhm"]
        self._parameters_to_ascii(write_dir=write_dir)

        self._data_file_to_ascii(write_dir=write_dir)

        # Write dynamic parameter files
        self._write_dynamic_params(write_dir=write_dir)

        # TODO: rename the data file in to the write_dir
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
            # check the order before output
            assert (vv.nhm_id.values == self._sub_nhm_ids_order).all()
            # The naming of cbh files, their control parameters, and their
            # internal variables is INSANE and a mess in PRMS.
            var_name = cbh_ctl_var_map[kk]
            file_name = f"{var_name}.nc"
            if var_name not in vv.data_vars:
                current_name = cbh_vars_rename_inv[var_name]
                vv = vv.rename(
                    {
                        current_name: var_name,
                    }
                )
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

        # Write dynamic parameter files
        self._write_dynamic_params(write_dir=write_dir)

        # write control file
        self._sub_control.write(write_dir / self._sub_control_file_name)
        return None

    def _write_dynamic_params(self, write_dir: pl.Path) -> None:
        """Write subsetted dynamic parameter files and update control.

        Args:
            write_dir: Directory to write the files to
        """
        if not hasattr(self, "_sub_dynamic_param_files"):
            return None

        if not self._sub_dynamic_param_files:
            return None

        for param_name, file_info in self._sub_dynamic_param_files.items():
            subset_param = file_info["data"]
            original_path = file_info["original_path"]

            # Use original filename, write directly to write_dir
            output_filename = original_path.name
            output_path = write_dir / output_filename
            subset_param.write(output_path)

            # Update control file to point to new filename (no path prefix)
            if param_name in self._sub_control.control_variables:
                self._sub_control.control_variables[
                    param_name
                ].values = output_filename
            else:
                # Add the control variable if it doesn't exist
                meta = additional_cbh_meta.get(param_name, {})
                vv = pp.ControlVariable(
                    name=param_name, strict=False, meta=meta
                )
                vv.__dict__["_ControlVariable__values"] = output_filename
                self._sub_control.control_variables[param_name] = vv

            print(f"  Wrote dynamic parameter file: {output_path}")

        return None

    def _cbh_dataset_to_ascii(self, write_dir):
        import uuid

        data_dir = pl.Path(pws.constants.__pywatershed_root__ / "data")
        # simply build and load a dummy CBH netcdf file from the pywatershed
        # repo and replace its dataset with ours.

        random_filename = str(uuid.uuid4()) + ".nc"
        dum_file_path = data_dir / random_filename
        make_dummy_netcdf_cbh_file(dum_file_path)

        pp_cbh = pp.Cbh(
            src_path=dum_file_path,
            metadata=pyprms_meta,
            engine="netcdf",
        )
        # the following two operation are only required on windows before
        # deleting the file, else it is busy.
        pp_cbh._Cbh__dataset.load()
        pp_cbh._Cbh__dataset.close()
        dum_file_path.unlink()

        cbh_ds = xr.merge(self._sub_cbh_files_dict.values())
        pp_cbh._Cbh__dataset = cbh_ds
        for kk, vv in self._sub_cbh_files_dict.items():
            var_name = list(vv.data_vars)[0]
            file_name = pl.Path(self._sub_control[kk].values).name
            self._sub_control[kk].values = file_name
            pp_cbh.write_ascii(
                filename=write_dir / file_name,
                variable=var_name,
                float_format=cbh_var_ascii_format[var_name],
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
        sub_params_ds = sub_params_ds.drop_vars(drop_vars)

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

    def _data_file_to_ascii(self, write_dir: pl.Path):
        # _subset_data_file is lazy and wont do anything if
        # hasattr(self, "_sub_data_file")
        self._subset_data_file()

        # there is no method for writing out the Datafile in pyprms.
        output_data_file = write_dir / self._data_file_name.name
        pws.utils.utils.write_data_file(self._sub_data_file, output_data_file)
        # edit the path to the data_file in the output control
        self._sub_control["data_file"].values = output_data_file.name

        return None


def make_dummy_netcdf_cbh_file(file_path: pl.Path):
    # it appears that any basic netcdf file works, no other requirements
    # imposed by pp.CBH()
    ds = xr.Dataset(
        data_vars=dict(
            foo=(
                ("time", "hru"),
                np.zeros([7, 3]),
                {"units": "parsec", "long_name": "foooo"},
            ),
        ),
        attrs=dict(
            title="Dummy",
        ),
    )
    ds.to_netcdf(file_path)

    return None
