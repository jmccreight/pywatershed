import pathlib as pl

import numpy as np
import xarray as xr

from . import meta
from .control import Control
from .model import Model

# TODO: Extend monthly stats to timeseries arrays in PRMSSolarGeometry and
#       PRMSAtmosphere.

# Implementing:
# * monthly accumulations
# * full-time data and stats on pois (eg median)
# * annual (or other periods) extremes (eg date of peak swe)

# Future possibilities:
# * fixed window stats on all data (e.g. monthly standard deviations)
# * rolling window stats on all data

spatial_dim_to_coord_name = {"nhru": "nhm_id", "nsegment": "nhm_seg"}


class CustomOutput:
    def __init__(
        self,
        control: Control,
        model: Model,
        monthly_accum_var_list: list | None = None,
        monthly_accum_stats: list | None = None,
        poi_var_list: list | None = None,
        poi_nhm_seg: list | None = None,
        poi_gage_segment: list | None = None,
        poi_stats: list | None = None,
        hru_sub_var_list: list | None = None,
        hru_sub_ids: list | None = None,
        hru_sub_stats: list | None = None,
    ):
        self._control = control
        self._model = model

        self._monthly_accum_var_list = monthly_accum_var_list
        self._monthly_accum_stats = monthly_accum_stats

        if (
            poi_var_list is not None
            and poi_nhm_seg is None
            and poi_gage_segment is None
        ):
            raise ValueError(
                "At least one of poi_nhm_seg or poi_gage_segment must be passed"
            )
        self._poi_var_list = poi_var_list
        self._poi_nhm_seg = poi_nhm_seg
        self._poi_gage_segment = poi_gage_segment
        self._poi_stats = poi_stats

        self._hru_sub_var_list = hru_sub_var_list
        self._hru_sub_ids = hru_sub_ids
        self._hru_sub_stats = hru_sub_stats

        self._current_time = self._control.init_time.copy()
        self._time_step = self._control.time_step.copy()

        self._init_monthly_stats()
        self._init_poi_sub_stats()

        return None

    def _init_monthly_stats(self):
        if self._monthly_accum_var_list is None:
            self._time_months = None
            self._n_days_per_month = None
            return None

        self._solve_monthly_time()
        self._map_monthly_vars_procs()
        self._declare_monthly_arrays()

        return None

    def _solve_monthly_time(self):
        import pandas as pd

        ctl = self._control
        self._time_months = pd.date_range(
            start=ctl.start_time, end=ctl.end_time, freq="MS"
        ).values.astype("datetime64[M]")
        self._n_days_per_month = self._time_months.copy().astype("int32") * 0

    def _map_monthly_vars_procs(self) -> None:
        self._monthly_vars_procs = {}
        for vv in self._monthly_accum_var_list:
            for pp in self._model.processes.keys():
                proc_vars = self._model.processes[pp].get_variables()
                if vv in proc_vars:
                    self._monthly_vars_procs[vv] = pp

        if not set(self._monthly_vars_procs.keys()) == set(
            self._monthly_accum_var_list
        ):
            raise ValueError(
                "Not all monthly accumulation variables were found among the "
                "model processes."
            )

    def _declare_monthly_arrays(self):
        self._monthly_arrays = {}
        for vv in self._monthly_accum_var_list:
            proc_name = self._monthly_vars_procs[vv]
            proc = self._model.processes[proc_name]
            var_meta = meta.find_variables(vv)[vv]
            spatial_dim_len = proc[vv].shape[0]
            spatial_dim_name = var_meta["dims"][0]
            spatial_coord_name = spatial_dim_to_coord_name[spatial_dim_name]
            spatial_coord = proc._params.coords[spatial_coord_name]
            new_shape = (len(self.time_months), spatial_dim_len)
            self._monthly_arrays[vv] = xr.DataArray(
                # zeros required for accumulations
                data=np.zeros(new_shape, dtype=var_meta["type"]),
                dims=["month", spatial_dim_name],
                coords={
                    "month": self._time_months,
                    spatial_dim_name: spatial_coord,
                },
                # reference_time=reference_time,
                attrs=dict(
                    description=var_meta["desc"],
                    units=var_meta["units"],
                    resolution="Monthly",
                ),
            )
            # Can not really be put into month resolution.
            # self._monthly_arrays[vv].month.attrs["units"] = "M"

    @property
    def time(self) -> np.ndarray | None:
        return self._time

    @property
    def time_months(self) -> np.ndarray | None:
        return self._time_months

    @property
    def n_days_per_month(self) -> np.ndarray | None:
        if self._finalized:
            return self._n_days_per_month
        else:
            return None

    @property
    def monthly_accumulations(self) -> dict | None:
        if self._finalized:
            return self._monthly_arrays
        else:
            return None

    def _get_month_index(self) -> None:
        current_month = self._current_time.astype("datetime64[M]")
        self._current_month_index = np.where(
            self._time_months == current_month
        )[0][0]

    def _accumulate_monthly_values(self) -> None:
        mon_ind = self._current_month_index
        self._n_days_per_month[mon_ind] += 1
        for vv in self._monthly_accum_var_list:
            proc_name = self._monthly_vars_procs[vv]
            self._monthly_arrays[vv][mon_ind, :] += self._model.processes[
                proc_name
            ][vv]

    def _init_poi_sub_stats(self) -> None:
        self._solve_time()
        self._map_poi_vars_procs()
        # self._map_hru_sub_vars_procs()
        self._declare_poi_hru_sub_arrays()

    def _solve_time(self):
        import pandas as pd

        ctl = self._control
        self._time = pd.date_range(
            start=ctl.start_time, end=ctl.end_time, freq="D"
        ).values.astype("datetime64[D]")

    def _map_poi_vars_procs(self):
        if self._poi_var_list is None:
            return
        self._poi_vars_procs = {}
        self._poi_indices = None
        for vv in self._poi_var_list:
            for pp in self._model.processes.keys():
                proc_vars = self._model.processes[pp].get_variables()
                if vv in proc_vars:
                    self._poi_vars_procs[vv] = pp
                    # check the dimensions are nsegment
                    vv_dims = meta.find_variables(vv)[vv]["dims"][0]
                    assert vv_dims == "nsegment"

        if self._poi_nhm_seg is not None:
            self._poi_inds = np.where(
                np.isin(
                    self._model.processes[pp]._params.parameters["nhm_seg"],
                    self._poi_nhm_seg,
                )
            )
            if self._poi_gage_segment is not None:
                assert (self._poi_inds == self._poi_gage_segment).all()
        else:
            self._poi_inds = self._poi_gage_segment

    def _map_hru_sub_vars_procs(self):
        if self._hru_sub_var_list is None:
            return
        self._hru_sub_vars_procs = {}
        self._hru_sub_indices = None
        for vv in self._hru_sub_var_list:
            for pp in self._model.processes.keys():
                proc_vars = self._model.processes[pp].get_variables()
                if vv in proc_vars:
                    self._hru_sub_vars_procs[vv] = pp
                    # check the dimensions are nsegment
                    vv_dims = meta.find_variables(vv)[vv]["dims"][0]
                    assert vv_dims == "nhru"

        self._hru_sub_inds = np.where(
            np.isin(
                self._model.processes[pp]._params.parameters["nhm_id"],
                self._hru_sub_ids,
            )
        )

    def _declare_poi_hru_sub_arrays(self):
        self._poi_arrays = {}
        self._hru_sub_arrays = {}

        poi_hru_sub_lists = [
            [
                self._poi_arrays,
                self._poi_var_list,
                self._poi_vars_procs,
            ],
            # [
            #     self._hru_sub_arrays,
            #     self._hru_sub_var_list,
            #     self._hru_sub_vars_procs,
            # ],
        ]

        for arrays, var_list, vars_procs in poi_hru_sub_lists:
            for vv in var_list:
                proc_name = vars_procs[vv]
                proc = self._model.processes[proc_name]
                var_meta = meta.find_variables(vv)[vv]
                spatial_dim_len = proc[vv][self._poi_inds].shape[0]
                spatial_dim_name = var_meta["dims"][0]
                spatial_coord_name = spatial_dim_to_coord_name[
                    spatial_dim_name
                ]
                spatial_coord = proc._params.coords[spatial_coord_name][
                    self._poi_inds
                ]
                new_shape = (len(self._time), spatial_dim_len)
                arrays[vv] = xr.DataArray(
                    # zeros required for accumulations
                    data=np.zeros(new_shape, dtype=var_meta["type"]) * np.nan,
                    dims=["time", spatial_dim_name],
                    coords={
                        "time": self._time,
                        spatial_dim_name: spatial_coord,
                    },
                    # reference_time=reference_time,
                    attrs=dict(
                        description=var_meta["desc"],
                        units=var_meta["units"],
                    ),
                )

    def _add_poi_hru_sub_data(self) -> None:
        poi_hru_sub_lists = [
            [
                self._poi_arrays,
                self._poi_var_list,
                self._poi_vars_procs,
            ],
            # [
            #     self._hru_sub_arrays,
            #     self._hru_sub_var_list,
            #     self._hru_sub_vars_procs,
            # ],
        ]
        time_ind = self._control.itime_step
        for arrays, var_list, vars_procs in poi_hru_sub_lists:
            for vv in var_list:
                proc_name = vars_procs[vv]
                arrays[vv][time_ind, :] = self._model.processes[proc_name][vv][
                    self._poi_inds
                ]

    @property
    def poi_arrays(self) -> dict | None:
        if self._finalized:
            return self._poi_arrays
        else:
            return None

    @property
    def hru_sub_arrays(self) -> dict | None:
        if self._finalized:
            return self._hru_sub_arrays
        else:
            return None

    def calculate(self, warn: bool = True) -> None:
        # The control.advance() must happen before the this calculate() method.
        if self._control.current_time != self._current_time + self._time_step:
            raise ValueError(
                "Calculation time requested does not match with control "
            )
        else:
            self._current_time = self._control.current_time.copy()

        self._get_month_index()
        self._accumulate_monthly_values()
        self._add_poi_hru_sub_data()

    def finalize(self):
        self._finalized = True

    def to_netcdf(self, output_dir: pl.Path):
        if not self._finalized:
            warn(
                "Output can only be written once the Output object is finalized"
            )

        self._monthly_to_netcdf(self)
        # self._poi_to_netcdf(self)
