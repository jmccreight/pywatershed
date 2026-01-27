"""Output collection and statistical analysis for pywatershed models.

Supports PRMS processes and FlowGraph. Collects three types of output:

1. Monthly accumulations - All spatial units (HRUs, segments, nodes)
2. POI (Points of Interest) - Time series and stats at specific segments/gages
3. HOI (HRUs of Interest) - Time series and stats for specific HRUs

Example
-------
>>> import pywatershed as pws
>>> from pywatershed.analysis.time_stats import mean
>>>
>>> def max_flow(da):
...     return da.max(dim="time")
...
>>>
>>> output = pws.base.Output(
...     control=control,
...     model=model,
...     monthly_accum_var_list=["sroff", "hru_actet"],
...     poi_var_list=["seg_outflow"],
...     poi_nhm_seg=poi_nhm_seg,
...     poi_stats={mean: ["seg_outflow"], max_flow: ["seg_outflow"]},
...     hoi_var_list=["hru_actet"],
...     hoi_ids=[1, 2, 3],
...     hoi_stats={mean: ["hru_actet"]},
... )
>>> model.run(finalize=True, output=output)
>>>
>>> # Access hierarchically: output.poi_stats[variable][statistic]
>>> output.poi_stats["seg_outflow"]["mean"]
>>> output.hoi_stats["hru_actet"]["mean"]

Notes
-----
- Statistics use dict[function: var_list] pattern
- Results stored hierarchically: output.poi_stats[variable][statistic]
- Each result has metadata: variable, statistic, period_of_record
- Must finalize before accessing statistics
"""

import pathlib as pl
import warnings
from typing import TYPE_CHECKING, Callable

import numpy as np
import xarray as xr

from . import meta
from .flow_graph import FlowGraph

if TYPE_CHECKING:
    from .control import Control
    from .model import Model

spatial_dim_to_coord_name = {
    "nhru": "nhm_id",
    "nsegment": "nhm_seg",
    "nnodes": "node_coord",
}


class Output:
    """Output collection and statistical analysis for models.

    Collects data during execution, computes statistics after finalization.
    Supports PRMS processes and FlowGraph.

    Parameters
    ----------
    control : Control
        Model control with timing information
    model : Model
        Pywatershed model instance
    monthly_accum_var_list : list[str], optional
        Variables to accumulate monthly (all spatial units)
    poi_var_list : list[str], optional
        Variables to collect at points of interest
    poi_nhm_seg : list[int], optional
        NHM segment IDs for POIs (portable across domains)
    poi_gage_segment : list[int], optional
        0-based segment indices for POIs (domain-specific)
    poi_stats : dict[Callable, list[str]], optional
        Statistics for POIs: {function: [var1, var2, ...]}
    hoi_var_list : list[str], optional
        Variables to collect for HRUs of interest
    hoi_ids : list[int], optional
        HRU IDs (nhm_id values) to include
    hoi_stats : dict[Callable, list[str]], optional
        Statistics for HOIs: {function: [var1, var2, ...]}

    Attributes
    ----------
    time : np.ndarray
        Daily time coordinate
    time_months : np.ndarray
        Monthly time coordinate
    n_days_per_month : xr.DataArray
        Days per month (for converting accumulations to means)
    monthly_accumulations : dict[str, xr.DataArray]
        Monthly values, available after finalization
    poi_arrays : dict[str, xr.DataArray]
        POI time series, available after finalization
    hoi_arrays : dict[str, xr.DataArray]
        HOI time series, available after finalization
    poi_stats : dict[str, dict[str, xr.DataArray]]
        POI statistics: poi_stats[variable][statistic]
    hoi_stats : dict[str, dict[str, xr.DataArray]]
        HOI statistics: hoi_stats[variable][statistic]

    Examples
    --------
    >>> from pywatershed.analysis.time_stats import mean
    >>> def max_flow(da):
    ...     return da.max(dim="time")
    ...
    >>>
    >>> output = pws.base.Output(
    ...     control=control,
    ...     model=model,
    ...     monthly_accum_var_list=["sroff", "hru_actet"],
    ...     poi_var_list=["seg_outflow"],
    ...     poi_nhm_seg=[12345, 67890],
    ...     poi_stats={mean: ["seg_outflow"], max_flow: ["seg_outflow"]},
    ...     hoi_var_list=["hru_actet"],
    ...     hoi_ids=[1, 2, 3],
    ...     hoi_stats={mean: ["hru_actet"]},
    ... )
    >>> model.run(finalize=True, output=output)
    >>>
    >>> # Access hierarchically
    >>> output.poi_stats["seg_outflow"]["mean"]
    >>> output.hoi_stats["hru_actet"]["mean"]
    """

    def __init__(
        self,
        control: "Control",
        model: "Model",
        monthly_accum_var_list: list | None = None,
        poi_var_list: list | None = None,
        poi_nhm_seg: list | None = None,
        poi_gage_segment: list | None = None,
        poi_stats: dict[Callable, list[str]] | None = None,
        hoi_var_list: list | None = None,
        hoi_ids: list | None = None,
        hoi_stats: dict[Callable, list[str]] | None = None,
    ):
        """Initialize Output and set up data collection."""
        self._finalized = False
        self._control = control
        self._model = model

        self._monthly_accum_var_list = monthly_accum_var_list

        if (
            poi_var_list is not None
            and poi_nhm_seg is None
            and poi_gage_segment is None
        ):
            raise ValueError(
                "At least one of poi_nhm_seg or poi_gage_segment must be "
                "passed when poi variables are requested."
            )

        self._poi_var_list = poi_var_list
        self._poi_nhm_seg = poi_nhm_seg
        self._poi_gage_segment = poi_gage_segment
        self._poi_stats = poi_stats

        self._hoi_var_list = hoi_var_list
        self._hoi_ids = hoi_ids
        self._hoi_stats = hoi_stats

        self._current_time = self._control.init_time.copy()
        self._time_step = self._control.time_step.copy()

        self._init_monthly()
        self._init_poi_sub()

        return None

    # ==== Properties =========================
    @property
    def time(self) -> np.ndarray | None:
        """Daily time coordinate for POI/HOI data."""
        return self._time

    @property
    def time_months(self) -> np.ndarray | None:
        """Monthly time coordinate for accumulations."""
        return self._time_months

    @property
    def n_days_per_month(self) -> xr.DataArray | None:
        """Number of days in each month for stats.

        Returns
        -------
        xr.DataArray or None
            DataArray of day counts per month with month dimension, useful for
            converting accumulations to means. Only available after finalization.
        """
        if self._finalized:
            return self._n_days_per_month
        else:
            warnings.warn(
                "n_days_per_month is only available after finalization. "
                "Call output.finalize() or model.run(finalize=True)."
            )
            return None

    @property
    def monthly_accumulations(self) -> dict[str, xr.DataArray] | None:
        """Monthly accumulations (available after finalization)."""
        if self._finalized:
            return self._monthly_arrays
        else:
            warnings.warn(
                "monthly_accumulations is only available after finalization. "
                "Call output.finalize() or model.run(finalize=True)."
            )
            return None

    @property
    def poi_arrays(self) -> dict[str, xr.DataArray] | None:
        """POI time series (available after finalization)."""
        if self._finalized:
            return self._poi_arrays
        else:
            warnings.warn(
                "poi_arrays is only available after finalization. "
                "Call output.finalize() or model.run(finalize=True)."
            )
            return None

    @property
    def hoi_arrays(self) -> dict[str, xr.DataArray] | None:
        """HOI time series (available after finalization)."""
        if self._finalized:
            return self._hoi_arrays
        else:
            warnings.warn(
                "hoi_arrays is only available after finalization. "
                "Call output.finalize() or model.run(finalize=True)."
            )
            return None

    @property
    def poi_stats(self) -> dict[str, dict[str, xr.DataArray]] | None:
        """POI statistics: poi_stats[variable][statistic] (after finalization)."""
        if self._finalized:
            return self._poi_stats_results
        else:
            warnings.warn(
                "poi_stats is only available after finalization. "
                "Call output.finalize() or model.run(finalize=True)."
            )
            return None

    @property
    def hoi_stats(self) -> dict[str, dict[str, xr.DataArray]] | None:
        """HOI statistics: hoi_stats[variable][statistic] (after finalization)."""
        if self._finalized:
            return self._hoi_stats_results
        else:
            warnings.warn(
                "hoi_stats is only available after finalization. "
                "Call output.finalize() or model.run(finalize=True)."
            )
            return None

    # ==== Validation methods =====================

    # ==== Monthly accumulation section =====================
    def _init_monthly(self) -> None:
        """Initialize monthly accumulation data structures."""
        if self._monthly_accum_var_list is None:
            self._time_months = None
            self._n_days_per_month = None
            return None

        self._solve_monthly_time()
        self._map_monthly_vars_procs()
        self._declare_monthly_arrays()

    def _solve_monthly_time(self) -> None:
        """Create monthly time coordinate and initialize day counter."""
        import pandas as pd

        ctl = self._control
        self._time_months = pd.date_range(
            start=ctl.start_time, end=ctl.end_time, freq="MS"
        ).values.astype("datetime64[M]")
        self._n_days_per_month = xr.DataArray(
            data=np.zeros(len(self._time_months), dtype="int32"),
            dims=["month"],
            coords={"month": self._time_months},
            attrs=dict(
                description="Number of days in each month",
                units="days",
            ),
        )

    def _map_monthly_vars_procs(self) -> None:
        """Map monthly variables to their source processes.

        Raises
        ------
        ValueError
            If any requested variable is not found in model processes
        """
        self._monthly_vars_procs = {}
        for vv in self._monthly_accum_var_list:
            for pp in self._model.processes.keys():
                proc_vars = self._model.processes[pp].get_variables()
                if vv in proc_vars:
                    self._monthly_vars_procs[vv] = pp
                elif (
                    isinstance(
                        proc := self._model.processes[pp],
                        FlowGraph,
                    )
                    and vv in proc["_addtl_output_vars"]
                ):
                    self._monthly_vars_procs[vv] = pp

        if not set(self._monthly_vars_procs.keys()) == set(
            self._monthly_accum_var_list
        ):
            raise ValueError(
                "Not all monthly accumulation variables were found among the "
                "model processes."
            )

    def _declare_monthly_arrays(self):
        """Declare xarray DataArrays for monthly accumulations."""
        self._monthly_arrays = {}
        for vv in self._monthly_accum_var_list:
            proc_name = self._monthly_vars_procs[vv]
            proc = self._model.processes[proc_name]
            var_meta = meta.find_variables(vv)
            if (
                not var_meta
                and hasattr(proc, "_addtl_output_vars")
                and vv in proc._addtl_output_vars
            ):
                var_meta = proc.meta[vv]
                var_meta["desc"] = vv
                var_meta["units"] = "unknown"
            else:
                var_meta = var_meta[vv]
            # <
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
                    spatial_coord_name: ([spatial_dim_name], spatial_coord),
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

    def _get_month_index(self) -> int:
        """Determine current month index for accumulation."""
        current_month = self._current_time.astype("datetime64[M]")
        self._current_month_index = np.where(
            self._time_months == current_month
        )[0][0]
        return self._current_month_index

    def _accumulate_monthly_values(self) -> None:
        """Accumulate current timestep values into monthly arrays."""
        if not self._monthly_accum_var_list:
            return

        self._get_month_index()
        mon_ind = self._current_month_index
        self._n_days_per_month.values[mon_ind] += 1
        for vv in self._monthly_accum_var_list:
            proc_name = self._monthly_vars_procs[vv]
            self._monthly_arrays[vv][mon_ind, :] += self._model.processes[
                proc_name
            ][vv]

    # ==== POI + HRU SUB section =====================
    def _init_poi_sub(self) -> None:
        """Initialize POI and HRU subset data structures."""
        if not self._poi_var_list and not self._hoi_var_list:
            # Initialize empty lists so other methods don't error
            self._poi_hoi_data_list = []
            self._poi_hoi_stats_list = []
            return None

        self._solve_time()
        self._solve_poi_stat_list()
        self._solve_hoi_stat_list()
        self._map_poi_vars_procs()
        self._map_hoi_vars_procs()
        # Initialize empty dicts first
        self._poi_arrays = {}
        self._hoi_arrays = {}
        # Build iteration lists (references the empty dicts)
        self._build_poi_hoi_iteration_lists()
        # Now populate the arrays
        self._declare_poi_hoi_arrays()

    def _solve_time(self) -> None:
        """Create daily time coordinate for full time series data."""
        import pandas as pd

        ctl = self._control
        self._time = pd.date_range(
            start=ctl.start_time, end=ctl.end_time, freq="D"
        ).values.astype("datetime64[D]")

    def _solve_poi_stat_list(self) -> None:
        """Build POI statistics dict from function: var_list mapping."""
        self._poi_stat_func_vars = {}
        if self._poi_stats is None:
            return
        for func, var_list in self._poi_stats.items():
            if not callable(func):
                raise ValueError("poi_stats keys must be callable functions.")
            self._poi_stat_func_vars[func] = var_list

    def _solve_hoi_stat_list(self) -> None:
        """Build HRU subset statistics dict from function: var_list mapping."""
        self._hoi_stat_func_vars = {}
        if self._hoi_stats is None:
            return
        for func, var_list in self._hoi_stats.items():
            if not callable(func):
                raise ValueError("hoi_stats keys must be callable functions.")
            self._hoi_stat_func_vars[func] = var_list

    def _map_poi_vars_procs(self) -> None:
        """Map POI variables to processes and resolve indices."""
        if self._poi_var_list is None:
            return
        self._poi_vars_procs = {}
        self._poi_indices = None
        for vv in self._poi_var_list:
            for pp in self._model.processes.keys():
                proc = self._model.processes[pp]
                proc_vars = proc.get_variables()
                if hasattr(proc, "_addtl_output_vars"):
                    proc_vars += proc._addtl_output_vars
                if vv in proc_vars:
                    self._poi_vars_procs[vv] = pp
                    # check the dimensions are nsegment
                    vv_dims = proc.meta[vv]["dims"][0]
                    # vv_dims = meta.find_variables(vv)[vv]["dims"][0]
                    if vv_dims != "nsegment" and vv_dims != "nnodes":
                        raise ValueError(
                            f"Variable '{vv}' does not have dimension "
                            "'nsegment' nor 'nnodes'."
                        )

        if "pp" not in locals().keys():
            return

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

    def _map_hoi_vars_procs(self) -> None:
        """Map HRU subset variables to processes and resolve indices."""
        if self._hoi_var_list is None:
            return
        self._hoi_vars_procs = {}
        self._hoi_indices = None
        for vv in self._hoi_var_list:
            for pp in self._model.processes.keys():
                proc_vars = self._model.processes[pp].get_variables()
                if vv in proc_vars:
                    self._hoi_vars_procs[vv] = pp
                    # check the dimensions are nsegment
                    vv_dims = meta.find_variables(vv)[vv]["dims"][0]
                    if vv_dims != "nhru":
                        raise ValueError(
                            f"Variable '{vv}' does not have dimension "
                            "'nsegment'."
                        )

        self._hoi_inds = np.where(
            np.isin(
                self._model.processes[pp]._params.parameters["nhm_id"],
                self._hoi_ids,
            )
        )

    def _build_poi_hoi_iteration_lists(self) -> None:
        """Build and cache iteration lists to avoid rebuilding each timestep."""
        # For _add_poi_hoi_data (called every timestep)
        self._poi_hoi_data_list = []
        if self._poi_var_list is not None:
            self._poi_hoi_data_list.append(
                (
                    self._poi_arrays,
                    self._poi_var_list,
                    self._poi_vars_procs,
                    self._poi_inds,
                )
            )
        if self._hoi_var_list is not None:
            self._poi_hoi_data_list.append(
                (
                    self._hoi_arrays,
                    self._hoi_var_list,
                    self._hoi_vars_procs,
                    self._hoi_inds,
                )
            )

        # For _calculate_poi_hoi_stats (called once at finalization)
        self._poi_hoi_stats_list = []
        if self._poi_stats is not None:
            self._poi_hoi_stats_list.append(
                (
                    "poi",  # marker to identify which stats dict to use
                    self._poi_arrays,
                    self._poi_stat_func_vars,
                )
            )
        if self._hoi_stats is not None:
            self._poi_hoi_stats_list.append(
                (
                    "hoi",  # marker to identify which stats dict to use
                    self._hoi_arrays,
                    self._hoi_stat_func_vars,
                )
            )

    def _declare_poi_hoi_arrays(self) -> None:
        """Declare xarray DataArrays for POI and HRU subset variables."""
        # Use cached iteration list instead of rebuilding
        for arrays, var_list, vars_procs, inds in self._poi_hoi_data_list:
            for vv in var_list:
                proc_name = vars_procs[vv]
                proc = self._model.processes[proc_name]
                var_meta = meta.find_variables(vv)
                if (
                    not var_meta
                    and hasattr(proc, "_addtl_output_vars")
                    and vv in proc._addtl_output_vars
                ):
                    var_meta = proc.meta[vv]
                    var_meta["desc"] = vv
                    var_meta["units"] = "unknown"
                else:
                    var_meta = var_meta[vv]
                # <
                spatial_dim_len = proc[vv][inds].shape[0]
                spatial_dim_name = var_meta["dims"][0]
                spatial_coord_name = spatial_dim_to_coord_name[
                    spatial_dim_name
                ]
                spatial_coord = proc._params.coords[spatial_coord_name][inds]
                new_shape = (len(self._time), spatial_dim_len)
                arrays[vv] = xr.DataArray(
                    data=np.full(new_shape, np.nan, dtype=var_meta["type"]),
                    dims=["time", spatial_dim_name],
                    coords={
                        "time": self._time,
                        spatial_coord_name: (
                            [spatial_dim_name],
                            spatial_coord,
                        ),
                    },
                    # reference_time=reference_time,
                    attrs=dict(
                        description=var_meta["desc"],
                        units=var_meta["units"],
                    ),
                )

    def _add_poi_hoi_data(self) -> None:
        """Add current timestep data to POI and HRU subset arrays."""
        if not self._poi_hoi_data_list:
            return

        time_ind = self._control.itime_step
        for arrays, var_list, vars_procs, inds in self._poi_hoi_data_list:
            for vv in var_list:
                proc_name = vars_procs[vv]
                arrays[vv][time_ind, :] = self._model.processes[proc_name][vv][
                    inds
                ]

    def _calculate_poi_hoi_stats(self) -> None:
        """Calculate POI/HOI statistics, store as stats[var][func_name]."""
        self._poi_stats_results = {}
        self._hoi_stats_results = {}

        for stats_type, arrays, stat_func_vars in self._poi_hoi_stats_list:
            # Get the appropriate stats dict
            stats = (
                self._poi_stats_results
                if stats_type == "poi"
                else self._hoi_stats_results
            )

            for func, var_list in stat_func_vars.items():
                func_name = func.__name__
                for vv in var_list:
                    if vv not in arrays:
                        continue
                    # Create nested dict structure: stats[var][stat_name]
                    if vv not in stats:
                        stats[vv] = {}

                    # Calculate the statistic
                    result = func(arrays[vv])

                    # Set the name to the variable name
                    result.name = vv

                    # Add metadata attributes
                    result.attrs["variable"] = vv
                    result.attrs["statistic"] = func_name

                    # Add period of record from original time series
                    time_coord = arrays[vv].coords["time"]
                    period_start = str(time_coord.values[0])
                    period_end = str(time_coord.values[-1])
                    result.attrs["period_of_record"] = (
                        f"{period_start} to {period_end}"
                    )

                    stats[vv][func_name] = result

    # ==== General methods ================
    def calculate(self) -> None:
        """Collect data for current timestep (called by model.run())."""
        # The control.advance() must happen before the this calculate() method.
        if self._control.current_time != self._current_time + self._time_step:
            raise ValueError(
                "Calculation time requested does not match with control "
            )
        else:
            self._current_time = self._control.current_time.copy()

        self._accumulate_monthly_values()
        self._add_poi_hoi_data()

    def finalize(self) -> None:
        """Finalize and calculate statistics (called by model.run())."""
        self._finalized = True
        self._calculate_poi_hoi_stats()

    def to_netcdf(self, output_dir: pl.Path) -> None:
        """Write output to netCDF files (not yet implemented)."""
        if not self._finalized:
            warnings.warn(
                "Output can only be written once the Output object is finalized"
            )
            return

        # self._monthly_to_netcdf(self)
        # self._poi_to_netcdf(self)

        raise NotImplementedError("YET.")
