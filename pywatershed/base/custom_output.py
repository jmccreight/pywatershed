"""Custom output functionality for pywatershed models.

This module provides flexible output collection and statistical analysis for
pywatershed models, including support for both PRMS processes and FlowGraph.

Output Types
------------
1. **Monthly accumulations**: Accumulate variable values over monthly periods
   for all spatial units (HRUs, segments, or nodes).

2. **Point of Interest (POI) data**: Collect full time series at specific
   locations (e.g., gage stations) with optional statistical aggregations.

3. **HRU subset data**: Collect full time series for specific HRUs with
   optional statistical aggregations.

Built-in Statistics
-------------------
Basic: mean, median, std
Hydrological: seven_day_mean_calendar_year_max, seven_day_mean_water_year_min,
etc.

All data uses xarray DataArrays with proper coordinate systems and metadata.

Example
-------
>>> import pywatershed as pws
>>>
>>> # Define a custom statistic function
>>> def max_flow(da, dim=None, **kwargs):
...     return da.max(dim=dim, **kwargs)
...
>>>
>>> # Create custom output collector
>>> output = pws.base.CustomOutput(
...     control=control,
...     model=model,
...     monthly_accum_var_list=["sroff", "hru_actet"],
...     poi_var_list=["seg_outflow"],
...     poi_nhm_seg=poi_nhm_seg,
...     poi_stats=["mean", "median", max_flow],
...     poi_stats_resample={"median": "1MS", "max_flow": "5D"},
...     hru_sub_var_list=["hru_actet", "pkwater_equiv"],
...     hru_sub_ids=[nhm_id_list],
...     hru_sub_stats=["mean", max_flow],
...     hru_sub_stats_resample={"mean": "1MS", "max_flow": "1YS"},
... )
>>>
>>> # Run model with custom output
>>> model.run(finalize=True, output=output)
>>>
>>> # Access results
>>> monthly_data = output.monthly_accumulations
>>> poi_timeseries = output.poi_arrays
>>> poi_statistics = output.poi_stats
>>> hru_data = output.hru_sub_arrays
>>> hru_statistics = output.hru_sub_stats

Notes
-----
- All time series data and statistics are returned as xarray DataArrays with
  proper coordinate systems and metadata.
- Custom statistic functions can be provided as callables that accept a
  DataArray as the first argument.
- Temporal grouping (e.g., by month) and resampling (e.g., to monthly) can be
  applied independently to different statistics.
- The output object must be finalized before accessing calculated statistics.

TODO
----
- Monthly stats for PRMSSolarGeometry and PRMSAtmosphere timeseries
- Additional annual extremes (e.g., date of peak SWE)
- Fixed/rolling window stats on all spatial units

See Also
--------
xarray time series documentation:
https://docs.xarray.dev/en/stable/user-guide/time-series.html#datetime-components
"""

import pathlib as pl
import warnings
from typing import TYPE_CHECKING, Callable

import numpy as np
import xarray as xr

from ..analysis import time_stats
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

# Re-export statistical functions from time_stats for convenience
mean = time_stats.mean
std = time_stats.std
median = time_stats.median
seasonal_min_max = time_stats.seasonal_min_max
rolling_mean_seasonal_min_max = time_stats.rolling_mean_seasonal_min_max
seven_day_mean_calendar_year_max = time_stats.seven_day_mean_calendar_year_max
seven_day_mean_water_year_max = time_stats.seven_day_mean_water_year_max
seven_day_mean_calendar_year_min = time_stats.seven_day_mean_calendar_year_min
seven_day_mean_water_year_min = time_stats.seven_day_mean_water_year_min


class CustomOutput:
    """Flexible output collection and statistical analysis for models.

    Collects data during model execution and computes statistics after
    finalization. Supports PRMS processes and FlowGraph. All data returned
    as xarray DataArrays.

    Output Types:
    - Monthly accumulations for all spatial units
    - POI (Point of Interest) time series and statistics
    - HRU subset time series and statistics

    Parameters
    ----------
    control : Control
        Model control object containing timing information
    model : Model
        The pywatershed model instance
    monthly_accum_var_list : list of str, optional
        List of variable names to accumulate monthly for all spatial units
    poi_var_list : list of str, optional
        List of variable names to collect at points of interest (POIs)
    poi_nhm_seg : list of int, optional if poi_gage_segment is supplied
        NHM segment IDs for POIs (portable across domains)
    poi_gage_segment : list of int, optional if poi_nhm_seg is supplied
        This is the name of the PRMS parameter which is 1-based, please
        subtract 1 before passing here as a 0-based segment index for POIs
        (domain-specific)
    poi_stats : list of str or callable, optional
        Statistics to calculate on POI time series. Can be strings referencing
        built-in functions ("mean", "median", "std") or custom callables
    poi_stats_groupby : dict, optional
        Mapping of statistic names to temporal grouping
        (e.g., {"median": "month"})
    poi_stats_resample : dict, optional
        Mapping of statistic names to resampling frequencies
        (e.g., {"median": "1MS", "max": "5D"})
    hru_sub_var_list : list of str, optional
        List of variable names to collect for HRU subsets
    hru_sub_ids : list of int, optional
        List of HRU IDs (nhm_id values) to include in subset
    hru_sub_stats : list of str or callable, optional
        Statistics to calculate on HRU subset time series
    hru_sub_stats_groupby : dict, optional
        Mapping of statistic names to temporal grouping for HRU subsets
    hru_sub_stats_resample : dict, optional
        Mapping of statistic names to resampling frequencies for HRU subsets

    Raises
    ------
    ValueError
        If poi_var_list is provided without poi_nhm_seg or poi_gage_segment

    Attributes
    ----------
    time : np.ndarray
        Daily time coordinate for full time series (POI and HRU subset data)
    time_months : np.ndarray
        Monthly time coordinate for monthly accumulations
    n_days_per_month : xr.DataArray
        DataArray of day counts per month with month dimension (useful for
        converting accumulations to means)
    monthly_accumulations : dict of xr.DataArray
        Monthly accumulated values for each variable (available after
        finalization)
    poi_arrays : dict of xr.DataArray
        Full time series for POI variables (available after finalization)
    hru_sub_arrays : dict of xr.DataArray
        Full time series for HRU subset variables (available after
        finalization)
    poi_stats : dict of xr.DataArray
        Calculated statistics for POI variables (available after finalization)
    hru_sub_stats : dict of xr.DataArray
        Calculated statistics for HRU subset variables (available after
        finalization)

    Examples
    --------
    >>> import pywatershed as pws
    >>>
    >>> # Create output with monthly accumulations and POI statistics
    >>> output = pws.base.CustomOutput(
    ...     control=control,
    ...     model=nhm_model,
    ...     monthly_accum_var_list=["sroff", "hru_actet", "seg_outflow"],
    ...     poi_var_list=["seg_outflow"],
    ...     poi_nhm_seg=poi_nhm_seg,
    ...     poi_stats=["mean", "median"],
    ...     poi_stats_groupby={"median": "month"},
    ...     poi_stats_resample={"mean": "1MS"},
    ... )
    >>>
    >>> # Run model (calculate method is called automatically each timestep)
    >>> nhm_model.run(finalize=True, output=output)
    >>>
    >>> # Access monthly accumulations
    >>> monthly_sroff = output.monthly_accumulations["sroff"]
    >>>
    >>> # Access POI statistics
    >>> monthly_mean_flow = output.poi_stats["seg_outflow_mean_1MS"]
    >>> monthly_median_by_month = output.poi_stats["seg_outflow_median_month"]

    Notes
    -----
    - The calculate() method is called automatically during model.run() at
      each timestep to accumulate data
    - The finalize() method is called automatically when
      model.run(finalize=True) to compute statistics
    - Custom statistic functions must accept a DataArray as first argument and
      should support dim, skipna, and keep_attrs parameters
    - Statistics naming convention: {variable}_{statistic}_{temporal_operation}
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
        hru_sub_var_list: list | None = None,
        hru_sub_ids: list | None = None,
        hru_sub_stats: dict[Callable, list[str]] | None = None,
    ):
        """Initialize CustomOutput and set up data collection structures."""
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

        self._hru_sub_var_list = hru_sub_var_list
        self._hru_sub_ids = hru_sub_ids
        self._hru_sub_stats = hru_sub_stats

        self._current_time = self._control.init_time.copy()
        self._time_step = self._control.time_step.copy()

        self._init_monthly()
        self._init_poi_sub()

        return None

    # ==== Properties =========================
    @property
    def time(self) -> np.ndarray | None:
        """Time coordinate for full-time stats.

        Returns
        -------
        np.ndarray or None
            Daily time coordinate array spanning the simulation period, or None
            if POI/HRU subset collection is not configured
        """
        return self._time

    @property
    def time_months(self) -> np.ndarray | None:
        """Month coordinate for monthly accumulations.

        Returns
        -------
        np.ndarray or None
            Monthly time coordinate array spanning the simulation period, or
            None if monthly accumulation is not configured
        """
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
        """Dictionary of monthly accumulated xr.DataArrays.

        Returns
        -------
        dict of xr.DataArray or None
            Monthly accumulated values for each variable. Only available after
            finalization.
        """
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
        """Dictionary of POI data in xr.DataArrays.

        Returns
        -------
        dict of xr.DataArray or None
            Full time series data for each POI variable. Only available after
            finalization.
        """
        if self._finalized:
            return self._poi_arrays
        else:
            warnings.warn(
                "poi_arrays is only available after finalization. "
                "Call output.finalize() or model.run(finalize=True)."
            )
            return None

    @property
    def hru_sub_arrays(self) -> dict[str, xr.DataArray] | None:
        """Dictionary of HRU subset data in xr.DataArrays.

        Returns
        -------
        dict of xr.DataArray or None
            Full time series data for each HRU subset variable. Only available
            after finalization.
        """
        if self._finalized:
            return self._hru_sub_arrays
        else:
            warnings.warn(
                "hru_sub_arrays is only available after finalization. "
                "Call output.finalize() or model.run(finalize=True)."
            )
            return None

    @property
    def poi_stats(self) -> dict[str, xr.DataArray] | None:
        """Dictionary of POI stats in xr.DataArrays.

        Returns
        -------
        dict of xr.DataArray or None
            Calculated statistics for POI variables. Only available after
            finalization.
        """
        if self._finalized:
            return self._poi_stats_results
        else:
            warnings.warn(
                "poi_stats is only available after finalization. "
                "Call output.finalize() or model.run(finalize=True)."
            )
            return None

    @property
    def hru_sub_stats(self) -> dict[str, xr.DataArray] | None:
        """Dictionary of HRU subset stats in xr.DataArrays.

        Returns
        -------
        dict of xr.DataArray or None
            Calculated statistics for HRU subset variables. Only available
            after finalization.
        """
        if self._finalized:
            return self._hru_sub_stats_results
        else:
            warnings.warn(
                "hru_sub_stats is only available after finalization. "
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
        if not self._poi_var_list and not self._hru_sub_var_list:
            # Initialize empty lists so other methods don't error
            self._poi_hru_sub_data_list = []
            self._poi_hru_sub_stats_list = []
            return None

        self._solve_time()
        self._solve_poi_stat_list()
        self._solve_hru_sub_stat_list()
        self._map_poi_vars_procs()
        self._map_hru_sub_vars_procs()
        # Initialize empty dicts first
        self._poi_arrays = {}
        self._hru_sub_arrays = {}
        # Build iteration lists (references the empty dicts)
        self._build_poi_hru_sub_iteration_lists()
        # Now populate the arrays
        self._declare_poi_hru_sub_arrays()

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

    def _solve_hru_sub_stat_list(self) -> None:
        """Build HRU subset statistics dict from function: var_list mapping."""
        self._hru_sub_stat_func_vars = {}
        if self._hru_sub_stats is None:
            return
        for func, var_list in self._hru_sub_stats.items():
            if not callable(func):
                raise ValueError(
                    "hru_sub_stats keys must be callable functions."
                )
            self._hru_sub_stat_func_vars[func] = var_list

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

    def _map_hru_sub_vars_procs(self) -> None:
        """Map HRU subset variables to processes and resolve indices."""
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
                    if vv_dims != "nhru":
                        raise ValueError(
                            f"Variable '{vv}' does not have dimension "
                            "'nsegment'."
                        )

        self._hru_sub_inds = np.where(
            np.isin(
                self._model.processes[pp]._params.parameters["nhm_id"],
                self._hru_sub_ids,
            )
        )

    def _build_poi_hru_sub_iteration_lists(self) -> None:
        """Build and cache iteration lists to avoid rebuilding each timestep."""
        # For _add_poi_hru_sub_data (called every timestep)
        self._poi_hru_sub_data_list = []
        if self._poi_var_list is not None:
            self._poi_hru_sub_data_list.append(
                (
                    self._poi_arrays,
                    self._poi_var_list,
                    self._poi_vars_procs,
                    self._poi_inds,
                )
            )
        if self._hru_sub_var_list is not None:
            self._poi_hru_sub_data_list.append(
                (
                    self._hru_sub_arrays,
                    self._hru_sub_var_list,
                    self._hru_sub_vars_procs,
                    self._hru_sub_inds,
                )
            )

        # For _calculate_poi_hru_sub_stats (called once at finalization)
        self._poi_hru_sub_stats_list = []
        if self._poi_stats is not None:
            self._poi_hru_sub_stats_list.append(
                (
                    "poi",  # marker to identify which stats dict to use
                    self._poi_arrays,
                    self._poi_stat_func_vars,
                )
            )
        if self._hru_sub_stats is not None:
            self._poi_hru_sub_stats_list.append(
                (
                    "hru_sub",  # marker to identify which stats dict to use
                    self._hru_sub_arrays,
                    self._hru_sub_stat_func_vars,
                )
            )

    def _declare_poi_hru_sub_arrays(self) -> None:
        """Declare xarray DataArrays for POI and HRU subset variables."""
        # Use cached iteration list instead of rebuilding
        for arrays, var_list, vars_procs, inds in self._poi_hru_sub_data_list:
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

    def _add_poi_hru_sub_data(self) -> None:
        """Add current timestep data to POI and HRU subset arrays."""
        if not self._poi_hru_sub_data_list:
            return

        time_ind = self._control.itime_step
        for arrays, var_list, vars_procs, inds in self._poi_hru_sub_data_list:
            for vv in var_list:
                proc_name = vars_procs[vv]
                arrays[vv][time_ind, :] = self._model.processes[proc_name][vv][
                    inds
                ]

    def _calculate_poi_hru_sub_stats(self) -> None:
        """Calculate statistics for POI and HRU subset data.

        Statistics stored hierarchically: stats[variable][function_name]
        Each DataArray is named and has attrs: variable, statistic, period_of_record
        """
        self._poi_stats_results = {}
        self._hru_sub_stats_results = {}

        for stats_type, arrays, stat_func_vars in self._poi_hru_sub_stats_list:
            # Get the appropriate stats dict
            stats = (
                self._poi_stats_results
                if stats_type == "poi"
                else self._hru_sub_stats_results
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
        """Collect data for current timestep.

        Called automatically by model.run() after control.advance().

        Raises
        ------
        ValueError
            If control time does not match expected timestep progression
        """
        # The control.advance() must happen before the this calculate() method.
        if self._control.current_time != self._current_time + self._time_step:
            raise ValueError(
                "Calculation time requested does not match with control "
            )
        else:
            self._current_time = self._control.current_time.copy()

        self._accumulate_monthly_values()
        self._add_poi_hru_sub_data()

    def finalize(self) -> None:
        """Finalize output and calculate statistics.

        Called automatically by model.run(finalize=True). After finalization,
        all output properties become accessible.
        """
        self._finalized = True
        self._calculate_poi_hru_sub_stats()

    def to_netcdf(self, output_dir: pl.Path) -> None:
        """Write output to netCDF files.

        Parameters
        ----------
        output_dir : pathlib.Path
            Output directory

        Raises
        ------
        NotImplementedError
            Not yet implemented
        """
        if not self._finalized:
            warnings.warn(
                "Output can only be written once the Output object is finalized"
            )
            return

        # self._monthly_to_netcdf(self)
        # self._poi_to_netcdf(self)

        raise NotImplementedError("YET.")
