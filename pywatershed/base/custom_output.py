"""Custom output functionality for pywatershed models.

This module provides flexible output collection and statistical analysis
capabilities for pywatershed models. It supports:

1. **Monthly accumulations**: Accumulate variable values over monthly periods
   for all spatial units (HRUs or segments).

2. **Point of Interest (POI) data**: Collect full time series data for specific
   stream segments (e.g., at gage locations) and calculate various statistics
   including temporal aggregations and resampling.

3. **HRU subset data**: Collect full time series data for specific HRUs and
   calculate statistics with flexible temporal grouping and resampling.

The module uses xarray DataArrays for efficient handling of multi-dimensional
time series data with proper coordinate systems and metadata.

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
- Extend monthly stats to timeseries arrays in PRMSSolarGeometry and
  PRMSAtmosphere
- Annual (or other periods) extremes (e.g., date of peak SWE)
- Show how to do numpy-based functions
- Fixed window stats on all spatial units (e.g., monthly standard deviations)
- Rolling window stats on all spatial units
- FlowGraph integration

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

from . import meta

if TYPE_CHECKING:
    from .control import Control
    from .model import Model

spatial_dim_to_coord_name = {"nhru": "nhm_id", "nsegment": "nhm_seg"}


def mean(da, dim=None, *, skipna=None, keep_attrs=None, **kwargs):
    """Calculate mean along specified dimension.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    dim : str, optional
        Dimension(s) over which to calculate mean
    skipna : bool, optional
        Whether to skip NaN values
    keep_attrs : bool, optional
        Whether to preserve attributes
    **kwargs
        Additional keyword arguments passed to xarray mean

    Returns
    -------
    xr.DataArray
        Mean values
    """
    return da.mean(dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)


def std(da, dim=None, *, skipna=None, keep_attrs=None, **kwargs):
    """Calculate standard deviation along specified dimension.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    dim : str, optional
        Dimension(s) over which to calculate standard deviation
    skipna : bool, optional
        Whether to skip NaN values
    keep_attrs : bool, optional
        Whether to preserve attributes
    **kwargs
        Additional keyword arguments passed to xarray std

    Returns
    -------
    xr.DataArray
        Standard deviation values
    """
    return da.std(dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)


def median(da, dim=None, *, skipna=None, keep_attrs=None, **kwargs):
    """Calculate median along specified dimension.

    Parameters
    ----------
    da : xr.DataArray
        Input data array
    dim : str, optional
        Dimension(s) over which to calculate median
    skipna : bool, optional
        Whether to skip NaN values
    keep_attrs : bool, optional
        Whether to preserve attributes
    **kwargs
        Additional keyword arguments passed to xarray median

    Returns
    -------
    xr.DataArray
        Median values
    """
    return da.median(dim=dim, skipna=skipna, keep_attrs=keep_attrs, **kwargs)


# TODO: need to differentiate between accumulated stats and full-time stats?
full_time_stat_functions = {
    "mean": mean,
    "median": median,
    "std": std,
}


class CustomOutput:
    """Flexible output collection and statistical analysis for models.

    This class provides three main types of output collection:

    1. **Monthly accumulations**: Variables are accumulated over each calendar
       month for all spatial units (HRUs or segments).

    2. **Point of Interest (POI) data**: Full time series data is collected
       for specific stream segments (typically at gage locations) with
       optional statistical aggregations using groupby and resample operations.

    3. **HRU subset data**: Full time series data is collected for specific
       HRUs with optional statistical aggregations.

    The output object is used during model execution and must be finalized
    before accessing calculated statistics. All data is returned as xarray
    DataArrays with proper coordinates and metadata.

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
        poi_stats: list[str | Callable] | None = None,
        poi_stats_groupby: dict | None = None,
        poi_stats_resample: dict | None = None,
        hru_sub_var_list: list | None = None,
        hru_sub_ids: list | None = None,
        hru_sub_stats: list[str | Callable] | None = None,
        hru_sub_stats_groupby: dict | None = None,
        hru_sub_stats_resample: dict | None = None,
    ):
        """Initialize CustomOutput instance.

        Sets up data structures for collecting monthly accumulations, POI
        data, and HRU subset data according to the specified configuration.
        """
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
        self._poi_stats_groupby = poi_stats_groupby
        self._poi_stats_resample = poi_stats_resample

        self._hru_sub_var_list = hru_sub_var_list
        self._hru_sub_ids = hru_sub_ids
        self._hru_sub_stats = hru_sub_stats
        self._hru_sub_stats_groupby = hru_sub_stats_groupby
        self._hru_sub_stats_resample = hru_sub_stats_resample

        self._current_time = self._control.init_time.copy()
        self._time_step = self._control.time_step.copy()

        self._init_monthly()
        self._init_poi_sub()

        # Validate configuration
        self._validate_stat_configs()

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
            return self._poi_stats
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
            return self._hru_sub_stats
        else:
            warnings.warn(
                "hru_sub_stats is only available after finalization. "
                "Call output.finalize() or model.run(finalize=True)."
            )
            return None

    # ==== Validation methods =====================
    def _validate_stat_configs(self):
        """Validate that resample/groupby dict keys match statistic names.

        This catches configuration errors early rather than at runtime.
        """
        # Validate POI stats configuration
        if self._poi_stats is not None:
            stat_names = set()
            for ss in self._poi_stats:
                if isinstance(ss, str):
                    stat_names.add(ss)
                else:
                    stat_names.add(ss.__name__)

            if self._poi_stats_groupby is not None:
                invalid_keys = set(self._poi_stats_groupby.keys()) - stat_names
                if invalid_keys:
                    raise ValueError(
                        f"poi_stats_groupby contains keys {invalid_keys} "
                        f"that are not in poi_stats {stat_names}"
                    )

            if self._poi_stats_resample is not None:
                invalid_keys = (
                    set(self._poi_stats_resample.keys()) - stat_names
                )
                if invalid_keys:
                    raise ValueError(
                        f"poi_stats_resample contains keys {invalid_keys} "
                        f"that are not in poi_stats {stat_names}"
                    )

        # Validate HRU subset stats configuration
        if self._hru_sub_stats is not None:
            stat_names = set()
            for ss in self._hru_sub_stats:
                if isinstance(ss, str):
                    stat_names.add(ss)
                else:
                    stat_names.add(ss.__name__)

            if self._hru_sub_stats_groupby is not None:
                invalid_keys = (
                    set(self._hru_sub_stats_groupby.keys()) - stat_names
                )
                if invalid_keys:
                    raise ValueError(
                        f"hru_sub_stats_groupby contains keys {invalid_keys} "
                        f"that are not in hru_sub_stats {stat_names}"
                    )

            if self._hru_sub_stats_resample is not None:
                invalid_keys = (
                    set(self._hru_sub_stats_resample.keys()) - stat_names
                )
                if invalid_keys:
                    raise ValueError(
                        f"hru_sub_stats_resample contains keys {invalid_keys} "
                        f"that are not in hru_sub_stats {stat_names}"
                    )

    @staticmethod
    def get_statistic_name(
        variable: str, statistic: str, temporal_op: str | None = None
    ) -> str:
        """Construct a statistic name following the naming convention.

        The naming convention is: {variable}_{statistic}_{temporal_op}
        where temporal_op is optional for full-time statistics.

        Parameters
        ----------
        variable : str
            Variable name (e.g., "seg_outflow")
        statistic : str
            Statistic name (e.g., "mean", "median")
        temporal_op : str, optional
            Temporal operation like "month", "1MS", "5D" for grouped or
            resampled statistics

        Returns
        -------
        str
            The constructed statistic name

        Examples
        --------
        >>> CustomOutput.get_statistic_name("seg_outflow", "mean")
        'seg_outflow_mean'
        >>> CustomOutput.get_statistic_name("seg_outflow", "median", "month")
        'seg_outflow_median_month'
        >>> CustomOutput.get_statistic_name("seg_outflow", "max", "5D")
        'seg_outflow_max_5D'
        """
        if temporal_op is None:
            return f"{variable}_{statistic}"
        else:
            return f"{variable}_{statistic}_{temporal_op}"

    # ==== Monthly accumulation section =====================
    def _init_monthly(self):
        """Initialize monthly accumulation data structures.

        Creates time coordinates, maps variables to processes, and declares
        storage arrays for monthly accumulations if configured.
        """
        if self._monthly_accum_var_list is None:
            self._time_months = None
            self._n_days_per_month = None
            return None

        self._solve_monthly_time()
        self._map_monthly_vars_procs()
        self._declare_monthly_arrays()

    def _solve_monthly_time(self):
        """Create monthly time coordinate and initialize day counter.

        Generates a monthly time coordinate array spanning the model
        simulation period and initializes an array to track the number of
        days in each month.
        """
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
        """Map monthly accumulation variables to their source processes.

        Searches through model processes to find which process provides each
        monthly accumulation variable.

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

        if not set(self._monthly_vars_procs.keys()) == set(
            self._monthly_accum_var_list
        ):
            raise ValueError(
                "Not all monthly accumulation variables were found among the "
                "model processes."
            )

    def _declare_monthly_arrays(self):
        """Declare and initialize xarray DataArrays for monthly accumulations.

        Creates zero-initialized arrays with proper dimensions
        (month x spatial_unit), coordinates, and metadata for each monthly
        accumulation variable.
        """
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

    def _get_month_index(self) -> None:
        """Determine the current month index for accumulation.

        Finds which month index in the monthly time coordinate corresponds
        to the current simulation time.
        """
        current_month = self._current_time.astype("datetime64[M]")
        self._current_month_index = np.where(
            self._time_months == current_month
        )[0][0]

    def _accumulate_monthly_values(self) -> None:
        """Accumulate current timestep values into monthly arrays.

        Adds current timestep values to the appropriate monthly accumulation
        arrays and increments the day counter for the current month.
        """
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
        """Initialize POI and HRU subset data structures.

        Creates time coordinates, resolves statistics, maps variables to
        processes, and declares storage arrays for POI and HRU subset data
        if configured.
        """
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

    def _solve_time(self):
        """Create daily time coordinate for full time series data.

        Generates a daily time coordinate array spanning the entire model
        simulation period for POI and HRU subset data collection.
        """
        import pandas as pd

        ctl = self._control
        self._time = pd.date_range(
            start=ctl.start_time, end=ctl.end_time, freq="D"
        ).values.astype("datetime64[D]")

    def _solve_poi_stat_list(self):
        """Resolve POI statistics list to callable functions.

        Converts string statistic names to their corresponding function
        references and stores custom callable functions with their names.
        """
        self._poi_stat_funcs = {}
        if self._poi_stats is None:
            return
        for ss in self._poi_stats:
            if isinstance(ss, str):
                self._poi_stat_funcs[ss] = full_time_stat_functions[ss]
            else:
                if not callable(ss):
                    raise ValueError(
                        "poi_stats must contain strings or functions."
                    )
                self._poi_stat_funcs[ss.__name__] = ss

    def _solve_hru_sub_stat_list(self):
        """Resolve HRU subset statistics list to callable functions.

        Converts string statistic names to their corresponding function
        references and stores custom callable functions with their names.
        """
        self._hru_sub_stat_funcs = {}
        if self._hru_sub_stats is None:
            return
        for ss in self._hru_sub_stats:
            if isinstance(ss, str):
                self._hru_sub_stat_funcs[ss] = full_time_stat_functions[ss]
            else:
                if not callable(ss):
                    raise ValueError(
                        "hru_sub_stats must contain strings or functions."
                    )
                self._hru_sub_stat_funcs[ss.__name__] = ss

    def _map_poi_vars_procs(self):
        """Map POI variables to processes and resolve POI indices.

        Searches through model processes to find which provides each POI
        variable, verifies variables have nsegment dimension, and calculates
        array indices corresponding to the requested POI segments.
        """
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
                    if vv_dims != "nsegment":
                        raise ValueError(
                            f"Variable '{vv}' does not have dimension "
                            "'nsegment'."
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

    def _map_hru_sub_vars_procs(self):
        """Map HRU subset variables to processes and resolve HRU indices.

        Searches through model processes to find which provides each HRU
        subset variable, verifies variables have nhru dimension, and
        calculates array indices corresponding to the requested HRU IDs.
        """
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

    def _build_poi_hru_sub_iteration_lists(self):
        """Build and cache iteration lists for POI and HRU subset processing.

        These lists are built once and cached to avoid rebuilding them every
        timestep. Different methods need different subsets of the data.
        """
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

        # For _calculate_poi_stats (called once at finalization)
        # Don't cache the stats dicts themselves, they're created fresh
        self._poi_hru_sub_stats_list = []
        if self._poi_var_list is not None:
            self._poi_hru_sub_stats_list.append(
                (
                    "poi",  # marker to identify which stats dict to use
                    self._poi_arrays,
                    self._poi_stat_funcs,
                    self._poi_vars_procs,
                    self._poi_stats_groupby,
                    self._poi_stats_resample,
                )
            )
        if self._hru_sub_var_list is not None:
            self._poi_hru_sub_stats_list.append(
                (
                    "hru_sub",  # marker to identify which stats dict to use
                    self._hru_sub_arrays,
                    self._hru_sub_stat_funcs,
                    self._hru_sub_vars_procs,
                    self._hru_sub_stats_groupby,
                    self._hru_sub_stats_resample,
                )
            )

    def _declare_poi_hru_sub_arrays(self):
        """Declare and initialize xarray DataArrays for POI and HRU subset.

        Creates NaN-initialized arrays with proper dimensions
        (time x spatial_unit), coordinates, and metadata for each POI and
        HRU subset variable.

        Note: self._poi_arrays and self._hru_sub_arrays are already initialized
        as empty dicts in _init_poi_sub before this method is called.
        """
        # Use cached iteration list instead of rebuilding
        for arrays, var_list, vars_procs, inds in self._poi_hru_sub_data_list:
            for vv in var_list:
                proc_name = vars_procs[vv]
                proc = self._model.processes[proc_name]
                var_meta = meta.find_variables(vv)[vv]
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
        """Add current timestep data to POI and HRU subset arrays.

        Extracts current values from model processes for the specified POI
        segments and HRU subset locations and stores them in the appropriate
        time index of the output arrays.
        """
        if not self._poi_hru_sub_data_list:
            return

        time_ind = self._control.itime_step
        for arrays, var_list, vars_procs, inds in self._poi_hru_sub_data_list:
            for vv in var_list:
                proc_name = vars_procs[vv]
                arrays[vv][time_ind, :] = self._model.processes[proc_name][vv][
                    inds
                ]

    def _calculate_poi_stats(self):
        """Calculate all requested statistics for POI and HRU subset data.

        Applies statistic functions to the collected time series data, with
        optional temporal grouping (e.g., by month) and resampling (e.g., to
        monthly or other frequencies). Statistics are stored with descriptive
        names following the pattern: {variable}_{statistic}_{temporal_op}.
        """
        self._poi_stats = {}
        self._hru_sub_stats = {}

        for (
            stats_type,
            arrays,
            stat_funcs,
            vars_procs,
            stats_groupby,
            stats_resample,
        ) in self._poi_hru_sub_stats_list:
            # Get the appropriate stats dict
            stats = (
                self._poi_stats if stats_type == "poi" else self._hru_sub_stats
            )
            for vv in arrays:
                for stat_name, stat_func in stat_funcs.items():
                    calc_full_time = True

                    if (
                        stats_groupby is not None
                        and stat_name in stats_groupby.keys()
                    ):
                        group = stats_groupby[stat_name]
                        stat_key = self.get_statistic_name(
                            vv, stat_name, group
                        )
                        stats[stat_key] = stat_func(
                            arrays[vv].groupby(f"time.{group}"),
                            dim="time",
                        )
                        calc_full_time = False

                    if (
                        stats_resample is not None
                        and stat_name in stats_resample.keys()
                    ):
                        resample = stats_resample[stat_name]
                        stat_key = self.get_statistic_name(
                            vv, stat_name, resample
                        )
                        stats[stat_key] = stat_func(
                            arrays[vv].resample(time=resample),
                            dim="time",
                        )
                        calc_full_time = False

                    if calc_full_time:
                        stat_key = self.get_statistic_name(vv, stat_name)
                        stats[stat_key] = stat_func(arrays[vv], dim="time")

    # ==== General methods ================
    def calculate(self) -> None:
        """Collect data for the current timestep.

        This method is called automatically during model execution to accumulate
        monthly values and collect POI/HRU subset data at each timestep.

        Raises
        ------
        ValueError
            If the control time does not match expected timestep progression

        Notes
        -----
        The control.advance() must be called before this calculate() method.
        This is handled automatically during model.run().
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

    def finalize(self):
        """Finalize output collection and calculate all statistics.

        This method marks the output as finalized and triggers calculation of
        all requested statistics. After finalization, the properties
        monthly_accumulations, poi_arrays, hru_sub_arrays, poi_stats, and
        hru_sub_stats become accessible.

        Notes
        -----
        This method is typically called automatically by model.run(finalize=True)
        and should not need to be called directly by users.
        """
        self._finalized = True
        self._calculate_poi_stats()

    def to_netcdf(self, output_dir: pl.Path):
        """Write output data to netCDF files.

        Parameters
        ----------
        output_dir : pathlib.Path
            Directory where netCDF files should be written

        Raises
        ------
        NotImplementedError
            This functionality is not yet available

        Notes
        -----
        Output can only be written once the Output object is finalized.
        """
        if not self._finalized:
            warnings.warn(
                "Output can only be written once the Output object is finalized"
            )
            return

        # self._monthly_to_netcdf(self)
        # self._poi_to_netcdf(self)

        raise NotImplementedError("YET.")
