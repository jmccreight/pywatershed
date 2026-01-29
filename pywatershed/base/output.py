"""Output collection and statistical analysis for pywatershed models.

Supports PRMS processes and FlowGraph. Collects three types of output:

1. Monthly accumulations - All spatial units (HRUs, segments, nodes)
2. NOI (Nodes of Interest) - Time series and stats at specific network nodes
3. HOI (HRUs of Interest) - Time series and stats for specific HRUs

Nomenclature
------------
This module adopts NOI (Nodes of Interest) and HOI (HRUs of Interest) to
denote subsets of the model's spatial discretization, distinguishing them
from NOI (Points of Interest) used in PRMS to denote real-world locations
like gage stations. While NOIs represent physical monitoring locations,
NOIs and HOIs represent subsets of the model grid for focused output
collection and analysis.

- NOI: Nodes (river segments in PRMSChannel, segments + features in FlowGraph)
- HOI: HRUs (Hydrologic Response Units)
- NOI: Real-world monitoring locations (PRMS parameter noi_gage_segment)

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
...     noi_var_list=["seg_outflow"],
...     noi_ids=[12345, 67890],
...     noi_stats={mean: ["seg_outflow"], max_flow: ["seg_outflow"]},
...     hoi_var_list=["hru_actet"],
...     hoi_ids=[1, 2, 3],
...     hoi_stats={mean: ["hru_actet"]},
... )
>>> model.run(finalize=True, output=output)
>>>
>>> # Access hierarchically
>>> output.noi_stats["seg_outflow"]["mean"]
>>> output.hoi_stats["hru_actet"]["mean"]

Notes
-----
- Statistics use dict[function: var_list] pattern
- Results stored hierarchically: output.noi_stats[variable][statistic]
- Each result has metadata: variable, statistic, period_of_record
- IDs can be list (same for all vars) or dict (per-variable)
- Must finalize before accessing statistics
"""

import pathlib as pl
import warnings
from typing import TYPE_CHECKING, Callable, Literal

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
    noi_var_list : list[str], optional
        Variables to collect at nodes of interest. Required if noi_ids is a list.
        Must NOT be provided if dict (use dict keys).
    noi_ids : list[int] or dict[str, list[int]] or list[tuple] or dict[str, list[tuple]], optional
        Node IDs for NOIs. Supports two ID types and two modes:

        ID types:
        - Simple IDs: Integer nhm_seg values (e.g., [12345, 67890])
        - FlowGraph tuples: (node_maker_name, node_maker_id) for FlowGraph nodes
          (e.g., [("prms_channel", 12345), ("starfit", 0)])

        Modes:
        - List mode: Same IDs for all vars (requires noi_var_list)
        - Dict mode: {var_name: [ids]} per-variable IDs (don't provide noi_var_list)
    noi_stats : dict[Callable, list[str]], optional
        Statistics for NOIs: {function: [var1, var2, ...]}
    hoi_var_list : list[str], optional
        Variables to collect for HRUs of interest. Required if hoi_ids is a list.
        Must NOT be provided if hoi_ids is dict (use dict keys).
    hoi_ids : list[int] or dict[str, list[int]], optional
        HRU IDs (nhm_id values). Two modes:
        - List mode: Same IDs for all vars (requires hoi_var_list)
        - Dict mode: {var_name: [ids]} per-variable IDs (don't provide hoi_var_list)
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
    noi_arrays : dict[str, xr.DataArray]
        NOI time series, available after finalization
    hoi_arrays : dict[str, xr.DataArray]
        HOI time series, available after finalization
    noi_stats : dict[str, dict[str, xr.DataArray]]
        NOI statistics: noi_stats[variable][statistic]
    hoi_stats : dict[str, dict[str, xr.DataArray]]
        HOI statistics: hoi_stats[variable][statistic]

    Examples
    --------
    >>> from pywatershed.analysis.time_stats import mean
    >>> def max_flow(da):
    ...     return da.max(dim="time")
    ...
    >>>
    >>> # List mode: same IDs for all variables
    >>> output = pws.base.Output(
    ...     control=control,
    ...     model=model,
    ...     noi_var_list=["seg_outflow"],  # Required with list mode
    ...     noi_ids=[12345, 67890],
    ...     noi_stats={mean: ["seg_outflow"]},
    ...     hoi_var_list=["hru_actet"],  # Required with list mode
    ...     hoi_ids=[1, 2, 3],
    ...     hoi_stats={mean: ["hru_actet"]},
    ... )
    >>>
    >>> # Dict mode: per-variable IDs
    >>> output = pws.base.Output(
    ...     control=control,
    ...     model=model,
    ...     noi_ids={  # Dict keys define variables
    ...         "seg_outflow": [12345, 67890],
    ...         "seg_upstream_inflow": [12345],
    ...     },
    ...     noi_stats={mean: ["seg_outflow"]},  # Not all vars need stats
    ...     hoi_ids={  # Dict keys define variables
    ...         "hru_actet": [1, 2],
    ...         "pkwater_equiv": [3, 4, 5],
    ...     },
    ...     hoi_stats={mean: ["hru_actet"]},
    ... )
    >>>
    >>> # FlowGraph mode: tuple IDs for nodes
    >>> output = pws.base.Output(
    ...     control=control,
    ...     model=flowgraph_model,
    ...     noi_ids={
    ...         "node_outflows": [("prms_channel", 12345), ("starfit", 0)],
    ...         "node_storages": [("starfit", 0)],
    ...     },
    ...     noi_stats={mean: ["node_outflows"]},
    ... )
    >>> model.run(finalize=True, output=output)
    >>>
    >>> # Access hierarchically
    >>> output.noi_stats["seg_outflow"]["mean"]
    >>> output.hoi_stats["hru_actet"]["mean"]
    """

    def __init__(
        self,
        control: "Control",
        model: "Model",
        monthly_accum_var_list: list | None = None,
        noi_var_list: list | None = None,
        noi_ids: list | dict | None = None,
        noi_stats: dict[Callable, list[str]] | None = None,
        hoi_var_list: list | None = None,
        hoi_ids: list | dict | None = None,
        hoi_stats: dict[Callable, list[str]] | None = None,
        netcdf_output_action: Literal["allow", "warn", "error"] = "error",
    ):
        """Initialize Output and set up data collection."""

        self._control = control
        self._netcdf_output_action = netcdf_output_action
        self._take_netcdf_output_action()

        self._finalized = False
        self._model = model

        self._monthly_accum_var_list = monthly_accum_var_list

        # Process NOI IDs (can be list or dict)
        self._noi_var_list, self._noi_ids = self._process_noi_ids(
            noi_var_list, noi_ids
        )
        self._noi_stats = noi_stats

        # Process HOI IDs (can be list or dict)
        self._hoi_var_list, self._hoi_ids = self._process_hoi_ids(
            hoi_var_list, hoi_ids
        )
        self._hoi_stats = hoi_stats

        self._current_time = self._control.init_time.copy()
        self._time_step = self._control.time_step.copy()

        self._init_monthly()
        self._init_noi_sub()

        return None

    def _take_netcdf_output_action(self):
        if (
            "netcdf_output_dir" in self._control.options.keys()
            and self._netcdf_output_action != "allow"
        ):
            msg = (
                "control.options['netcdf_output_dir'] is defined in "
                "addition to Output object being intitalized with argument "
                f"{self._netcdf_output_action=}."
            )
            if self._netcdf_output_action == "warn":
                warnings.warn(msg, UserWarning)
            else:
                raise ValueError(msg)

    # ==== ID Processing Methods =========================
    def _process_noi_ids(
        self, var_list, ids
    ) -> tuple[list | None, list | dict | None]:
        """Process NOI IDs - handle list or dict for per-variable IDs.

        Two modes:
        - List mode: var_list required, same IDs for all variables
        - Dict mode: dict keys define variables, var_list must be None
        """
        if var_list is None and ids is None:
            return None, None

        if ids is None:
            raise ValueError(
                "noi_ids must be passed when noi variables are requested."
            )

        # Handle dict mode
        if isinstance(ids, dict):
            if var_list is not None:
                raise ValueError(
                    "noi_var_list should not be provided when noi_ids is a dict. "
                    "Use dict keys to specify variables."
                )
            var_list = list(ids.keys())
            return var_list, ids

        # List mode - existing behavior
        if var_list is None:
            raise ValueError("noi_var_list required when noi_ids is a list")
        return var_list, ids

    def _process_hoi_ids(
        self, var_list, ids
    ) -> tuple[list | None, list | dict | None]:
        """Process HOI IDs - handle list or dict for per-variable IDs.

        Two modes:
        - List mode: var_list required, same IDs for all variables
        - Dict mode: dict keys define variables, var_list must be None
        """
        if var_list is None and ids is None:
            return None, None

        # Handle dict mode
        if isinstance(ids, dict):
            if var_list is not None:
                raise ValueError(
                    "hoi_var_list should not be provided when hoi_ids is a dict. "
                    "Use dict keys to specify variables."
                )
            var_list = list(ids.keys())
            return var_list, ids

        # List mode - existing behavior
        if ids is not None and var_list is None:
            raise ValueError("hoi_var_list required when hoi_ids is a list")
        return var_list, ids

    # ==== Properties =========================
    @property
    def time(self) -> np.ndarray | None:
        """Daily time coordinate for NOI/HOI data."""
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
    def noi_arrays(self) -> dict[str, xr.DataArray] | None:
        """NOI time series (available after finalization)."""
        if self._finalized:
            return self._noi_arrays
        else:
            warnings.warn(
                "noi_arrays is only available after finalization. "
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
    def noi_stats(self) -> dict[str, dict[str, xr.DataArray]] | None:
        """NOI statistics: noi_stats[variable][statistic] (after finalization)."""
        if self._finalized:
            return self._noi_stats_results
        else:
            warnings.warn(
                "noi_stats is only available after finalization. "
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

    # ==== NOI + HRU SUB section =====================
    def _init_noi_sub(self) -> None:
        """Initialize NOI and HRU subset data structures."""
        if not self._noi_var_list and not self._hoi_var_list:
            # Initialize empty lists so other methods don't error
            self._noi_hoi_data_list = []
            self._noi_hoi_stats_list = []
            return None

        self._solve_time()
        self._solve_noi_stat_list()
        self._solve_hoi_stat_list()
        self._map_noi_vars_procs()
        self._map_hoi_vars_procs()
        # Initialize empty dicts first
        self._noi_arrays = {}
        self._hoi_arrays = {}
        # Build iteration lists (references the empty dicts)
        self._build_noi_hoi_iteration_lists()
        # Now populate the arrays
        self._declare_noi_hoi_arrays()

    def _solve_time(self) -> None:
        """Create daily time coordinate for full time series data."""
        import pandas as pd

        ctl = self._control
        self._time = pd.date_range(
            start=ctl.start_time, end=ctl.end_time, freq="D"
        ).values.astype("datetime64[D]")

    def _solve_noi_stat_list(self) -> None:
        """Build NOI statistics dict from function: var_list mapping."""
        self._noi_stat_func_vars = {}
        if self._noi_stats is None:
            return
        for func, var_list in self._noi_stats.items():
            if not callable(func):
                raise ValueError("noi_stats keys must be callable functions.")
            self._noi_stat_func_vars[func] = var_list

    def _solve_hoi_stat_list(self) -> None:
        """Build HRU subset statistics dict from function: var_list mapping."""
        self._hoi_stat_func_vars = {}
        if self._hoi_stats is None:
            return
        for func, var_list in self._hoi_stats.items():
            if not callable(func):
                raise ValueError("hoi_stats keys must be callable functions.")
            self._hoi_stat_func_vars[func] = var_list

    @staticmethod
    def _solve_flowgraph_inds(tup_list, params, check=True):
        """Resolve FlowGraph node indices from (node_maker_name, node_maker_id) tuples.

        For FlowGraph nodes, IDs are specified as 2-tuples:
        (node_maker_name, node_maker_id) rather than simple integer IDs.

        Parameters
        ----------
        tup_list : list of tuple
            List of (node_maker_name, node_maker_id) tuples
        params : dict
            Process parameters containing node_maker_name and node_maker_id arrays
        check : bool, optional
            Whether to validate results (default True)

        Returns
        -------
        list of int
            Indices matching the requested tuples
        """
        flowgraph_inds = []
        for tup in tup_list:
            matches = np.where(
                (params["node_maker_name"] == tup[0])
                & (params["node_maker_id"] == tup[1])
            )[0]
            if len(matches) == 0:
                raise ValueError(
                    f"FlowGraph node not found: "
                    f"node_maker_name='{tup[0]}', node_maker_id={tup[1]}"
                )
            flowgraph_inds += matches.tolist()

        if check:
            # Validate: check that found indices match requested tuples
            found_names = params["node_maker_name"][flowgraph_inds].tolist()
            expected_names = [tt for tt, _ in tup_list]
            if found_names != expected_names:
                raise ValueError(
                    f"FlowGraph index resolution failed: "
                    f"node_maker_name mismatch. Expected {expected_names}, "
                    f"got {found_names}"
                )

            found_ids = params["node_maker_id"][flowgraph_inds].tolist()
            expected_ids = [tt for _, tt in tup_list]
            if found_ids != expected_ids:
                raise ValueError(
                    f"FlowGraph index resolution failed: "
                    f"node_maker_id mismatch. Expected {expected_ids}, "
                    f"got {found_ids}"
                )

        return flowgraph_inds

    def _map_noi_vars_procs(self) -> None:
        """Map NOI variables to processes and resolve indices."""
        if self._noi_var_list is None:
            return
        self._noi_vars_procs = {}

        # Map variables to processes
        for vv in self._noi_var_list:
            for pp in self._model.processes.keys():
                proc = self._model.processes[pp]
                proc_vars = proc.get_variables()
                if hasattr(proc, "_addtl_output_vars"):
                    proc_vars += proc._addtl_output_vars
                if vv in proc_vars:
                    self._noi_vars_procs[vv] = pp
                    vv_dims = proc.meta[vv]["dims"][0]
                    if vv_dims != "nsegment" and vv_dims != "nnodes":
                        raise ValueError(
                            f"Variable '{vv}' does not have dimension "
                            "'nsegment' nor 'nnodes'."
                        )

        if not self._noi_vars_procs:
            return

        # Handle dict mode (per-variable IDs) or list mode (same IDs for all)
        # IDs can be simple integers (nhm_seg) or tuples for FlowGraph nodes
        if isinstance(self._noi_ids, dict):
            # Dict mode: per-variable IDs
            self._noi_inds = {}
            for vv in self._noi_var_list:
                proc_name = self._noi_vars_procs[vv]
                proc = self._model.processes[proc_name]
                if not isinstance(self._noi_ids[vv][0], tuple):
                    self._noi_inds[vv] = np.where(
                        np.isin(
                            proc._params.parameters["nhm_seg"],
                            self._noi_ids[vv],
                        )
                    )[0]
                else:
                    tup_list = self._noi_ids[vv]
                    self._noi_inds[vv] = self._solve_flowgraph_inds(
                        tup_list, proc._params.parameters
                    )

        else:
            # List mode: same IDs for all variables
            proc_name = list(self._noi_vars_procs.values())[0]
            proc = self._model.processes[proc_name]
            if not isinstance(self._noi_ids, tuple):
                proc_coord = spatial_dim_to_coord_name[
                    list(proc._params.dims.keys())[0]
                ]
                self._noi_inds = np.where(
                    np.isin(
                        proc._params.parameters["nhm_seg"],
                        self._noi_ids,
                    )
                )[0]
            else:
                tup_list = self._noi_ids
                self._noi_inds = self._solve_flowgraph_inds(
                    tup_list, proc._params.parameters
                )

    def _map_hoi_vars_procs(self) -> None:
        """Map HRU subset variables to processes and resolve indices."""
        if self._hoi_var_list is None:
            return
        self._hoi_vars_procs = {}

        # Map variables to processes
        for vv in self._hoi_var_list:
            for pp in self._model.processes.keys():
                proc_vars = self._model.processes[pp].get_variables()
                if vv in proc_vars:
                    self._hoi_vars_procs[vv] = pp
                    vv_dims = meta.find_variables(vv)[vv]["dims"][0]
                    if vv_dims != "nhru":
                        raise ValueError(
                            f"Variable '{vv}' does not have dimension 'nhru'."
                        )

        if not self._hoi_vars_procs:
            return

        # Handle dict mode (per-variable IDs) or list mode (same IDs for all)
        if isinstance(self._hoi_ids, dict):
            # Dict mode: per-variable IDs
            self._hoi_inds = {}
            for vv in self._hoi_var_list:
                proc_name = self._hoi_vars_procs[vv]
                self._hoi_inds[vv] = np.where(
                    np.isin(
                        self._model.processes[proc_name]._params.parameters[
                            "nhm_id"
                        ],
                        self._hoi_ids[vv],
                    )
                )[0]
        else:
            # List mode: same IDs for all variables
            proc_name = list(self._hoi_vars_procs.values())[0]
            self._hoi_inds = np.where(
                np.isin(
                    self._model.processes[proc_name]._params.parameters[
                        "nhm_id"
                    ],
                    self._hoi_ids,
                )
            )[0]

    def _build_noi_hoi_iteration_lists(self) -> None:
        """Build and cache iteration lists to avoid rebuilding each timestep."""
        # For _add_noi_hoi_data (called every timestep)
        self._noi_hoi_data_list = []
        if self._noi_var_list is not None:
            self._noi_hoi_data_list.append(
                (
                    self._noi_arrays,
                    self._noi_var_list,
                    self._noi_vars_procs,
                    self._noi_inds,
                )
            )
        if self._hoi_var_list is not None:
            self._noi_hoi_data_list.append(
                (
                    self._hoi_arrays,
                    self._hoi_var_list,
                    self._hoi_vars_procs,
                    self._hoi_inds,
                )
            )

        # For _calculate_noi_hoi_stats (called once at finalization)
        self._noi_hoi_stats_list = []
        if self._noi_stats is not None:
            self._noi_hoi_stats_list.append(
                (
                    "noi",  # marker to identify which stats dict to use
                    self._noi_arrays,
                    self._noi_stat_func_vars,
                )
            )
        if self._hoi_stats is not None:
            self._noi_hoi_stats_list.append(
                (
                    "hoi",  # marker to identify which stats dict to use
                    self._hoi_arrays,
                    self._hoi_stat_func_vars,
                )
            )

    def _declare_noi_hoi_arrays(self) -> None:
        """Declare xarray DataArrays for NOI and HRU subset variables."""
        # Use cached iteration list instead of rebuilding
        for arrays, var_list, vars_procs, inds in self._noi_hoi_data_list:
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

                # Handle dict mode (per-variable indices) or list mode
                var_inds = inds[vv] if isinstance(inds, dict) else inds

                spatial_dim_len = proc[vv][var_inds].shape[0]
                spatial_dim_name = var_meta["dims"][0]
                spatial_coord_name = spatial_dim_to_coord_name[
                    spatial_dim_name
                ]
                spatial_coord = proc._params.coords[spatial_coord_name][
                    var_inds
                ]
                coords = {
                    "time": self._time,
                    spatial_coord_name: (
                        [spatial_dim_name],
                        spatial_coord,
                    ),
                }
                if spatial_coord_name == "node_coord":
                    coords["node_maker_name"] = (
                        [spatial_dim_name],
                        proc._params.parameters["node_maker_name"][var_inds],
                    )
                    coords["node_maker_id"] = (
                        [spatial_dim_name],
                        proc._params.parameters["node_maker_id"][var_inds],
                    )
                # <
                new_shape = (len(self._time), spatial_dim_len)
                arrays[vv] = xr.DataArray(
                    data=np.full(new_shape, np.nan, dtype=var_meta["type"]),
                    dims=["time", spatial_dim_name],
                    coords=(coords),
                    attrs=dict(
                        description=var_meta["desc"],
                        units=var_meta["units"],
                    ),
                )

    def _add_noi_hoi_data(self) -> None:
        """Add current timestep data to NOI and HRU subset arrays."""
        if not self._noi_hoi_data_list:
            return

        time_ind = self._control.itime_step
        for arrays, var_list, vars_procs, inds in self._noi_hoi_data_list:
            for vv in var_list:
                proc_name = vars_procs[vv]
                # Handle dict mode (per-variable indices) or list mode
                var_inds = inds[vv] if isinstance(inds, dict) else inds
                arrays[vv][time_ind, :] = self._model.processes[proc_name][vv][
                    var_inds
                ]

    def _calculate_noi_hoi_stats(self) -> None:
        """Calculate NOI/HOI statistics, store as stats[var][func_name]."""
        self._noi_stats_results = {}
        self._hoi_stats_results = {}

        for stats_type, arrays, stat_func_vars in self._noi_hoi_stats_list:
            # Get the appropriate stats dict
            stats = (
                self._noi_stats_results
                if stats_type == "noi"
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
        self._add_noi_hoi_data()

    def finalize(self) -> None:
        """Finalize and calculate statistics (called by model.run())."""
        self._finalized = True
        self._calculate_noi_hoi_stats()

    def to_netcdf(self, output_dir: pl.Path) -> None:
        """Write output to netCDF files (not yet implemented)."""
        if not self._finalized:
            warnings.warn(
                "Output can only be written once the Output object is finalized"
            )
            return

        # self._monthly_to_netcdf(self)
        # self._noi_to_netcdf(self)

        raise NotImplementedError("YET.")
