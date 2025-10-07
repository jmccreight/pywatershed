import pathlib as pl
from typing import Literal, Tuple, Union
from warnings import warn

import networkx as nx
import numpy as np

try:
    from numba import float64, int32, jitclass

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy types for when numba is not available
    float64 = None
    int32 = None
    jitclass = lambda spec: lambda cls: cls

from ..base.adapter import adaptable
from ..base.conservative_process import ConservativeProcess
from ..base.control import Control
from ..constants import SegmentType, nan, zero
from ..parameters import Parameters

# Numba type specification for Calc class
calc_spec = [
    ("nhru", int32),
    ("nsegment", int32),
    ("hru_segment", int32[:]),
    ("segment_order", int32[:]),
    ("tosegment", int32[:]),
    ("tsi", int32[:]),
    ("ts", float64[:]),
    ("c0", float64[:]),
    ("c1", float64[:]),
    ("c2", float64[:]),
    ("outflow_mask", int32[:]),
]


@jitclass(calc_spec)
class Calc:
    """
    High-performance Numba JIT-compiled class for channel flow calculations.

    This class contains the core computational logic from the _calculate method,
    extracted from PRMSChannel for better performance through Numba compilation.

    Usage Example:
    -------------
    # The Calc instance is created automatically when using calc_method='jitclass'
    channel = PRMSChannel(
        control=control,
        discretization=discretization,
        parameters=parameters,
        calc_method='jitclass'  # This will use the Calc class
    )

    Performance Benefits:
    -------------------
    - JIT compilation provides significant speedup for channel calculations
    - Eliminates Python overhead in computational loops
    - Optimized memory access patterns
    - Particularly beneficial for large channel networks and long simulations
    """

    def __init__(
        self,
        nhru,
        nsegment,
        hru_segment,
        segment_order,
        tosegment,
        tsi,
        ts,
        c0,
        c1,
        c2,
        outflow_mask,
    ):
        self.nhru = nhru
        self.nsegment = nsegment
        self.hru_segment = hru_segment
        self.segment_order = segment_order
        self.tosegment = tosegment
        self.tsi = tsi
        self.ts = ts
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.outflow_mask = outflow_mask

    def calculate(
        self,
        sroff_vol,
        ssres_flow_vol,
        gwres_flow_vol,
        seg_lateral_inflow,
        inflow_ts_prev,
        outflow_ts,
        s_per_time,
    ):
        """
        Perform channel flow calculations using Muskingum-Mann routing.

        This method contains the core logic from PRMSChannel._calculate(),
        optimized for Numba JIT compilation.

        Returns:
            tuple: (seg_upstream_inflow, inflow_ts_prev, seg_inflow,
                   seg_outflow, inflow_ts, outflow_ts, seg_current_sum,
                   seg_stor_change, channel_outflow_vol, channel_sroff_vol,
                   channel_ssres_flow_vol, channel_gwres_flow_vol)
        """
        # Initialize output arrays
        channel_sroff_vol = np.zeros(self.nhru)
        channel_ssres_flow_vol = np.zeros(self.nhru)
        channel_gwres_flow_vol = np.zeros(self.nhru)

        # Calculate lateral flow term
        seg_lateral_inflow.fill(0.0)
        for ihru in range(self.nhru):
            iseg = self.hru_segment[ihru]
            if iseg < 0:
                # Discard fluxes for segments not connected
                channel_sroff_vol[ihru] = 0.0
                channel_ssres_flow_vol[ihru] = 0.0
                channel_gwres_flow_vol[ihru] = 0.0
                continue
            else:
                channel_sroff_vol[ihru] = sroff_vol[ihru]
                channel_ssres_flow_vol[ihru] = ssres_flow_vol[ihru]
                channel_gwres_flow_vol[ihru] = gwres_flow_vol[ihru]

            # Convert cubic feet to cfs
            lateral_inflow = (
                channel_sroff_vol[ihru]
                + channel_ssres_flow_vol[ihru]
                + channel_gwres_flow_vol[ihru]
            ) / s_per_time

            seg_lateral_inflow[iseg] += lateral_inflow

        # Solve muskingum_mann routing
        (
            seg_upstream_inflow,
            inflow_ts_prev,
            seg_inflow,
            seg_outflow,
            inflow_ts,
            outflow_ts,
            seg_current_sum,
        ) = self._muskingum_mann_routing(
            seg_lateral_inflow, inflow_ts_prev, outflow_ts
        )

        # Calculate storage changes
        seg_stor_change = (seg_inflow - seg_outflow) * s_per_time

        # Calculate channel outflow volumes
        channel_outflow_vol = np.zeros(self.nsegment)
        for i in range(self.nsegment):
            if self.outflow_mask[i] == 1:
                channel_outflow_vol[i] = seg_outflow[i] * s_per_time

        return (
            seg_upstream_inflow,
            inflow_ts_prev,
            seg_inflow,
            seg_outflow,
            inflow_ts,
            outflow_ts,
            seg_current_sum,
            seg_stor_change,
            channel_outflow_vol,
            channel_sroff_vol,
            channel_ssres_flow_vol,
            channel_gwres_flow_vol,
        )

    def _muskingum_mann_routing(
        self, seg_lateral_inflow, inflow_ts_prev, outflow_ts
    ):
        """Muskingum routing function for segment flow calculations."""
        # Initialize variables for the day
        seg_inflow = np.zeros_like(inflow_ts_prev)
        seg_outflow = np.zeros_like(inflow_ts_prev)
        inflow_ts = np.zeros_like(inflow_ts_prev)
        seg_current_sum = np.zeros_like(inflow_ts_prev)

        for ihr in range(24):
            seg_upstream_inflow = np.zeros_like(seg_inflow)

            for jseg in self.segment_order:
                # Current inflow to the segment
                seg_current_inflow = (
                    seg_lateral_inflow[jseg] + seg_upstream_inflow[jseg]
                )

                seg_inflow[jseg] += seg_current_inflow
                inflow_ts[jseg] += seg_current_inflow
                seg_current_sum[jseg] += seg_upstream_inflow[jseg]

                remainder = (ihr + 1) % self.tsi[jseg]
                if remainder == 0:
                    # Segment routed on current hour
                    inflow_ts[jseg] /= self.ts[jseg]

                    if self.tsi[jseg] > 0:
                        # Muskingum routing equation
                        outflow_ts[jseg] = (
                            inflow_ts[jseg] * self.c0[jseg]
                            + inflow_ts_prev[jseg] * self.c1[jseg]
                            + outflow_ts[jseg] * self.c2[jseg]
                        )
                    else:
                        # Travel time is 1 hour or less
                        outflow_ts[jseg] = inflow_ts[jseg]

                    # Update previous inflow
                    inflow_ts_prev[jseg] = inflow_ts[jseg]
                    inflow_ts[jseg] = 0.0

                # Segment outflow (daily average)
                seg_outflow[jseg] += outflow_ts[jseg]

                # Add to upstream flow for connected segment
                to_seg = self.tosegment[jseg]
                if to_seg >= 0:
                    seg_upstream_inflow[to_seg] += outflow_ts[jseg]

        # Convert to daily averages
        seg_outflow /= 24.0
        seg_inflow /= 24.0
        seg_upstream_inflow = seg_current_sum.copy() / 24.0

        return (
            seg_upstream_inflow,
            inflow_ts_prev,
            seg_inflow,
            seg_outflow,
            inflow_ts,
            outflow_ts,
            seg_current_sum,
        )


class PRMSChannel(ConservativeProcess):
    """PRMS channel flow (muskingum_mann).

    A representation of channel flow from PRMS.

    Implementation based on PRMS 5.2.1 with theoretical documentation given in
    the PRMS-IV documentation:

    `Markstrom, S. L., Regan, R. S., Hay, L. E., Viger, R. J., Webb, R. M.,
    Payn, R. A., & LaFontaine, J. H. (2015). PRMS-IV, the
    precipitation-runoff modeling system, version 4. US Geological Survey
    Techniques and Methods, 6, B7.
    <https://pubs.usgs.gov/tm/6b7/pdf/tm6-b7.pdf>`__

    The muskingum module was originally developed for the Precipitation Runoff
    Modeling System (PRMS) by Mastin and Vaccaro (2002) and developed further
    by Markstrom and others (2008). This module has been modified from past
    versions to make it more stable for stream network routing in watersheds
    with stream segments with varying travel times. Although this module runs
    on the same daily time step as the rest of the modules in PRMS, it has an
    internal structure which allows for a different computational time step for
    each segment in the stream network, ensuring that the simulation produces
    stable values. Flow values computed at these finer time steps are
    aggregated by the Muskingum module to provide consistent daily time step
    values, regardless of the segment length.

    Delta t, which is the travel time (in hours), is rounded down to an even
    divisor of 24 hours (for example 24, 12, 6, 4, 3, 2, and 1). PRMS is
    restricted to daily time steps, so Delta t segment can never be more than
    one day in length. This means that the travel time of any segment in the
    stream network (K_coef) must be less than one day. An implication of this
    is that the routed streamflow in each segment is computed using different
    solution time steps. Consequently, streamflow must be aggregated when
    flowing from segments with shorter Delta t segment to segments with longer
    Delta t. Likewise, streamflow must be disaggregated when flowing from
    segments with longer Delta to shorter Delta t. In either case, flow
    from upstream segments is averaged and summed to the appropriate value of
    Delta t.

    The muskingum_mann method is a modified version of the original muskingum
    function in PRMS that was introduced in PRMS version 5.2.1 (1/20/2021).
    The muskingum_mann method provides a method to calculate K_coef values
    using mann_n, seg_length, seg_depth (bank full), and seg_slope. The
    velocity at bank full segment depth is calculated using Manning's equation

        ``velocity = ((1/n) sqrt(seg_slope) seg_depth**(2/3)``

    K_coef ,in hours, is then calculated using

        ``K_coef = seg_length / (velocity * 60 * 60)``

    K_coef values computed greater than 24.0 are set to 24.0, values computed
    less than 0.01 are set to 0.01, and the value for lake HRUs is set to 24.0.

    Args:
        control: a Control object
        discretization: a discretization of class Parameters
        parameters: a parameter object of class Parameters
        sroff_vol: Surface runoff to the stream network for each HRU
        ssres_flow_vol: Interflow volume from gravity and preferential-flow
            reservoirs to the stream network for each HRU
        gwres_flow_vol: Groundwater discharge volume from each GWR to the
            stream network
        budget_type: one of ["defer", None, "warn", "error"] with "defer" being
            the default and defering to control.options["budget_type"] when
            available. When control.options["budget_type"] is not avaiable,
            budget_type is set to "warn".
        calc_method: one of ["numba", "numpy"]. None defaults to
            "numba".
        adjust_parameters: one of ["warn", "error", "no"]. Default is "warn",
            the code edits the parameters and issues a warning. If "error" is
            selected the the code issues warnings about all edited parameters
            before raising the error to give you information. If "no" is
            selected then no parameters are adjusted and there will be no
            warnings or errors.
        verbose: Print extra information or not?
    """

    def __init__(
        self,
        control: Control,
        discretization: Parameters,
        parameters: Parameters,
        sroff_vol: adaptable,
        ssres_flow_vol: adaptable,
        gwres_flow_vol: adaptable,
        budget_type: Literal["defer", None, "warn", "error"] = "defer",
        calc_method: Literal["numba", "numpy"] = None,
        adjust_parameters: Literal["warn", "error", "no"] = "warn",
        verbose: bool = None,
        restart_read: Union[pl.Path, bool] = False,
        restart_write: Union[pl.Path, bool] = False,
        restart_write_freq: Literal["y", "m", "d", False] = False,
    ) -> None:
        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
            restart_read=restart_read,
            restart_write=restart_write,
            restart_write_freq=restart_write_freq,
        )
        self.name = "PRMSChannel"

        self._set_inputs(locals())
        self._set_options(locals())

        self._set_budget(basis="global")
        self._initialize_channel_data()
        self._init_calc_method()

        return

    @staticmethod
    def get_dimensions() -> tuple:
        return ("nhru", "nsegment")

    @staticmethod
    def get_parameters() -> tuple:
        return (
            "hru_area",
            "hru_segment",
            "mann_n",
            "seg_depth",
            "seg_length",
            "seg_slope",
            "segment_type",
            "tosegment",
            "tosegment_nhm",
            "x_coef",
            "segment_flow_init",
            "obsin_segment",
            "obsout_segment",
        )

    @staticmethod
    def get_inputs() -> tuple:
        return (
            "sroff_vol",
            "ssres_flow_vol",
            "gwres_flow_vol",
        )

    @staticmethod
    def get_init_values() -> dict:
        return {
            "channel_sroff_vol": nan,
            "channel_ssres_flow_vol": nan,
            "channel_gwres_flow_vol": nan,
            "channel_outflow_vol": nan,
            "seg_lateral_inflow": zero,
            "seg_upstream_inflow": zero,
            "seg_inflow": zero,
            "inflow_ts_prev": nan,
            "seg_outflow": zero,
            "seg_stor_change": zero,
            "outflow_ts": zero,
        }

    @staticmethod
    def get_restart_variables() -> list:
        """Get a dict of restart varible names mapping current: previous."""
        return ["seg_inflow", "outflow_ts"]

    @staticmethod
    def get_mass_budget_terms() -> dict:
        return {
            "inputs": [
                "channel_sroff_vol",
                "channel_ssres_flow_vol",
                "channel_gwres_flow_vol",
            ],
            "outputs": ["channel_outflow_vol"],
            "storage_changes": ["seg_stor_change"],
        }

    def get_outflow_mask(self):
        return self._outflow_mask

    @property
    def outflow_mask(self):
        return self._outflow_mask

    def _set_initial_conditions(self) -> None:
        self.seg_outflow[:] = self.segment_flow_init
        self.inflow_ts_prev[:] = self.seg_inflow
        return

    def _init_diagnostic_vars(self) -> None:
        return

    def _initialize_channel_data(self) -> None:
        """Initialize internal variables from raw channel data"""

        # convert prms data to zero-based
        self._hru_segment = self.hru_segment - 1
        self._tosegment = self.tosegment - 1
        self._tosegment = self._tosegment.astype("int64")

        # calculate connectivity
        self._outflow_mask = np.full((len(self._tosegment)), False)
        connectivity = []
        for iseg in range(self.nsegment):
            tosegment = self._tosegment[iseg]
            if tosegment < 0:
                self._outflow_mask[iseg] = True
                continue
            connectivity.append(
                (
                    iseg,
                    tosegment,
                )
            )

        # use networkx to calculate the Directed Acyclic Graph
        if self.nsegment > 1:
            self._graph = nx.DiGraph()
            self._graph.add_edges_from(connectivity)
            segment_order = list(nx.topological_sort(self._graph))
        else:
            segment_order = [0]

        # if the domain contains links with no upstream or
        # downstream reaches, we just throw these back at the
        # top of the order since networkx wont handle such nonsense
        wh_mask_set = set(np.where(self._outflow_mask)[0])
        seg_ord_set = set(segment_order)
        mask_not_seg_ord = list(wh_mask_set - seg_ord_set)
        if len(mask_not_seg_ord):
            segment_order = mask_not_seg_ord + segment_order
            # for pp in mask_not_seg_ord:
            #    assert (tosegment[pp] == -1) and (not pp in tosegment)

        self._segment_order = np.array(segment_order, dtype="int64")

        # calculate the Muskingum parameters
        velocity = (
            (
                (1.0 / self.mann_n)
                * np.sqrt(self.seg_slope)
                * self.seg_depth ** (2.0 / 3.0)
            )
            * 60.0
            * 60.0
        )
        # JLM: This is a bad idea and should throw an error rather than edit
        # inputs in place during run
        # should also be done before computing velocity
        mask_too_flat = self.seg_slope < 1e-7
        if mask_too_flat.any() and self._adjust_parameters != "no":
            msg = (
                "seg_slope < 1.0e-7, set to 1.0e-4 at indices:"
                f"{np.where(mask_too_flat)[0]}"
            )
            warn(msg, UserWarning)
            if self._adjust_parameters == "error":
                raise ValueError(
                    "seg_slope parameter values were edited and an error was "
                    "requested. See warnings for additional details."
                )
            # not in prms6
            self.seg_slope = np.where(mask_too_flat, 1.0e-4, self.seg_slope)

        # JDH: initialize Kcoef to 24.0 for segments with zero velocities
        # this is different from PRMS, which relied on divide by zero resulting
        # in a value of infinity that when evaluated relative to a maximum
        # desired Kcoef value of 24 would be reset to 24. This approach is
        # equivalent and avoids the occurence of a divide by zero.
        Kcoef = np.full(self.nsegment, 24.0, dtype=float)

        # only calculate Kcoef for cells with velocities greater than zero
        idx = velocity > 0.0
        Kcoef[idx] = self.seg_length[idx] / velocity[idx]
        Kcoef = np.where(
            self.segment_type == SegmentType.LAKE.value, 24.0, Kcoef
        )
        Kcoef = np.where(Kcoef < 0.01, 0.01, Kcoef)
        self._Kcoef = np.where(Kcoef > 24.0, 24.0, Kcoef)

        self._ts = np.ones(self.nsegment, dtype=float)
        self._tsi = np.ones(self.nsegment, dtype="int64")

        # todo: vectorize this
        for iseg in range(self.nsegment):
            k = self._Kcoef[iseg]
            if k < 1.0:
                self._tsi[iseg] = -1
            elif k < 2.0:
                self._ts[iseg] = 1.0
                self._tsi[iseg] = 1
            elif k < 3.0:
                self._ts[iseg] = 2.0
                self._tsi[iseg] = 2
            elif k < 4.0:
                self._ts[iseg] = 3.0
                self._tsi[iseg] = 3
            elif k < 6.0:
                self._ts[iseg] = 4.0
                self._tsi[iseg] = 4
            elif k < 8.0:
                self._ts[iseg] = 6.0
                self._tsi[iseg] = 6
            elif k < 12.0:
                self._ts[iseg] = 8.0
                self._tsi[iseg] = 8
            elif k < 24.0:
                self._ts[iseg] = 12.0
                self._tsi[iseg] = 12
            else:
                self._ts[iseg] = 24.0
                self._tsi[iseg] = 24

        d = self._Kcoef - (self._Kcoef * self.x_coef) + (0.5 * self._ts)
        d = np.where(np.abs(d) < 1e-6, 0.0001, d)
        self._c0 = (-(self._Kcoef * self.x_coef) + (0.5 * self._ts)) / d
        self._c1 = ((self._Kcoef * self.x_coef) + (0.5 * self._ts)) / d
        self._c2 = (
            self._Kcoef - (self._Kcoef * self.x_coef) - (0.5 * self._ts)
        ) / d

        # Short travel time
        idx = self._c2 < 0.0
        self._c1[idx] += self._c2[idx]
        self._c2[idx] = 0.0

        # Long travel time
        idx = self._c0 < 0.0
        self._c1[idx] += self._c0[idx]
        self._c0[idx] = 0.0

        # local flow variables
        self._inflow_ts = np.zeros(self.nsegment, dtype=float)
        self._seg_current_sum = np.zeros(self.nsegment, dtype=float)

        if not self._restart_read:
            for iseg in range(self.nsegment):
                jseg = self._tosegment[iseg]
                if jseg < 0:
                    continue
                self.seg_inflow[jseg] = self.seg_outflow[iseg]

        return

    def _init_calc_method(self):
        if self._calc_method is None:
            self._calc_method = "numba"

        avail_methods = ["numpy", "numba", "jitclass"]
        if self._calc_method.lower() not in avail_methods:
            msg = (
                f"Invalid calc_method={self._calc_method} for {self.name}. "
                f"Setting calc_method to 'numba' for {self.name}"
            )
            warn(msg)
            self._calc_method = "numba"

        if self._calc_method.lower() == "jitclass":
            if not NUMBA_AVAILABLE:
                msg = (
                    f"numba not available for jitclass method in {self.name}. "
                    f"Falling back to 'numba' method"
                )
                warn(msg)
                self._calc_method = "numba"
            else:
                # Use the new Calc jitclass for maximum performance
                print(f"{self.name} using Calc jitclass", flush=True)
                outflow_mask_int = self._outflow_mask.astype(np.int32)
                self._calc_instance = Calc(
                    self.nhru,
                    self.nsegment,
                    self._hru_segment.astype(np.int32),
                    self._segment_order.astype(np.int32),
                    self._tosegment.astype(np.int32),
                    self._tsi.astype(np.int32),
                    self._ts,
                    self._c0,
                    self._c1,
                    self._c2,
                    outflow_mask_int,
                )
                self._use_calc_instance = True

        elif self._calc_method.lower() == "numba":
            import numba as nb

            numba_msg = f"{self.name} jit compiling with numba "
            # this method can not be parallelized (? true?)
            print(numba_msg, flush=True)

            self._muskingum_mann = nb.njit(
                nb.types.UniTuple(nb.float64[:], 7)(
                    nb.int64[:],  # _segment_order
                    nb.int64[:],  # _tosegment
                    nb.float64[:],  # seg_lateral_inflow
                    nb.float64[:],  # inflow_ts_prev
                    nb.float64[:],  # _outflow_ts
                    nb.int64[:],  # _tsi
                    nb.float64[:],  # _ts
                    nb.float64[:],  # _c0
                    nb.float64[:],  # _c1
                    nb.float64[:],  # _c2
                ),
                fastmath=True,
                parallel=False,
            )(self._muskingum_mann_numpy)
            self._use_calc_instance = False

        else:
            self._muskingum_mann = self._muskingum_mann_numpy
            self._use_calc_instance = False

    def _advance_variables(self) -> None:
        # Seems strange to set an instantaneous value to an avg one.
        # Cant the method use instantaneous values across timesteps?
        self.inflow_ts_prev[:] = self.seg_inflow

        return

    def _calculate(self, simulation_time: float) -> None:
        self._simulation_time = simulation_time

        # This could vary with timestep so leave here
        s_per_time = self.control.time_step_seconds

        if self._use_calc_instance:
            # Use the Calc jitclass for high-performance calculation
            (
                self.seg_upstream_inflow[:],
                self.inflow_ts_prev[:],
                self.seg_inflow[:],
                self.seg_outflow[:],
                self._inflow_ts[:],
                self.outflow_ts[:],
                self._seg_current_sum[:],
                self.seg_stor_change[:],
                self.channel_outflow_vol[:],
                self.channel_sroff_vol[:],
                self.channel_ssres_flow_vol[:],
                self.channel_gwres_flow_vol[:],
            ) = self._calc_instance.calculate(
                self.sroff_vol,
                self.ssres_flow_vol,
                self.gwres_flow_vol,
                self.seg_lateral_inflow,
                self.inflow_ts_prev,
                self.outflow_ts,
                s_per_time,
            )
        else:
            # Use the original calculation method
            # WRITE a function for this?
            # calculate lateral flow term
            self.seg_lateral_inflow[:] = 0.0
            for ihru in range(self.nhru):
                iseg = self._hru_segment[ihru]
                if iseg < 0:
                    # This is bad, selective handling of fluxes is not cool,
                    # mass is being discarded in a way that has to be coordinated
                    # with other parts of the code.
                    # This code shuold be removed evenutally.
                    self.channel_sroff_vol[ihru] = zero
                    self.channel_ssres_flow_vol[ihru] = zero
                    self.channel_gwres_flow_vol[ihru] = zero
                    continue

                else:
                    self.channel_sroff_vol[ihru] = self.sroff_vol[ihru]
                    self.channel_ssres_flow_vol[ihru] = self.ssres_flow_vol[
                        ihru
                    ]
                    self.channel_gwres_flow_vol[ihru] = self.gwres_flow_vol[
                        ihru
                    ]

                # cubicfeet to cfs
                lateral_inflow = (
                    self.channel_sroff_vol[ihru]
                    + self.channel_ssres_flow_vol[ihru]
                    + self.channel_gwres_flow_vol[ihru]
                ) / (s_per_time)

                self.seg_lateral_inflow[iseg] += lateral_inflow

            # solve muskingum_mann routing
            (
                self.seg_upstream_inflow[:],
                self.inflow_ts_prev[:],
                self.seg_inflow[:],
                self.seg_outflow[:],
                self._inflow_ts[:],
                self.outflow_ts[:],
                self._seg_current_sum[:],
            ) = self._muskingum_mann(
                self._segment_order,
                self._tosegment,
                self.seg_lateral_inflow,
                self.inflow_ts_prev,
                self.outflow_ts,
                self._tsi,
                self._ts,
                self._c0,
                self._c1,
                self._c2,
            )

            self.seg_stor_change[:] = (
                self.seg_inflow - self.seg_outflow
            ) * s_per_time

            self.channel_outflow_vol[:] = (
                np.where(self._outflow_mask, self.seg_outflow, zero)
            ) * s_per_time

        return

    @staticmethod
    def _muskingum_mann_numpy(
        segment_order: np.ndarray,
        to_segment: np.ndarray,
        seg_lateral_inflow: np.ndarray,
        inflow_ts_prev: np.ndarray,
        outflow_ts: np.ndarray,
        tsi: np.ndarray,
        ts: np.ndarray,
        c0: np.ndarray,
        c1: np.ndarray,
        c2: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Muskingum routing function that calculates the upstream inflow and
        outflow for each segment

        Args:
            segment_order: segment routing order
            to_segment: downstream segment for each segment
            seg_lateral_inflow: segment lateral inflow
            inflow_ts_prev: previous segment inflow variable (subtimestep
                calculations)
            outflow_ts: outflow timeseries variable (subtimestep calculations)
            tsi: integer flood wave travel time
            ts: float version of integer flood wave travel time
            c0: Muskingum c0 variable
            c1: Muskingum c1 variable
            c2: Muskingum c2 variable

        Returns:
            seg_upstream_inflow: inflow for each segment for the current day
            inflow_ts_prev: segment inflow variable on subtimestep
            seg_inflow: segment inflow variable
            seg_outflow: outflow for each segment for the current day
            inflow_ts: inflow timeseries variable
            outflow_ts: outflow timeseries variable (internal calculations)
            seg_current_sum: summation variable
        """
        # initialize variables for the day

        seg_inflow = inflow_ts_prev * zero
        seg_outflow = inflow_ts_prev * zero
        inflow_ts = inflow_ts_prev * zero
        seg_current_sum = inflow_ts_prev * zero

        for ihr in range(24):
            seg_upstream_inflow = seg_inflow * zero

            for jseg in segment_order:
                # current inflow to the segment is the time-weighted average
                # of the outflow of the upstream segments and the lateral HRU
                # inflow plus any gains
                seg_current_inflow = (
                    seg_lateral_inflow[jseg] + seg_upstream_inflow[jseg]
                )

                # todo: evaluate if obsin_segment needs to be implemented -
                #  would be needed needed if headwater basins are not included
                #  in a simulation
                # seg_current_inflow += seg_upstream_inflow[jseg]

                seg_inflow[jseg] += seg_current_inflow
                inflow_ts[jseg] += seg_current_inflow
                seg_current_sum[jseg] += seg_upstream_inflow[jseg]

                remainder = (ihr + 1) % tsi[jseg]
                if remainder == 0:
                    # segment routed on current hour
                    inflow_ts[jseg] /= ts[jseg]

                    if tsi[jseg] > 0:
                        # todo: evaluate if denormal results should be treated

                        # Muskingum routing equation
                        outflow_ts[jseg] = (
                            inflow_ts[jseg] * c0[jseg]
                            + inflow_ts_prev[jseg] * c1[jseg]
                            + outflow_ts[jseg] * c2[jseg]
                        )
                    else:
                        # travel time is 1 hour or less so outflow is set
                        # equal to the inflow - outflow_ts is the value for
                        # the previous hour
                        outflow_ts[jseg] = inflow_ts[jseg]

                    # previous inflow is equal to inflow_ts from the previous
                    # routed time step
                    inflow_ts_prev[jseg] = inflow_ts[jseg]

                    # upstream inflow is used, reset it to zero so a new
                    # average can be calculated next routing time step
                    inflow_ts[jseg] = 0.0

                # todo: evaluate if obsout_segment needs to be implemented -
                #  would be needed needed fixing ourflow to observed data is
                #  required in a simulation

                # todo: water use

                # segment outflow (the mean daily flow rate for each segment)
                # will be the average of hourly values
                seg_outflow[jseg] += outflow_ts[jseg]

                # add current time step flow rate to the upstream flow rate
                # for the segment this segment is connected to
                to_seg = to_segment[jseg]
                if to_seg >= 0:
                    seg_upstream_inflow[to_seg] += outflow_ts[jseg]

        seg_outflow /= 24.0
        seg_inflow /= 24.0
        seg_upstream_inflow = seg_current_sum.copy() / 24.0

        return (
            seg_upstream_inflow,
            inflow_ts_prev,
            seg_inflow,
            seg_outflow,
            inflow_ts,
            outflow_ts,
            seg_current_sum,
        )
