import math
from typing import Literal

import numpy as np

from ..base.adapter import adaptable
from ..base.conservative_process import ConservativeProcess
from ..base.control import Control
from ..constants import nan, zero
from ..parameters import Parameters
from .prms_stream_shade import PRMSStreamShade

# Constants from PRMS
NEARZERO = 1e-6
PI = np.pi
HALF_PI = PI / 2.0
RADTOHOUR = 24.0 / (2.0 * PI)
CFS_TO_CMS = 0.028316847
NOFLOW_TEMP = -98.9
DAYS_YR = 365.25
MAX_DAYS_PER_YEAR = int(DAYS_YR)
ZERO_C = 273.15
TOLRN = 1.0e-4
AKZ = 1.65
A = 5.40e-8
MPS_CONVERT = 2.93981481e-07

# Gaussian quadrature points and weights (15-point)
EPSLON = np.array(
    [
        0.006003741,
        0.031363304,
        0.075896109,
        0.137791135,
        0.214513914,
        0.302924330,
        0.399402954,
        0.500000000,
        0.600597047,
        0.697075674,
        0.785486087,
        0.862208866,
        0.924103292,
        0.968636696,
        0.993996259,
    ]
)

WEIGHT = np.array(
    [
        0.015376621,
        0.035183024,
        0.053579610,
        0.069785339,
        0.083134603,
        0.093080500,
        0.099215743,
        0.101289120,
        0.099215743,
        0.093080500,
        0.083134603,
        0.069785339,
        0.053579610,
        0.035183024,
        0.015376621,
    ]
)


class PRMSStreamTemp(ConservativeProcess):
    """PRMS stream temperature with composition-based design.

    A representation of stream temperature from PRMS. This class uses a
    composition-based design where:
    - Hydraulic geometry (seg_flow_*) is provided by upstream PRMSHydraulicGeometry process
    - Shade computation is handled by composed PRMSStreamShade strategy

    Implementation based on PRMS 5.2.1 with theoretical documentation given in
    the PRMS-IV documentation:

    `Markstrom, S. L., Regan, R. S., Hay, L. E., Viger, R. J., Webb, R. M.,
    Payn, R. A., & LaFontaine, J. H. (2015). PRMS-IV, the
    precipitation-runoff modeling system, version 4. US Geological Survey
    Techniques and Methods, 6, B7.
    <https://pubs.usgs.gov/tm/6b7/pdf/tm6-b7.pdf>`__

    The stream temperature module computes daily mean water temperature for
    each stream segment using an energy balance approach. The module accounts
    for:
    - Solar radiation (shortwave)
    - Longwave radiation (atmospheric and vegetation)
    - Convection and evaporation
    - Conduction with the streambed
    - Temperature of inflows (upstream, lateral, groundwater, subsurface)
    - Shading from riparian vegetation and topography

    Args:
        control: a Control object
        discretization: a discretization of class Parameters
        parameters: a parameter object of class Parameters
        seg_outflow: Streamflow leaving each segment
        seg_lateral_inflow: Lateral inflow entering each segment
        swrad: Solar radiation for each HRU
        potet: Potential ET for each HRU
        sroff: Surface runoff for each HRU
        ssres_flow: Subsurface flow for each HRU
        gwres_flow: Groundwater flow for each HRU
        seg_humid: Humidity for each segment
        seg_ccov: Cloud cover for each segment
        seg_melt: Snowmelt for each segment
        seg_rain: Rainfall for each segment
        seg_tave_air: Air temperature for each segment
        seg_flow_width: Flow-dependent width from PRMSHydraulicGeometry
        seg_flow_depth: Flow-dependent depth from PRMSHydraulicGeometry
        seg_flow_area: Flow-dependent cross-sectional area from PRMSHydraulicGeometry
        seg_flow_velocity: Flow-dependent velocity from PRMSHydraulicGeometry
        shade_computer: PRMSStreamShade instance (Dynamic or Constant)
        budget_type: one of ["defer", None, "warn", "error"]
        verbose: Print extra information or not?
    """

    def __init__(
        self,
        control: Control,
        discretization: Parameters,
        parameters: Parameters,
        seg_outflow: adaptable,
        seg_lateral_inflow: adaptable,
        swrad: adaptable,
        potet: adaptable,
        sroff: adaptable,
        ssres_flow: adaptable,
        gwres_flow: adaptable,
        seg_humid: adaptable,
        seg_ccov: adaptable,
        seg_melt: adaptable,
        seg_rain: adaptable,
        seg_tave_air: adaptable,
        seg_flow_width: adaptable,
        seg_flow_depth: adaptable,
        seg_flow_area: adaptable,
        seg_flow_velocity: adaptable,
        shade_computer: PRMSStreamShade,
        budget_type: Literal["defer", None, "warn", "error"] = "defer",
        verbose: bool = False,
    ) -> None:
        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
        )
        self.name = "PRMSStreamTemp"

        self._set_inputs(locals())
        self._set_options(locals())

        # Store the composed shade computer
        self.shade_computer = shade_computer
        # Set parent process reference so shade computer can access _shday methods
        if hasattr(self.shade_computer, "parent_process"):
            self.shade_computer.parent_process = self

        self._set_budget(basis="global")
        self._initialize_stream_temp()

        return

    @staticmethod
    def get_dimensions() -> tuple:
        return ("nhru", "nsegment", "nmonth")

    @staticmethod
    def get_parameters() -> tuple:
        return (
            "hru_segment",
            "hru_area",
            "tosegment",
            "albedo",
            "lat_temp_adj",
            "seg_length",
            "seg_slope",
            "seg_lat",
            "seg_elev",
            "ss_tau",
            "gw_tau",
            "melt_temp",
            "maxiter_sntemp",
            "stream_tave_init",
            # Note: shade parameters are provided by composed shade_computer
        )

    @staticmethod
    def get_inputs() -> tuple:
        return (
            "seg_outflow",
            "seg_lateral_inflow",
            "swrad",
            "potet",
            "sroff",
            "ssres_flow",
            "gwres_flow",
            "seg_humid",
            "seg_ccov",
            "seg_melt",
            "seg_rain",
            "seg_tave_air",
            "seg_flow_width",
            "seg_flow_depth",
            "seg_flow_area",
            "seg_flow_velocity",
        )

    @staticmethod
    def get_init_values() -> dict:
        return {
            "seg_tave_water": nan,
            "seg_tave_upstream": 0.0,
            "seg_tave_gw": 0.0,
            "seg_tave_ss": 0.0,
            "seg_tave_lat": 0.0,
            "seg_shade": 0.0,
            "seg_potet": 0.0,
            "seginc_sroff": 0.0,
            "seginc_ssflow": 0.0,
            "seginc_gwflow": 0.0,
            "seginc_swrad": 0.0,
        }

    @staticmethod
    def get_mass_budget_terms() -> dict:
        # Temperature is not a mass, so no budget terms
        return {
            "inputs": [],
            "outputs": [],
            "storage_changes": [],
        }

    def _set_initial_conditions(self) -> None:
        # Initialize state variables
        # seg_tave_water starts as NaN, but will be set to -99.9 for segments
        # with no upstream HRUs (done after upstream info is computed)
        self.seg_tave_water[:] = np.nan
        self.seg_tave_upstream[:] = np.nan
        self.seg_tave_gw[:] = zero
        self.seg_tave_ss[:] = zero
        self.seg_tave_lat[:] = zero
        self.seg_shade[:] = zero
        self._seg_inflow = np.zeros(self.nsegment, dtype=np.float64)

        # Initialize circular buffers for temperature averaging if not already done
        if not hasattr(self, "gw_silo"):
            self.gw_silo = np.zeros(
                (self.nsegment, MAX_DAYS_PER_YEAR), dtype=np.float64
            )
            self.ss_silo = np.zeros(
                (self.nsegment, MAX_DAYS_PER_YEAR), dtype=np.float64
            )
            self.gw_index = np.zeros(self.nsegment, dtype=np.int32)
            self.ss_index = np.zeros(self.nsegment, dtype=np.int32)

        # Initialize circular buffers with 0.0 to match Fortran PRMS
        # The running sum will gradually build up as air temps are added
        self.gw_silo[:, :] = 0.0
        self.ss_silo[:, :] = 0.0

        # Initialize running sum arrays for temperature averaging
        if not hasattr(self, "gw_sum"):
            self.gw_sum = np.zeros(self.nsegment, dtype=np.float64)
            self.ss_sum = np.zeros(self.nsegment, dtype=np.float64)

        self.gw_sum[:] = 0.0
        self.ss_sum[:] = 0.0

        self.gw_index[:] = 0
        self.ss_index[:] = 0

        return

    def _initialize_stream_temp(self) -> None:
        """Initialize stream temperature data structures."""
        self.nsegment = self._params.dims["nsegment"]
        self.nhru = self._params.dims["nhru"]

        # Extract scalar parameters (dimension "one")
        # These are stored as arrays but should be scalars
        if hasattr(self, "maxiter_sntemp") and hasattr(
            self.maxiter_sntemp, "__len__"
        ):
            self.maxiter_sntemp = float(self.maxiter_sntemp[0])
        if hasattr(self, "albedo") and hasattr(self.albedo, "__len__"):
            self.albedo = float(self.albedo[0])
        if hasattr(self, "melt_temp") and hasattr(self.melt_temp, "__len__"):
            self.melt_temp = float(self.melt_temp[0])

        # Get segment ordering for upstream-to-downstream calculations
        self._compute_segment_order()

        # Note: Circular buffers (gw_silo, ss_silo, gw_index, ss_index) are
        # initialized in _set_initial_conditions to ensure proper timing

        # Initialize segment aggregate arrays
        self.seg_potet = np.zeros(self.nsegment, dtype=np.float64)
        self.seginc_sroff = np.zeros(self.nsegment, dtype=np.float64)
        self.seginc_ssflow = np.zeros(self.nsegment, dtype=np.float64)
        self.seginc_gwflow = np.zeros(self.nsegment, dtype=np.float64)
        self.seginc_swrad = np.zeros(self.nsegment, dtype=np.float64)

        # Note: seg_flow_* are inputs from PRMSHydraulicGeometry, not computed here

        # Compute segment HRU areas (sum of HRU areas contributing to each segment)
        self.segment_hruarea = np.zeros(self.nsegment, dtype=np.float64)
        for j in range(self.nhru):
            seg_idx = self.hru_segment[j]
            if seg_idx > 0:
                i = seg_idx - 1
                self.segment_hruarea[i] += self.hru_area[j]

        # Compute upstream segment information
        self._compute_upstream_info()

        # Compute segment_up - the single immediate upstream segment (matches Fortran Segment_up)
        # Fortran's Segment_up is computed by iterating segments and assigning Segment_up(toseg) = j
        # This means when multiple segments flow into one, it keeps the LAST one (highest j)
        # So we need to find the last upstream in segment numbering order
        segment_up = np.zeros(self.nsegment, dtype=np.int32)
        for j in range(self.nsegment):
            toseg = self.tosegment[j]
            if toseg > 0:
                # toseg is 1-based, convert to 0-based
                segment_up[toseg - 1] = j
        # Note: segment_up[i] = 0 means no upstream (default from zeros initialization)

        # Save segment_up as instance variable (needed for routing aggregation logic)
        self.segment_up = segment_up

        # Compute seg_close for segments without HRUs (matches Fortran line 529-570)
        # Initialize seg_close = segment_up (Fortran line 529)
        self.seg_close = np.copy(segment_up)

        # Now update seg_close for segments without HRUs (Fortran line 530-570)
        for jj in range(self.nsegment):
            i = self.segment_order[jj]

            # Only modify seg_close for segments without HRUs (Fortran line 539)
            if self.segment_hruarea[i] <= NEARZERO:
                # If no upstream segment (Fortran line 541)
                if self.segment_up[i] == 0:
                    # Try downstream segment (Fortran line 542-543)
                    if self.tosegment[i] > 0:
                        self.seg_close[i] = (
                            self.tosegment[i] - 1
                        )  # Convert to 0-based
                    else:
                        # No upstream or downstream - use previous/next in order (Fortran line 544-549)
                        if jj > 0:
                            self.seg_close[i] = self.segment_order[jj - 1]
                        else:
                            self.seg_close[i] = self.segment_order[jj + 1]

                # Check if seg_close points to invalid segment (Fortran line 551-563)
                # If elevation is exactly 30000 (invalid marker), find a different segment
                if self.seg_elev[self.seg_close[i]] == 30000.0:
                    found = False
                    # Find first segment with HRUs in forward order (Fortran line 553-558)
                    for k in range(jj + 1, self.nsegment):
                        ii = self.segment_order[k]
                        if self.segment_hruarea[ii] > NEARZERO:
                            self.seg_close[i] = ii
                            found = True
                            break

                    # If not found, use previous segment in order (Fortran line 560-565)
                    if not found:
                        if jj > 0:
                            self.seg_close[i] = self.segment_order[jj - 1]

        # Mark segments with no upstream HRUs as never having flow
        # (matches Fortran initialization around line 648)
        for i in range(self.nsegment):
            if self.segment_hruarea[i] <= NEARZERO:
                # Check if any upstream segments have HRUs
                has_upstream_hrus = False
                this_seg = i
                visited = set()

                while this_seg not in visited:
                    visited.add(this_seg)
                    # Check upstream segments
                    found_upstream = False
                    for j in range(self.nsegment):
                        if (
                            self.tosegment[j] > 0
                            and self.tosegment[j] == this_seg + 1
                        ):
                            if self.segment_hruarea[j] > NEARZERO:
                                has_upstream_hrus = True
                                break
                            this_seg = j
                            found_upstream = True
                            break

                    if has_upstream_hrus or not found_upstream:
                        break

                # If no upstream HRUs, mark as never having flow
                if not has_upstream_hrus:
                    self.seg_tave_water[i] = -99.9

        # Convert seg_length from meters to kilometers
        # Parameter file has seg_length in meters, but calculations use km
        self.seg_length = self.seg_length / 1000.0

        # Convert latitude to radians
        # Note: When using dynamic shade computation, seg_lat should be zero
        # to match Fortran behavior (see stream_temp.f90 line ~760)
        from .prms_stream_shade import PRMSStreamShadeDynamic

        if isinstance(self.shade_computer, PRMSStreamShadeDynamic):
            # Dynamic shade - zero latitude to match Fortran
            self.seg_lat_rad = np.zeros(self.nsegment, dtype=np.float64)
        else:
            # Constant shade or other - use actual latitude
            self.seg_lat_rad = np.deg2rad(self.seg_lat)

        # Precompute solar geometry for each day of year
        self._precompute_solar_geometry()

        return

    def _precompute_solar_geometry(self) -> None:
        """Precompute solar declination for each day of year."""
        # Solar declination for each day of year
        self.declination = np.zeros(MAX_DAYS_PER_YEAR)
        for jday in range(MAX_DAYS_PER_YEAR):
            k = jday + 1  # Convert to 1-based day of year
            self.declination[jday] = 0.40928 * np.cos(
                ((2.0 * PI) / DAYS_YR) * (172.0 - k)
            )  # radians

        return

    def _compute_segment_order(self) -> None:
        """Compute topologically sorted segment order for calculation."""
        # Simple topological sort based on tosegment
        self.segment_order = []
        visited = set()
        temp_mark = set()

        def visit(seg):
            if seg in temp_mark:
                raise ValueError(
                    f"Circular dependency detected at segment {seg}"
                )
            if seg in visited:
                return

            temp_mark.add(seg)

            # Visit upstream segments first
            for upstream_seg in range(self.nsegment):
                toseg = self.tosegment[upstream_seg]
                # tosegment is 1-based, with 0 meaning no downstream segment
                if toseg > 0 and toseg == seg + 1:
                    visit(upstream_seg)

            temp_mark.remove(seg)
            visited.add(seg)
            self.segment_order.append(seg)

        for seg in range(self.nsegment):
            if seg not in visited:
                visit(seg)

        self.segment_order = np.array(self.segment_order, dtype=np.int32)

        return

    def _compute_upstream_info(self) -> None:
        """Compute upstream segment information for each segment."""
        # Count upstream segments
        self.upstream_count = np.zeros(self.nsegment, dtype=np.int32)
        max_upstream = 10  # Assume max 10 upstream segments

        self.upstream_idx = np.zeros(
            (self.nsegment, max_upstream), dtype=np.int32
        )

        for jj in range(self.nsegment):
            count = 0
            for ii in range(self.nsegment):
                toseg = self.tosegment[ii]
                if (
                    toseg > 0 and toseg == jj + 1
                ):  # 1-based, with 0 meaning outlet
                    self.upstream_idx[jj, count] = ii
                    count += 1
            self.upstream_count[jj] = count

        return

    def _advance_variables(self) -> None:
        """Advance variables from current to previous timestep."""
        # No explicit time advancement needed for this module
        return

    def _calculate(self, time_length) -> None:
        """Calculate stream temperature for all segments."""
        # Get current month (1-based) and day of year
        nowmonth = self.control.current_month
        doy = min(self.control.current_doy, 365) - 1

        # Get declination for current day
        declination = self.declination[doy]

        # Determine summer flag for vegetation density (1=summer, 0=winter)
        summer_flag = 1 if 121 <= (doy + 1) <= 273 else 0

        # Compute segment aggregate variables from HRU inputs
        self._compute_segment_aggregates()

        # Compute seg_potet using stream_temp.f90 logic (after aggregates)
        self._compute_seg_potet()

        # Note: Hydraulic geometry (seg_flow_*) is now provided by upstream
        # PRMSHydraulicGeometry process, not computed here

        # Compute running average temperatures for groundwater and subsurface
        # Skip segments marked as never having flow (matches Fortran checks)
        for jj in self.segment_order:
            # Skip if marked as never having flow (Fortran line 887)
            if self.seg_tave_water[jj] < -99.0:
                continue
            # Skip if marked as permanently invalid (Fortran line 894)
            if self.seginc_swrad[jj] < -99.0:
                continue
            self._update_running_avg_temp(jj, "gw")
            self._update_running_avg_temp(jj, "ss")

        # Compute lateral flow temperatures
        # Skip segments marked as never having flow or data
        for jj in self.segment_order:
            if self.seg_tave_water[jj] < -99.0:
                continue
            if self.seginc_swrad[jj] < -99.0:
                continue
            self._compute_lateral_temp(jj, nowmonth)

        # Initialize seg_tave_upstream to 0.0 each timestep (matches Fortran line 463)
        # Segments that are skipped will keep this 0.0 value
        self.seg_tave_upstream[:] = 0.0

        # Compute shade and water temperature for each segment
        # Don't reset segments marked as -99.9 (never have flow)
        for jj in self.segment_order:
            if self.seg_tave_water[jj] >= -99.0:
                self.seg_tave_water[jj] = np.nan
        for jj in self.segment_order:
            # Skip segments marked as never having flow (matches Fortran cycle at line 887)
            if self.seg_tave_water[jj] < -99.0:
                continue

            # Compute upstream temperature
            self._compute_upstream_temp(jj)

            # Compute shade
            svi = self._compute_shade(jj, declination, summer_flag)

            # Compute segment inflow
            self._seg_inflow[jj] = self._compute_inflow(jj)

            # Compute water temperature
            self._compute_water_temp(jj, svi)

        return

    def _compute_segment_aggregates(self) -> None:
        """Compute segment aggregate variables from HRU inputs.

        This implements the aggregation calculations from PRMS routing.f90
        around line 699-805.
        """
        # Initialize segment aggregate variables
        self.seginc_sroff[:] = 0.0
        self.seginc_ssflow[:] = 0.0
        self.seginc_gwflow[:] = 0.0
        self.seginc_swrad[:] = 0.0

        # Constants (from PRMS_SET_TIME)
        # Cfs_conv converts acre-inches/day to cfs
        # FT2_PER_ACRE / INCHES_PER_FOOT / SECS_PER_DAY
        # = 43560 / 12 / 86400
        cfs_conv = 43560.0 / 12.0 / 86400.0

        # Aggregate HRU values to segments
        for j in range(self.nhru):
            seg_idx = self.hru_segment[j]

            # Check if HRU contributes to a segment (seg_idx > 0)
            # hru_segment is 1-based, so valid segments are > 0
            if seg_idx > 0:
                # Convert to 0-based index
                i = seg_idx - 1

                # Convert from inches to cfs (area * inches/day * cfs_conv)
                tocfs = self.hru_area[j] * cfs_conv

                # Accumulate flow components (converted to cfs)
                self.seginc_sroff[i] += self.sroff[j] * tocfs
                self.seginc_ssflow[i] += self.ssres_flow[j] * tocfs
                self.seginc_gwflow[i] += self.gwres_flow[j] * tocfs

                # Accumulate area-weighted radiation
                # (will be divided by total HRU area later)
                self.seginc_swrad[i] += self.swrad[j] * self.hru_area[j]

        # First: Process seginc_swrad in numerical order (routing.f90 logicDivide radiation and PET by segment HRU area to get averages
        # Process in numerical order to match routing.f90 (line 741-810)
        for i in range(self.nsegment):
            if self.segment_hruarea[i] > NEARZERO:
                self.seginc_swrad[i] /= self.segment_hruarea[i]

            else:
                # Segment has no HRUs - search upstream then downstream (matches routing.f90 line 746-805)
                # Search upstream first (routing.f90 line 749-772)
                this_seg = i
                found = False
                while not found:
                    if self.segment_hruarea[this_seg] <= NEARZERO:
                        # Check if headwater (no upstream)
                        upstream_seg = self.segment_up[this_seg]
                        if upstream_seg == 0:
                            found = False
                            break
                        # Move to upstream segment
                        this_seg = upstream_seg
                    else:
                        # Found segment with HRUs - copy values (already averaged)
                        self.seginc_swrad[i] = self.seginc_swrad[this_seg]
                        found = True
                        break

                # If not found upstream, search downstream (routing.f90 line 776-800)
                if not found:
                    this_seg = i
                    while not found:
                        if self.segment_hruarea[this_seg] <= NEARZERO:
                            # Check if terminal segment (no downstream)
                            if self.tosegment[this_seg] == 0:
                                found = False
                                break
                            # Move to downstream segment (tosegment is 1-based)
                            this_seg = self.tosegment[this_seg] - 1
                        else:
                            # Found segment with HRUs - copy values (already averaged)
                            self.seginc_swrad[i] = self.seginc_swrad[this_seg]
                            found = True
                            break

                # If still not found, set to invalid marker (routing.f90 line 803-805)
                if not found:
                    self.seginc_swrad[i] = -99.9

        return

    def _compute_seg_potet(self) -> None:
        """Compute seg_potet using stream_temp.f90 logic.

        This matches stream_temp.f90 lines 807-848, using segment_order and seg_close.
        """
        # Initialize
        self.seg_potet[:] = 0.0

        # Accumulate from HRUs
        for j in range(self.nhru):
            seg_idx = self.hru_segment[j]
            if seg_idx > 0:
                i = seg_idx - 1
                self.seg_potet[i] += self.potet[j] * self.hru_area[j]

        # Process in segment_order (stream_temp.f90 line 810)
        for jj in range(self.nsegment):
            i = self.segment_order[jj]

            if self.segment_hruarea[i] > NEARZERO:
                self.seg_potet[i] /= self.segment_hruarea[i]

            else:
                # Segment has no HRUs - use seg_close (stream_temp.f90 line 817)
                close_seg = self.seg_close[i]

                self.seg_potet[i] = self.seg_potet[close_seg]

        return

    # Note: _compute_hydraulic_geometry removed - now handled by PRMSHydraulicGeometry process

    def _update_running_avg_temp(self, seg_idx: int, comp_type: str) -> None:
        """Update running average temperature for groundwater or subsurface.

        This matches the Fortran PRMS implementation which uses a running sum
        divided by tau, with a circular buffer.

        Args:
            seg_idx: Segment index
            comp_type: "gw" for groundwater or "ss" for subsurface
        """
        if comp_type == "gw":
            tau = self.gw_tau[seg_idx]
            index = self.gw_index[seg_idx]
            silo = self.gw_silo
            sum_array = self.gw_sum
        else:  # "ss"
            tau = self.ss_tau[seg_idx]
            index = self.ss_index[seg_idx]
            silo = self.ss_silo
            sum_array = self.ss_sum

        # Remove old value from sum (circular buffer behavior)
        sum_array[seg_idx] -= silo[seg_idx, index]

        # Add new air temperature to silo
        silo[seg_idx, index] = self.seg_tave_air[seg_idx]

        # Add new value to sum
        sum_array[seg_idx] += silo[seg_idx, index]

        # Compute average as sum / tau (matches Fortran)
        avg_temp = sum_array[seg_idx] / tau

        if comp_type == "gw":
            self.seg_tave_gw[seg_idx] = avg_temp
            self.gw_index[seg_idx] = (index + 1) % int(tau)
        else:
            self.seg_tave_ss[seg_idx] = avg_temp
            self.ss_index[seg_idx] = (index + 1) % int(tau)

        return

    def _compute_lateral_temp(self, seg_idx: int, nowmonth: int) -> None:
        """Compute lateral flow temperature for a segment.

        Args:
            seg_idx: Segment index
            nowmonth: Current month (1-based)
        """
        sroff = self.seginc_sroff[seg_idx]
        ssflow = self.seginc_ssflow[seg_idx]
        gwflow = self.seginc_gwflow[seg_idx]
        melt = self.seg_melt[seg_idx]
        rain = self.seg_rain[seg_idx]

        tave_gw = self.seg_tave_gw[seg_idx]
        tave_air = self.seg_tave_air[seg_idx]
        tave_ss = self.seg_tave_ss[seg_idx]
        melt_temp = self.melt_temp

        # Use lat_inflow function for detailed lateral temperature calculation
        tl_avg, qlat = self._lat_inflow(
            self.seg_lateral_inflow[seg_idx],
            sroff,
            ssflow,
            gwflow,
            melt_temp,
            tave_gw,
            tave_air,
            tave_ss,
            melt,
            rain,
        )

        # Apply monthly adjustment
        if not np.isnan(tl_avg):
            tl_avg += self.lat_temp_adj[nowmonth - 1, seg_idx]

        # Ensure non-negative (also converts NaN to 0.0 to match Fortran)
        if np.isnan(tl_avg) or tl_avg < 0.0:
            tl_avg = 0.0

        self.seg_tave_lat[seg_idx] = tl_avg

        return

    def _lat_inflow(
        self,
        seg_lateral_inflow,
        seginc_sroff,
        seginc_ssflow,
        seginc_gwflow,
        melt_temp,
        tave_gw,
        tave_air,
        tave_ss,
        melt,
        rain,
    ):
        """Compute lateral inflow temperature from components.

        This is the lat_inflow function from PRMS.
        """
        weight_roff = 0.0
        weight_ss = 0.0
        weight_gw = 0.0
        melt_wt = 0.0
        rain_wt = 0.0
        troff = 0.0
        tss = 0.0

        qlat = seg_lateral_inflow * CFS_TO_CMS
        tl_avg = 0.0

        if qlat > 0.0:
            weight_roff = float((seginc_sroff * CFS_TO_CMS) / qlat)
            weight_ss = float((seginc_ssflow * CFS_TO_CMS) / qlat)
            weight_gw = float((seginc_gwflow * CFS_TO_CMS) / qlat)
        else:
            weight_roff = 0.0
            weight_ss = 0.0
            weight_gw = 0.0

        if melt > 0.0:
            melt_wt = melt / (melt + rain)
            if melt_wt < 0.0:
                melt_wt = 0.0
            if melt_wt > 1.0:
                melt_wt = 1.0
            rain_wt = 1.0 - melt_wt
            if rain == 0.0:
                troff = melt_temp
                tss = melt_temp
            else:
                troff = melt_temp * melt_wt + tave_air * rain_wt
                tss = melt_temp * melt_wt + tave_ss * rain_wt
        else:
            troff = tave_air
            tss = tave_ss

        if weight_roff == 0.0 and weight_ss == 0.0 and weight_gw == 0.0:
            tl_avg = np.nan
            qlat = np.nan
        else:
            tl_avg = (
                weight_roff * troff + weight_ss * tss + weight_gw * tave_gw
            )

        return tl_avg, qlat

    def _compute_upstream_temp(self, seg_idx: int) -> None:
        """Compute temperature from upstream segments.

        Args:
            seg_idx: Segment index
        """
        flow_sum = 0.0
        temp_sum = 0.0

        for kk in range(self.upstream_count[seg_idx]):
            up_idx = self.upstream_idx[seg_idx, kk]
            if not np.isnan(self.seg_tave_water[up_idx]):
                flow = self.seg_outflow[up_idx]
                temp_sum += self.seg_tave_water[up_idx] * flow
                flow_sum += flow

        if flow_sum > 0.0:
            self.seg_tave_upstream[seg_idx] = temp_sum / flow_sum
        else:
            self.seg_tave_upstream[seg_idx] = NOFLOW_TEMP

        return

    def _compute_shade(
        self, seg_idx: int, declination: float, summer_flag: int
    ) -> float:
        """Compute shade using the composed shade_computer.

        Args:
            seg_idx: Segment index
            declination: Solar declination (radians)
            summer_flag: 1 for summer, 0 for winter

        Returns:
            svi: Vegetation shade index
        """
        # Delegate to composed shade computer
        shade, svi = self.shade_computer.compute(
            seg_idx,
            declination,
            summer_flag,
            self.seg_flow_width[seg_idx],
        )

        self.seg_shade[seg_idx] = shade

        return svi

    def _shday(
        self,
        seg_lat,
        declination,
        seg_width,
        azrh,
        alte,
        altw,
        vce,
        voe,
        vhe,
        vdemx,
        vdemn,
        summer_flag,
        vcw,
        vow,
        vhw,
        vdwmx,
        vdwmn,
    ):
        """Compute daily shade from topography and vegetation.

        This is the shday function from PRMS.
        """
        if seg_width <= 0.0:
            return 0.0, 0.0

        coso = math.cos(seg_lat)
        if coso == 0.0:
            coso = NEARZERO

        sino = math.sin(seg_lat)
        sin_d = math.sin(declination)
        cos_d = math.cos(declination)
        sinod = sino * sin_d
        cosod = coso * cos_d

        hrsr = 0.0
        hrss = 0.0
        max_solar_altitude = np.arcsin(sinod + cosod)
        alsmx = max_solar_altitude

        # Calculate level-plain solar azimuth
        temp = -sin_d / coso

        if temp > 1.0:
            temp = 1.0
        elif temp < -1.0:
            temp = -1.0

        level_sunset_azimuth = np.arccos(temp)

        # Check for reach azimuth less than sunrise
        if azrh <= (-level_sunset_azimuth):
            alrs = 0.0
        # Check for reach azimuth greater than sunset
        elif azrh >= level_sunset_azimuth:
            alrs = 0.0
        # Reach azimuth is between sunrise and sunset
        elif azrh == 0.0:
            alrs = max_solar_altitude
        else:
            alrs = self._solalt(
                coso, sino, sin_d, azrh, cosod, max_solar_altitude
            )

        sin_alrs = np.sin(alrs)

        # Calculate level-plain sunrise/set hour angle
        tano = sino / coso
        tan_d = sin_d / cos_d
        tanod = tano * tan_d
        horizontal_hour_angle = np.arccos(-tanod)
        sinhro = np.sin(horizontal_hour_angle)

        # Calculate total potential shade on level-plain
        total_shade = 2.0 * (
            (horizontal_hour_angle * sinod) + (sinhro * cosod)
        )
        if total_shade < 0.0:
            total_shade = NEARZERO

        hrso = horizontal_hour_angle
        azso = level_sunset_azimuth
        totsh = total_shade

        if azrh <= (-azso):
            hrrs = -hrso
        elif azrh >= azso:
            hrrs = hrso
        elif azrh == 0.0:
            hrrs = 0.0
        else:
            temp = (sin_alrs - sinod) / cosod

            if temp > 1.0:
                temp = 1.0
            elif temp < -1.0:
                temp = -1.0

            if azrh > 0.0:
                hrrs = np.abs(np.arccos(temp))
            else:
                hrrs = -(np.abs(np.arccos(temp)))

        if alte == 0.0 and altw == 0.0:
            hrsr = -hrso
            hrss = hrso
            sti = 0.0
            svi = self._rprnvg(
                hrsr,
                hrrs,
                hrss,
                sino,
                coso,
                sin_d,
                cosod,
                sinod,
                vce,
                voe,
                vhe,
                azrh,
                vdemx,
                vdemn,
                seg_width,
                summer_flag,
                vcw,
                vow,
                vhw,
                vdwmx,
                vdwmn,
            ) / (seg_width * totsh)
        else:
            if -azso <= azrh:
                altop_0 = alte
                aztop_0 = azso * (alte / HALF_PI) - azso
            else:
                altop_0 = altw
                aztop_0 = azso * (altw / HALF_PI) - azso

            if altop_0 == 0.0:
                hrsr = -hrso
            else:
                azmn = -azso
                azmx = 0.0
                azs = aztop_0
                altmx = altop_0
                almn = 0.0
                almx = 1.5708
                als = self._solalt(coso, sino, sin_d, azs, almn, almx)
                azs, als, hrs = self._snr_sst(
                    coso,
                    sino,
                    sin_d,
                    altmx,
                    almn,
                    almx,
                    azmn,
                    azmx,
                    azs,
                    als,
                    azrh,
                )
                hrsr = hrs

            if azso <= azrh:
                altop_1 = alte
                aztop_1 = azso - azso * (alte / HALF_PI)
            else:
                altop_1 = altw
                aztop_1 = azso - azso * (altw / HALF_PI)

            if altop_1 == 0.0:
                hrss = hrso
            else:
                azmn = 0.0
                azmx = azso
                azs = aztop_1
                altmx = altop_1
                almn = 0.0
                almx = 1.5708
                als = self._solalt(coso, sino, sin_d, azs, almn, almx)
                azs, als, hrs = self._snr_sst(
                    coso,
                    sino,
                    sin_d,
                    altmx,
                    almn,
                    almx,
                    azmn,
                    azmx,
                    azs,
                    als,
                    azrh,
                )
                hrss = hrs

            if hrrs < hrsr:
                hrrh = hrsr
            elif hrrs > hrss:
                hrrh = hrss
            else:
                hrrh = hrrs

            seg_daylight = (hrss - hrsr) * RADTOHOUR
            sti = 1.0 - (
                (
                    ((hrss - hrsr) * sinod)
                    + ((math.sin(hrss) - math.sin(hrsr)) * cosod)
                )
                / (totsh)
            )
            svi = (
                self._rprnvg(
                    hrsr,
                    hrrh,
                    hrss,
                    sino,
                    coso,
                    sin_d,
                    cosod,
                    sinod,
                    vce,
                    voe,
                    vhe,
                    azrh,
                    vdemx,
                    vdemn,
                    seg_width,
                    summer_flag,
                    vcw,
                    vow,
                    vhw,
                    vdwmx,
                    vdwmn,
                )
            ) / (seg_width * totsh)

        if sti < 0.0:
            sti = 0.0
        if sti > 1.0:
            sti = 1.0
        if svi < 0.0:
            svi = 0.0
        if svi > 1.0:
            svi = 1.0

        shade = sti + svi

        return shade, svi

    def _solalt(self, coso, sino, sin_d, az, almn, almx):
        """Determine solar altitude from trigonometric parameters.

        This is the solalt function from PRMS.
        """
        maxiter_sntemp = int(self.maxiter_sntemp)

        if abs(abs(az) - HALF_PI) < NEARZERO:
            temp = abs(sin_d / sino)
            if temp > 1.0:
                temp = 1.0
            al = np.arcsin(temp)
        else:
            cosaz = np.cos(az)
            a = sino / (cosaz * coso)
            b = sin_d / (cosaz * coso)

            al = (almn + almx) / 2.0
            kount = 0
            fal = np.cos(al) - (a * np.sin(al)) + b
            delal = fal / (-np.sin(al) - (a * np.cos(al)))

            for kount in range(1, int(maxiter_sntemp + 1)):
                if abs(fal) < NEARZERO:
                    break
                if abs(delal) < NEARZERO:
                    break
                alold = al
                cosal = np.cos(al)
                sinal = np.sin(al)
                fal = cosal - (a * sinal) + b
                fpal = -sinal - (a * cosal)
                if kount <= 3:
                    delal = fal / fpal
                else:
                    fppal = b - fal
                    delal = (2.0 * fal * fpal) / (
                        (2.0 * fpal * fpal) - (fal * fppal)
                    )
                al = al - delal
                if al < almn:
                    al = (alold + almn) / 2.0
                if al > almx:
                    al = (alold + almx) / 2.0

        return al

    def _snr_sst(
        self,
        coso,
        sino,
        sin_d,
        alt,
        almn,
        almx,
        azmn,
        azmx,
        azs,
        als,
        azrh,
    ):
        """Determine local solar sunrise/set azimuth, altitude, and hour angle.

        This is the snr_sst function from PRMS.
        """
        maxiter_sntemp = int(self.maxiter_sntemp)

        # Trig function for local altitude
        tanalt = np.tan(alt)
        tano = sino / coso
        f = 999999.0
        delazs = 9999999.0
        g = 99999999.0
        delals = 99999999.0

        # Begin Newton-Raphson solution
        for count in range(int(maxiter_sntemp)):
            if abs(delazs) < NEARZERO:
                break
            if abs(delals) < NEARZERO:
                break
            if abs(f) < NEARZERO:
                break
            if abs(g) < NEARZERO:
                break

            cosazs = np.cos(azs)
            sinazs = np.sin(azs)

            sinazr = abs(np.sin(azs - azrh))
            if ((azs - azrh) <= 0.0 and (azs - azrh) <= -PI) or (
                (azs - azrh) > 0.0 and (azs - azrh) <= PI
            ):
                cosazr = np.cos(azs - azrh)
            else:
                cosazr = -np.cos(azs - azrh)

            cosals = np.cos(als)
            if cosals < NEARZERO:
                cosals = NEARZERO
            sinals = np.sin(als)
            tanals = sinals / cosals

            # Functions of Azs & Als
            f = cosazs - (((sino * sinals) - sin_d) / (coso * cosals))
            g = tanals - (tanalt * sinazr)

            # First partials derivatives of f & g
            fazs = -sinazs
            fals = ((tanals * (sin_d / coso)) - (tano / cosals)) / cosals
            gazs = -tanalt * cosazr
            gals = 1.0 / (cosals * cosals)

            # Jacobian
            xjacob = (fals * gazs) - (fazs * gals)

            # Delta corrections
            delazs = ((f * gals) - (g * fals)) / xjacob
            delals = ((g * fazs) - (f * gazs)) / xjacob

            # New values of Azs & Als
            azs = azs + delazs
            als = als + delals

            # Check for limits
            if azs < (azmn + NEARZERO):
                azs = azmn + NEARZERO
            if azs > (azmx - NEARZERO):
                azs = azmx - NEARZERO
            if als < (almn + NEARZERO):
                als = almn + NEARZERO
            if als > (almx - NEARZERO):
                als = almx - NEARZERO

        # Ensure azimuth remains between -PI & PI
        if azs < -PI:
            azs = azs + PI
        elif azs > PI:
            azs = azs - PI

        # Determine local sunrise/set hour angle
        sinals = np.sin(als)
        temp = (sinals - (sino * sin_d)) / (coso * np.cos(np.arcsin(sin_d)))

        if temp > 1.0:
            temp = 1.0
        elif temp < -1.0:
            temp = -1.0

        if azs > 0.0:
            hrs = np.abs(np.arccos(temp))
        else:
            hrs = -(np.abs(np.arccos(temp)))

        return azs, als, hrs

    def _rprnvg(
        self,
        hrsr,
        hrrs,
        hrss,
        sino,
        coso,
        sin_d,
        cosod,
        sinod,
        vce,
        voe,
        vhe,
        azrh,
        vdemx,
        vdemn,
        seg_width,
        summer_flag,
        vcw,
        vow,
        vhw,
        vdwmx,
        vdwmn,
    ):
        """Compute riparian vegetation shade.

        This is the rprnvg function from PRMS.
        """
        # Determine seasonal shade
        if hrsr == hrss:
            svri = 0.0
            svsi = 0.0
        else:
            svri = 0.0
            if hrsr < hrrs:
                vco = (vce / 2.0) - voe
                delhsr = hrrs - hrsr

                for n in range(15):
                    hrs = hrsr + (EPSLON[n] * delhsr)
                    coshrs = np.cos(hrs)
                    sinhrs = np.sin(hrs)
                    temp = sinod + (cosod * coshrs)
                    if temp > 1.0:
                        temp = 1.0
                    als = np.arcsin(temp)
                    cosals = np.cos(als)
                    sinals = np.sin(als)
                    if sinals == 0.0:
                        sinals = NEARZERO

                    temp = ((sino * sinals) - sin_d) / (coso * cosals)

                    if temp > 1.0:
                        temp = 1.0
                    elif temp < -1.0:
                        temp = -1.0

                    azs = np.arccos(temp)
                    if azs < 0.0:
                        azs = HALF_PI - azs
                    if hrs < 0.0:
                        azs = -azs

                    bs = (
                        (vhe * (cosals / sinals)) * abs(np.sin(azs - azrh))
                    ) + vco
                    if bs < 0.0:
                        bs = 0.0
                    if bs > seg_width:
                        bs = seg_width

                    if summer_flag == 1:
                        svri += vdemx * bs * sinals * WEIGHT[n]
                    else:
                        svri += vdemn * bs * sinals * WEIGHT[n]

                svri = svri * delhsr

            svsi = 0.0
            if hrss > hrrs:
                vco = (vcw / 2.0) - vow
                delhss = hrss - hrrs

                for n in range(15):
                    hrs = hrrs + (EPSLON[n] * delhss)
                    coshrs = np.cos(hrs)
                    sinhrs = np.sin(hrs)
                    temp = sinod + (cosod * coshrs)
                    if temp > 1.0:
                        temp = 1.0
                    als = np.arcsin(temp)
                    cosals = np.cos(als)
                    sinals = np.sin(als)
                    if sinals == 0.0:
                        sinals = NEARZERO

                    temp = ((sino * sinals) - sin_d) / (coso * cosals)
                    if temp > 1.0:
                        temp = 1.0
                    elif temp < -1.0:
                        temp = -1.0

                    azs = np.arccos(temp)
                    if azs < 0.0:
                        azs = HALF_PI - azs
                    if hrs < 0.0:
                        azs = -azs

                    bs = (
                        (vhw * (cosals / sinals)) * abs(np.sin(azs - azrh))
                    ) + vco
                    if bs < 0.0:
                        bs = 0.0
                    if bs > seg_width:
                        bs = seg_width

                    if summer_flag == 1:
                        svsi += vdwmx * bs * sinals * WEIGHT[n]
                    else:
                        svsi += vdwmn * bs * sinals * WEIGHT[n]

                svsi = svsi * delhss

        return svri + svsi

    def _compute_inflow(self, seg_idx: int) -> float:
        """Compute total inflow to a segment.

        Args:
            seg_idx: Segment index

        Returns:
            Total inflow (cfs)
        """
        # Sum upstream flows
        upstream_flow = 0.0
        for kk in range(self.upstream_count[seg_idx]):
            up_idx = self.upstream_idx[seg_idx, kk]
            if not np.isnan(self.seg_tave_water[up_idx]):
                upstream_flow += self.seg_outflow[up_idx]

        # Add lateral inflow
        total_inflow = upstream_flow + self.seg_lateral_inflow[seg_idx]

        return total_inflow

    def _compute_water_temp(self, seg_idx: int, svi: float) -> None:
        """Compute water temperature for a segment.

        Args:
            seg_idx: Segment index
            svi: Vegetation shade index
        """
        # Skip segments marked as permanently invalid (matches Fortran line 887, 894)
        if self.seg_tave_water[seg_idx] < -99.0:
            # Never has flow - skip all calculations
            return

        if self.seginc_swrad[seg_idx] < -99.0:
            # Never has data - mark and skip
            self.seg_tave_water[seg_idx] = -99.9
            return

        # Check for no-flow conditions (matches Fortran check for seg_outflow <= 0)
        if self.seg_outflow[seg_idx] <= 0.0:
            self.seg_tave_water[seg_idx] = NOFLOW_TEMP
            return

        # Get upstream flow
        upstream_flow = 0.0
        for kk in range(self.upstream_count[seg_idx]):
            up_idx = self.upstream_idx[seg_idx, kk]
            if not np.isnan(self.seg_tave_water[up_idx]):
                upstream_flow += self.seg_outflow[up_idx]

        lateral_flow = self.seg_lateral_inflow[seg_idx]

        # Check if we have any inflow
        if upstream_flow < NEARZERO and lateral_flow < NEARZERO:
            self.seg_tave_water[seg_idx] = NOFLOW_TEMP
            return

        # Compute mixed inlet temperature
        t_in = self._compute_mixed_inlet_temp(
            seg_idx, upstream_flow, lateral_flow
        )

        if np.isnan(t_in):
            self.seg_tave_water[seg_idx] = NOFLOW_TEMP
            return

        # Compute equilibrium temperature using full energy balance
        te, ak1, ak2 = self._equilb(seg_idx, t_in, svi)

        # Compute final temperature using twavg function
        qlat = lateral_flow * CFS_TO_CMS
        qup = upstream_flow  # Upstream flow only (CFS)
        tl_avg = self.seg_tave_lat[seg_idx]

        self.seg_tave_water[seg_idx] = self._twavg(
            qup, t_in, qlat, tl_avg, te, ak1, ak2, seg_idx
        )

        return

    def _compute_mixed_inlet_temp(
        self, seg_idx: int, upstream_flow: float, lateral_flow: float
    ) -> float:
        """Compute mixed inlet temperature from upstream and lateral sources.

        Args:
            seg_idx: Segment index
            upstream_flow: Flow from upstream segments (cfs)
            lateral_flow: Lateral inflow (cfs)

        Returns:
            Mixed inlet temperature (degC)
        """
        upstream_ready = upstream_flow > 0.0 and not np.isnan(
            self.seg_tave_upstream[seg_idx]
        )
        lateral_ready = lateral_flow > 0.0 and not np.isnan(
            self.seg_tave_lat[seg_idx]
        )

        if not upstream_ready and not lateral_ready:
            return np.nan
        elif upstream_ready and not lateral_ready:
            return self.seg_tave_upstream[seg_idx]
        elif lateral_ready and not upstream_ready:
            return self.seg_tave_lat[seg_idx]
        else:
            # Both sources present - compute weighted average
            t_up = self.seg_tave_upstream[seg_idx]
            t_lat = self.seg_tave_lat[seg_idx]
            return (t_up * upstream_flow + t_lat * lateral_flow) / (
                upstream_flow + lateral_flow
            )

    def _equilb(self, seg_idx: int, t_o: float, svi: float):
        """Compute equilibrium temperature using full energy balance.

        This is the equilb function from PRMS.

        Args:
            seg_idx: Segment index
            t_o: Initial temperature (degC)
            svi: Vegetation shade index

        Returns:
            te: Equilibrium temperature (degC)
            ak1: First-order thermal exchange coefficient
            ak2: Second-order thermal exchange coefficient
        """
        # Local Variables
        taabs = float(t_o + ZERO_C)

        vp_sat = 6.108 * np.exp(17.26939 * t_o / (t_o + 237.3))

        # Convert units and set up parameters
        q_init = max(self._seg_inflow[seg_idx] * CFS_TO_CMS, NEARZERO)

        sw_power = 11.63 / 24.0 * float(self.seginc_swrad[seg_idx])

        # If humidity is 1.0, there is a divide by zero below
        foo = min(self.seg_humid[seg_idx], 0.99)

        # Compute atmospheric pressure based on segment elevation
        press = 1013.0 - (0.1055 * self.seg_elev[seg_idx])

        bow_coeff = (0.00061 * press) / (vp_sat * (1.0 - foo))
        evap = float(self.seg_potet[seg_idx] * MPS_CONVERT)

        # Heat flux components
        # Ha: atmospheric-emitted longwave radiation
        ha = (
            (3.354939e-8 + 2.74995e-9 * np.sqrt(foo * vp_sat))
            * (1.0 - self.seg_shade[seg_idx])
            * (1.0 + (0.17 * (self.seg_ccov[seg_idx] ** 2)))
        ) * (taabs**4)

        # Hf: heat dissipated from potential energy by friction
        hf = (
            9805.0
            * (q_init / self.seg_flow_width[seg_idx])
            * self.seg_slope[seg_idx]
        )

        # Hs: net flux from shortwave solar radiation
        hs = (1.0 - self.seg_shade[seg_idx]) * sw_power * (1.0 - self.albedo)

        # Hv: longwave radiation emitted by riparian vegetation
        hv = 5.24e-8 * svi * (taabs**4)

        # Determine equilibrium coefficients
        del_ht = 2.36e06
        ltnt_ht = 2495.0e06

        b = (
            bow_coeff * evap * (ltnt_ht + (del_ht * t_o))
            + AKZ
            - (del_ht * evap)
        )
        c = bow_coeff * del_ht * evap
        d = (ha + hv + hf + hs) + (
            ltnt_ht * evap * ((bow_coeff * t_o) - 1.0)
            + (self.seg_tave_gw[seg_idx] * AKZ)
        )

        # Determine equilibrium temperature & 1st order thermal exchange coef
        ted = t_o
        ted, ak1d = self._teak1(A, b, c, d, ted)

        # Determine 2nd order thermal exchange coefficient
        hnet = (A * ((t_o + ZERO_C) ** 4)) + (b * t_o) - (c * (t_o**2.0)) - d
        delt = t_o - ted

        if abs(delt) < NEARZERO:
            ak2d = 0.0
        else:
            ak2d = ((delt * ak1d) - hnet) / (delt**2)

        return ted, ak1d, ak2d

    def _teak1(self, a_coef, b_coef, c_coef, d_coef, teq):
        """Solve for equilibrium temperature using Newton-Raphson iteration.

        This is the teak1 function from PRMS.

        Args:
            a_coef, b_coef, c_coef, d_coef: Coefficients
            teq: Initial guess for equilibrium temperature

        Returns:
            teq: Equilibrium temperature (degC)
            ak1c: First-order thermal exchange coefficient
        """
        maxiter_sntemp = int(self.maxiter_sntemp)

        # Local variables
        fte = 99999.0
        delte = 99999.0
        kount = 0

        # Begin Newton iteration solution for TE
        while kount < maxiter_sntemp:
            if np.abs(fte) < TOLRN:
                break
            if abs(delte) < TOLRN:
                break
            teabs = teq + ZERO_C
            fte = (
                (a_coef * (teabs**4.0))
                + (b_coef * teq)
                - (c_coef * (teq**2.0))
                - d_coef
            )
            fpte = (
                (4.0 * a_coef * (teabs**3.0)) + b_coef - (2.0 * c_coef * teq)
            )
            delte = fte / fpte
            teq = teq - delte
            kount += 1

        # Determine 1st thermal exchange coefficient
        ak1c = (
            (4.0 * a_coef * ((teq + ZERO_C) ** 3.0))
            + b_coef
            - (2.0 * c_coef * teq)
        )

        return teq, ak1c

    def _twavg(
        self,
        qup,
        t0,
        qlat,
        tl_avg,
        te,
        ak1,
        ak2,
        seg_idx,
    ):
        """Compute average water temperature with lateral inflows.

        This is the twavg function from PRMS.

        Args:
            qup: Upstream flow (cfs)
            t0: Inlet temperature (degC)
            qlat: Lateral flow (cms)
            tl_avg: Lateral flow temperature (degC)
            te: Equilibrium temperature (degC)
            ak1: First-order thermal exchange coefficient
            ak2: Second-order thermal exchange coefficient
            seg_idx: Segment index

        Returns:
            tw: Average water temperature (degC)
        """
        # Determine equation parameters
        q_init = float(qup * CFS_TO_CMS)
        ql = float(qlat)
        width = self.seg_flow_width[seg_idx]
        length = self.seg_length[seg_idx]

        # Local Variables
        tep = 0.0
        b = 0.0
        r = 0.0
        rexp = 0.0
        tw = 0.0
        delt = 0.0
        denom = 0.0

        if ql <= NEARZERO:
            # Zero lateral flow
            tep = te
            b = (ak1 * width) / 4182.0e03
            rexp = -1.0 * (b * length) / q_init
            r = np.exp(rexp)

        elif ql < 0.0:
            # Losing stream (should not happen in PRMS)
            tep = te
            b = (ql / length) + ((ak1 * width) / 4182.0e03)
            rexp = (ql - (b * length)) / ql
            r = 1.0 + (ql / q_init)
            r = r**rexp

        elif ql > NEARZERO and q_init <= NEARZERO:
            tep = te
            b = (ak1 * width) / 4182.0e03
            rexp = -1.0 * (b * length) / ql
            r = np.exp(rexp)

        else:
            b = (ql / length) + ((ak1 * width) / 4182.0e03)
            tep = (
                ((ql / length) * tl_avg) + (((ak1 * width) / (4182.0e03)) * te)
            ) / b

            if ql > 0.0:
                rexp = -b / (ql / length)
            else:
                rexp = 0.0

            if q_init < NEARZERO:
                r = 2.0
            else:
                r = 1.0 + (ql / q_init)
            r = r**rexp

        # Determine water temperature
        delt = tep - t0
        denom = 1.0 + (ak2 / ak1) * delt * (1.0 - r)

        if denom < 0.0:
            denom = np.abs(denom)

        tw = tep - (delt * r / denom)
        if tw < 0.0:
            tw = 0.0

        return tw
