import pathlib as pl
from typing import Literal
from warnings import warn

import networkx as nx
import numba as nb
import numpy as np

from ..base.adapter import adaptable
from ..base.conservative_process import ConservativeProcess
from ..base.control import Control
from ..constants import cfs_to_cms, nan, nearzero, zero
from ..parameters import Parameters
from .prms_stream_shade import PRMSStreamShade

# Constants from PRMS
NEARZERO = nearzero
PI = np.pi
CFS_TO_CMS = cfs_to_cms
NOFLOW_TEMP = -98.9
DAYS_YR = 365.25
MAX_DAYS_PER_YEAR = 366
ZERO_C = 273.15
TOLRN = 1.0e-4
AKZ = 1.65
A = 5.40e-8
MPS_CONVERT = 2.93981481e-07

# Physical constants for energy calculations
WATER_DENSITY = 1000.0  # kg/m³
SPECIFIC_HEAT_WATER = 4182.0  # J/(kg·°C)
LATENT_HEAT_VAPORIZATION = 2495.0e06  # J/m³


class PRMSStreamTemp(ConservativeProcess):
    """PRMS stream temperature.

    A representation of stream temperature from PRMS. with two structural
    differences. This class uses:
    - PRMSHydraulicGeometry as an upstream process to get the hydraulic
      geometry variables (which were renamed seg_flow_*)
    - PRMSStreamShade as a shade representation to be passed/composed on
      initialization.

    Implementation based on PRMS 5.2.1.1 with theoretical documentation given
    by:

    `Markstrom, Steven L. P2S -- Coupled simulation with the
    Precipitation-Runoff Modeling System (PRMS) and the Stream Temperature
    Network (SNTemp) Models.
    No. 2012-1116. US Geological Survey, 2012.
    <https://pubs.usgs.gov/publication/ofr20121116>`__

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
        seg_flow_area: Flow-dependent cross-sectional area from
            PRMSHydraulicGeometry
        seg_flow_velocity: Flow-dependent velocity from
            PRMSHydraulicGeometry
        stream_shade: PRMSStreamShade instance (Dynamic or Constant)
        budget_type: one of ["defer", None, "warn", "error"]
        verbose: Print extra information or not?
        use_vectorized_shade: Use vectorized shade computation for all
            segments at once (default True)
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
        stream_shade: PRMSStreamShade,
        budget_type: Literal["defer", None, "warn", "error"] = "defer",
        verbose: bool = False,
        use_vectorized_shade: bool = True,
        track_energy_fluxes: bool = True,
    ) -> None:
        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
            budget_type=budget_type,
        )
        self.name = "PRMSStreamTemp"

        self._set_inputs(locals())
        self._set_options(locals())

        # Store the composed shade computer
        self.stream_shade = stream_shade

        # Store vectorization preference
        self.use_vectorized_shade = use_vectorized_shade

        # Store energy flux tracking preference
        self.track_energy_fluxes = track_energy_fluxes

        # Store energy flux variable names for consistency checks
        self._energy_flux_vars = [
            "heat_upstream",
            "heat_lateral",
            "solar_radiation",
            "atmospheric_longwave",
            "friction_heat",
            "groundwater_conduction",
            "heat_outflow",
            "longwave_emission",
            "longwave_vegetation",
            "evaporative_cooling",
            "convective_exchange",
        ]

        self._set_budget(basis="unit", quantity="energy")
        self._initialize_stream_temp()

        # Consistency checks for energy flux tracking
        if not track_energy_fluxes:
            # Check 1: budget_type must be None if not tracking energy fluxes
            if budget_type is not None:
                msg = (
                    "Inconsistent options: track_energy_fluxes=False "
                    f"requires budget_type=None, but "
                    f"budget_type={budget_type!r}"
                )
                raise ValueError(msg)

            # Check 2: Set energy flux variables to None if not tracking
            for var in self._energy_flux_vars:
                if hasattr(self, var):
                    setattr(self, var, None)

        return

    def initialize_netcdf(
        self,
        output_dir: [str, pl.Path] = None,
        separate_files: bool = None,
        budget_args: dict = None,
        output_vars: list = None,
        extra_coords: dict = None,
        addtl_output_vars: list = None,
    ) -> None:
        """Initialize NetCDF output with energy flux tracking checks.

        This method overrides the parent class to add consistency checks
        for energy flux tracking.

        Args:
            output_dir: base directory path or NetCDF file path if
                separate_files is True
            separate_files: boolean indicating if storage component output
                variables should be written to a separate file for each
                variable
            budget_args: arguments to pass to budget initialization
            output_vars: list of variable names to output
            extra_coords: extra coordinates to add to the output
            addtl_output_vars: additional output variables

        Returns:
            None

        """
        # Set output_vars appropriately based on tracking
        if output_vars is None:
            if self.track_energy_fluxes:
                # Include all variables
                output_vars = self.get_variables()
            else:
                # Exclude energy flux variables
                output_vars = [
                    v
                    for v in self.get_variables()
                    if v not in self._energy_flux_vars
                ]
        else:
            # Check if energy flux variables are requested when not tracking
            if not self.track_energy_fluxes:
                conflicting_vars = set(self._energy_flux_vars) & set(
                    output_vars
                )
                if conflicting_vars:
                    # Warn and filter out the conflicting variables
                    msg = (
                        "Variables omitted from NetCDF output because "
                        "PRMSStreamTemp energy fluxes are not tracked: "
                        f"{sorted(conflicting_vars)}"
                    )
                    warn(msg)
                    # Remove conflicting variables from output_vars
                    output_vars = [
                        v for v in output_vars if v not in conflicting_vars
                    ]

        # Call parent class initialize_netcdf
        super().initialize_netcdf(
            output_dir=output_dir,
            separate_files=separate_files,
            budget_args=budget_args,
            output_vars=output_vars,
            extra_coords=extra_coords,
            addtl_output_vars=addtl_output_vars,
        )

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
            # Energy flux variables (W)
            "heat_upstream": 0.0,
            "heat_lateral": 0.0,
            "solar_radiation": 0.0,
            "atmospheric_longwave": 0.0,
            "friction_heat": 0.0,
            "groundwater_conduction": 0.0,
            "heat_outflow": 0.0,
            "longwave_emission": 0.0,
            "longwave_vegetation": 0.0,
            "evaporative_cooling": 0.0,
            "convective_exchange": 0.0,
        }

    @staticmethod
    def get_mass_budget_terms() -> dict:
        # Temperature is not a mass, so no budget terms
        return {
            "inputs": [],
            "outputs": [],
            "storage_changes": [],
        }

    @staticmethod
    def get_energy_budget_terms() -> dict:
        """Get energy budget terms for stream temperature.

        Returns:
            Dictionary with inputs, outputs, and storage_changes for
            energy budget.

        Notes:
            Energy fluxes are computed in Watts (J/s). The budget tracks:
            - Advective heat transport (upstream, lateral, outflow)
            - Surface energy exchange (solar, longwave, evaporation)
            - Internal sources (friction, groundwater conduction)

            Storage changes are empty because the kinematic wave assumption
            means water storage is constant - only temperature (and thus
            heat content) changes, which is captured by the balance of
            inputs and outputs.
        """
        return {
            "inputs": [
                "heat_upstream",  # Advective heat from upstream (W)
                "heat_lateral",  # Advective heat from lateral (W)
                "solar_radiation",  # Net shortwave radiation (W)
                "atmospheric_longwave",  # Atmospheric LW radiation (W)
                "friction_heat",  # Friction heating (W)
                "groundwater_conduction",  # Heat from groundwater (W)
            ],
            "outputs": [
                "heat_outflow",  # Advective heat leaving (W)
                "longwave_emission",  # LW radiation emitted (W)
                "longwave_vegetation",  # LW from vegetation (W)
                "evaporative_cooling",  # Latent heat loss (W)
                "convective_exchange",  # Sensible heat exchange (W)
            ],
            "storage_changes": [
                # Empty - kinematic wave assumption (no storage change)
            ],
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

        # Initialize circular buffers for temperature averaging if not
        # already done
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
        self.gw_index[:] = 0
        self.ss_index[:] = 0

        # Initialize running sum arrays for temperature averaging
        self.gw_sum = np.zeros(self.nsegment, dtype=np.float64)
        self.ss_sum = np.zeros(self.nsegment, dtype=np.float64)

        self.gw_sum[:] = 0.0
        self.ss_sum[:] = 0.0

        return

    def _initialize_stream_temp(self) -> None:
        """Initialize stream temperature data structures."""
        self.nsegment = self._params.dims["nsegment"]
        self.nhru = self._params.dims["nhru"]

        # Extract scalar parameters (dimension "one")
        # These are stored as arrays but should be scalars
        self.maxiter_sntemp = float(self.maxiter_sntemp[0])
        self.albedo = float(self.albedo[0])
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

        # Compute segment HRU areas (sum of HRU areas contributing to
        # each segment)
        self.segment_hruarea = np.zeros(self.nsegment, dtype=np.float64)
        for j in range(self.nhru):
            seg_idx = self.hru_segment[j]
            if seg_idx > 0:
                i = seg_idx - 1
                self.segment_hruarea[i] += self.hru_area[j]

        # Compute upstream segment information
        self._compute_upstream_info()

        # Compute segment_up - the single immediate upstream segment
        # (matches Fortran Segment_up)
        # Fortran's Segment_up is computed by iterating segments and
        # assigning Segment_up(toseg) = j
        # This means when multiple segments flow into one, it keeps the
        # LAST one (highest j)
        # So we need to find the last upstream in segment numbering order
        segment_up = np.zeros(self.nsegment, dtype=np.int32)
        for j in range(self.nsegment):
            toseg = self.tosegment[j]
            if toseg > 0:
                # toseg is 1-based, convert to 0-based
                segment_up[toseg - 1] = j
        # Note: segment_up[i] = 0 means no upstream (default from zeros
        # initialization)

        # Save segment_up as instance variable (needed for routing
        # aggregation logic)
        self.segment_up = segment_up

        # Compute seg_close for segments without HRUs (matches Fortran
        # line 529-570)
        # Initialize seg_close = segment_up (Fortran line 529)
        self.seg_close = np.copy(segment_up)

        # Now update seg_close for segments without HRUs (Fortran line 530-570)
        for jj in range(self.nsegment):
            i = self.segment_order[jj]

            # Only modify seg_close for segments without HRUs
            # (Fortran line 539)
            if self.segment_hruarea[i] <= NEARZERO:
                # If no upstream segment (Fortran line 541)
                if self.segment_up[i] == 0:
                    # Try downstream segment (Fortran line 542-543)
                    if self.tosegment[i] > 0:
                        self.seg_close[i] = (
                            self.tosegment[i] - 1
                        )  # Convert to 0-based
                    else:
                        # No upstream or downstream - use previous/next in
                        # order (Fortran line 544-549)
                        if jj > 0:
                            self.seg_close[i] = self.segment_order[jj - 1]
                        else:
                            self.seg_close[i] = self.segment_order[jj + 1]

                # Check if seg_close points to invalid segment
                # (Fortran line 551-563)
                # If elevation is exactly 30000 (invalid marker), find a
                # different segment
                if self.seg_elev[self.seg_close[i]] == 30000.0:
                    found = False
                    # Find first segment with HRUs in forward order
                    # (Fortran line 553-558)
                    for k in range(jj + 1, self.nsegment):
                        ii = self.segment_order[k]
                        if self.segment_hruarea[ii] > NEARZERO:
                            self.seg_close[i] = ii
                            found = True
                            break

                    # If not found, use previous segment in order
                    # (Fortran line 560-565)
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

        # Precompute solar geometry for each day of year
        self._precompute_solar_geometry()

        return

    def _precompute_solar_geometry(self) -> None:
        """Precompute solar declination for each day of year."""
        # Solar declination for each day of year (vectorized)
        jdays = np.arange(MAX_DAYS_PER_YEAR)
        k = jdays + 1  # Convert to 1-based day of year
        self.declination = 0.40928 * np.cos(
            ((2.0 * PI) / DAYS_YR) * (172.0 - k)
        )  # radians

        return

    def _compute_segment_order(self) -> None:
        """Compute topologically sorted segment order for calculation.

        Uses NetworkX for topological sorting, matching the approach in
        PRMSChannel.
        """
        # Build connectivity list (tosegment is 1-based, convert to 0-based)
        connectivity = []
        for iseg in range(self.nsegment):
            toseg = self.tosegment[iseg] - 1  # Convert to 0-based
            if toseg >= 0:  # -1 means outlet in 0-based indexing
                connectivity.append((iseg, toseg))

        # Use NetworkX for topological sort
        if self.nsegment > 1 and len(connectivity) > 0:
            graph = nx.DiGraph()
            graph.add_edges_from(connectivity)
            segment_order = list(nx.topological_sort(graph))
        else:
            segment_order = list(range(self.nsegment))

        self.segment_order = np.array(segment_order, dtype=np.int32)

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
        doy = self.control.current_doy - 1

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

        # Initialize seg_tave_upstream to 0.0 each timestep
        # (matches Fortran line 463)
        # Segments that are skipped will keep this 0.0 value
        self.seg_tave_upstream[:] = 0.0

        # Compute shade - vectorized or loop-based depending on flag
        if self._use_vectorized_shade:
            # VECTORIZED: Compute shade for all segments at once
            self.seg_shade[:], seg_svi_all = self._stream_shade.compute_all(
                declination, summer_flag, self.seg_flow_width
            )
        else:
            # LOOP-BASED: Compute shade one segment at a time (original method)
            seg_svi_all = np.zeros(self.nsegment, dtype=np.float64)

        # Compute water temperature for each segment (must be done in
        # segment_order)
        # Don't reset segments marked as -99.9 (never have flow)
        for jj in self.segment_order:
            if self.seg_tave_water[jj] >= -99.0:
                self.seg_tave_water[jj] = np.nan

        for jj in self.segment_order:
            # Skip segments marked as never having flow
            # (matches Fortran cycle at line 887)
            if self.seg_tave_water[jj] < -99.0:
                continue

            # Compute upstream temperature
            self._compute_upstream_temp(jj)

            # Compute shade and get svi
            if self._use_vectorized_shade:
                # Get svi for this segment (already computed above)
                svi = seg_svi_all[jj]
            else:
                # Compute shade for this segment
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

        # First: Process seginc_swrad in numerical order
        # Divide radiation and PET by segment HRU area to get averages
        # Process in numerical order to match routing.f90 (line 741-810)
        for i in range(self.nsegment):
            if self.segment_hruarea[i] > NEARZERO:
                self.seginc_swrad[i] /= self.segment_hruarea[i]

            else:
                # Segment has no HRUs - search upstream then downstream
                # (matches routing.f90 line 746-805)
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
                        # Found segment with HRUs - copy values
                        # (already averaged)
                        self.seginc_swrad[i] = self.seginc_swrad[this_seg]
                        found = True
                        break

                # If not found upstream, search downstream
                # (routing.f90 line 776-800)
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
                            # Found segment with HRUs - copy values
                            # (already averaged)
                            self.seginc_swrad[i] = self.seginc_swrad[this_seg]
                            found = True
                            break

                # If still not found, set to invalid marker
                # (routing.f90 line 803-805)
                if not found:
                    self.seginc_swrad[i] = -99.9

        return

    def _compute_seg_potet(self) -> None:
        """Compute seg_potet using stream_temp.f90 logic.

        This matches stream_temp.f90 lines 807-848, using segment_order
        and seg_close.
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
                # Segment has no HRUs - use seg_close
                # (stream_temp.f90 line 817)
                close_seg = self.seg_close[i]

                self.seg_potet[i] = self.seg_potet[close_seg]

        return

    # Note: _compute_hydraulic_geometry removed - now handled by
    # PRMSHydraulicGeometry process

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
        elif comp_type == "ss":
            tau = self.ss_tau[seg_idx]
            index = self.ss_index[seg_idx]
            silo = self.ss_silo
            sum_array = self.ss_sum
        else:
            raise ValueError(f"Invalid component type: {comp_type}")

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
        elif comp_type == "ss":
            self.seg_tave_ss[seg_idx] = avg_temp
            self.ss_index[seg_idx] = (index + 1) % int(tau)
        else:
            raise ValueError(f"Invalid component type: {comp_type}")

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

        Args:
            seg_lateral_inflow: Total lateral inflow to segment (cfs)
            seginc_sroff: Surface runoff component (cfs)
            seginc_ssflow: Subsurface flow component (cfs)
            seginc_gwflow: Groundwater flow component (cfs)
            melt_temp: Snowmelt temperature (degC)
            tave_gw: Groundwater temperature (degC)
            tave_air: Air temperature (degC)
            tave_ss: Subsurface temperature (degC)
            melt: Snowmelt (inches)
            rain: Rainfall (inches)

        Returns:
            tl_avg: Weighted average lateral inflow temperature (degC)
            qlat: Lateral inflow (cms)
        """
        return _lat_inflow(
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
        )

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
        """Compute shade using the composed stream_shade object.

        NOTE: This method is used when self._use_vectorized_shade=False.
        For vectorized computation, see compute_all in stream_shade.

        Args:
            seg_idx: Segment index
            declination: Solar declination (radians)
            summer_flag: 1 for summer, 0 for winter

        Returns:
            svi: Vegetation shade index
        """
        # Delegate to composed shade computer
        shade, svi = self._stream_shade.compute(
            seg_idx,
            declination,
            summer_flag,
            self.seg_flow_width[seg_idx],
        )

        self.seg_shade[seg_idx] = shade

        return svi

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
        # Skip segments marked as permanently invalid
        # (matches Fortran line 887, 894)
        if self.seg_tave_water[seg_idx] < -99.0:
            # Never has flow - skip all calculations
            return

        if self.seginc_swrad[seg_idx] < -99.0:
            # Never has data - mark and skip
            self.seg_tave_water[seg_idx] = -99.9
            return

        # Check for no-flow conditions (matches Fortran check for
        # seg_outflow <= 0)
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

        # Compute energy fluxes for budget (if enabled)
        if self.track_energy_fluxes:
            self._compute_energy_fluxes(
                seg_idx, upstream_flow, lateral_flow, svi, te, ak1, ak2
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
        return _compute_mixed_inlet_temp(
            upstream_flow,
            lateral_flow,
            self.seg_tave_upstream[seg_idx],
            self.seg_tave_lat[seg_idx],
        )

    def _compute_energy_fluxes(
        self,
        seg_idx: int,
        upstream_flow: float,
        lateral_flow: float,
        svi: float,
        te: float,
        ak1: float,
        ak2: float,
    ) -> None:
        """Compute energy fluxes for budget tracking.

        Args:
            seg_idx: Segment index
            upstream_flow: Flow from upstream segments (cfs)
            lateral_flow: Lateral inflow (cfs)
            svi: Vegetation shade index
            te: Equilibrium temperature (degC)
            ak1: First-order thermal exchange coefficient
            ak2: Second-order thermal exchange coefficient
        """
        # Get segment properties
        width = self.seg_flow_width[seg_idx]
        length = self.seg_length[seg_idx]
        surface_area = width * length  # m²

        # Convert flows to m³/s
        q_up_cms = upstream_flow * CFS_TO_CMS
        q_lat_cms = lateral_flow * CFS_TO_CMS
        q_out_cms = self.seg_outflow[seg_idx] * CFS_TO_CMS

        # Water temperature (degC)
        t_water = self.seg_tave_water[seg_idx]
        t_water_abs = t_water + ZERO_C  # K

        # === INPUTS (Heat gains) ===

        # 1. Advective heat from upstream (W)
        if q_up_cms > NEARZERO:
            self.heat_upstream[seg_idx] = (
                q_up_cms
                * self.seg_tave_upstream[seg_idx]
                * SPECIFIC_HEAT_WATER
                * WATER_DENSITY
            )
        else:
            self.heat_upstream[seg_idx] = 0.0

        # 2. Advective heat from lateral inflow (W)
        if q_lat_cms > NEARZERO:
            self.heat_lateral[seg_idx] = (
                q_lat_cms
                * self.seg_tave_lat[seg_idx]
                * SPECIFIC_HEAT_WATER
                * WATER_DENSITY
            )
        else:
            self.heat_lateral[seg_idx] = 0.0

        # 3. Net shortwave solar radiation (W)
        sw_power = 11.63 / 24.0 * self.seginc_swrad[seg_idx]  # W/m²
        hs = (1.0 - self.seg_shade[seg_idx]) * sw_power * (1.0 - self.albedo)
        self.solar_radiation[seg_idx] = hs * surface_area

        # 4. Atmospheric longwave radiation (W)
        # From _equilb calculation
        vp_sat = 6.108 * np.exp(17.26939 * t_water / (t_water + 237.3))
        humidity = min(self.seg_humid[seg_idx], 0.99)
        ha = (
            (3.354939e-8 + 2.74995e-9 * np.sqrt(humidity * vp_sat))
            * (1.0 - self.seg_shade[seg_idx])
            * (1.0 + (0.17 * (self.seg_ccov[seg_idx] ** 2)))
        ) * (t_water_abs**4)
        self.atmospheric_longwave[seg_idx] = ha * surface_area

        # 5. Friction heating (W)
        q_init = max(self._seg_inflow[seg_idx] * CFS_TO_CMS, NEARZERO)
        hf = 9805.0 * (q_init / width) * self.seg_slope[seg_idx]  # W/m²
        self.friction_heat[seg_idx] = hf * surface_area

        # 6. Groundwater conduction (W)
        self.groundwater_conduction[seg_idx] = (
            self.seg_tave_gw[seg_idx] * AKZ * surface_area
        )

        # === OUTPUTS (Heat losses) ===

        # 7. Advective heat leaving segment (W)
        if q_out_cms > NEARZERO:
            self.heat_outflow[seg_idx] = (
                q_out_cms * t_water * SPECIFIC_HEAT_WATER * WATER_DENSITY
            )
        else:
            self.heat_outflow[seg_idx] = 0.0

        # 8. Longwave radiation emitted by water surface (W)
        self.longwave_emission[seg_idx] = A * (t_water_abs**4) * surface_area

        # 9. Longwave radiation from riparian vegetation (W)
        hv = 5.24e-8 * svi * (t_water_abs**4)  # W/m²
        self.longwave_vegetation[seg_idx] = hv * surface_area

        # 10. Evaporative cooling (W)
        evap = self.seg_potet[seg_idx] * MPS_CONVERT  # m/s
        self.evaporative_cooling[seg_idx] = (
            evap * LATENT_HEAT_VAPORIZATION * surface_area
        )

        # 11. Convective/sensible heat exchange (W)
        # This is computed as the residual to close the energy balance
        # Total inputs
        total_inputs = (
            self.heat_upstream[seg_idx]
            + self.heat_lateral[seg_idx]
            + self.solar_radiation[seg_idx]
            + self.atmospheric_longwave[seg_idx]
            + self.friction_heat[seg_idx]
            + self.groundwater_conduction[seg_idx]
        )

        # Total outputs (excluding convection)
        total_outputs_no_conv = (
            self.heat_outflow[seg_idx]
            + self.longwave_emission[seg_idx]
            + self.longwave_vegetation[seg_idx]
            + self.evaporative_cooling[seg_idx]
        )

        # Convection closes the balance (can be positive or negative)
        self.convective_exchange[seg_idx] = (
            total_inputs - total_outputs_no_conv
        )

        return

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
        return _equilb(
            t_o,
            svi,
            self._seg_inflow[seg_idx],
            self.seginc_swrad[seg_idx],
            self.seg_humid[seg_idx],
            self.seg_elev[seg_idx],
            self.seg_potet[seg_idx],
            self.seg_shade[seg_idx],
            self.seg_ccov[seg_idx],
            self.seg_flow_width[seg_idx],
            self.seg_slope[seg_idx],
            self.seg_tave_gw[seg_idx],
            self.albedo,
            int(self.maxiter_sntemp),
        )

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
        return _teak1(
            a_coef, b_coef, c_coef, d_coef, teq, int(self.maxiter_sntemp)
        )

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
        return _twavg(
            qup,
            t0,
            qlat,
            tl_avg,
            te,
            ak1,
            ak2,
            self.seg_flow_width[seg_idx],
            self.seg_length[seg_idx],
        )


@nb.jit(nopython=True)
def _teak1(a_coef, b_coef, c_coef, d_coef, teq, maxiter_sntemp):
    """Solve for equilibrium temperature using Newton-Raphson iteration.

    This is the teak1 function from PRMS.

    Args:
        a_coef, b_coef, c_coef, d_coef: Coefficients
        teq: Initial guess for equilibrium temperature
        maxiter_sntemp: Maximum iterations

    Returns:
        teq: Equilibrium temperature (degC)
        ak1c: First-order thermal exchange coefficient
    """
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
        fpte = (4.0 * a_coef * (teabs**3.0)) + b_coef - (2.0 * c_coef * teq)
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


@nb.jit(nopython=True)
def _equilb(
    t_o,
    svi,
    seg_inflow,
    seginc_swrad,
    seg_humid,
    seg_elev,
    seg_potet,
    seg_shade,
    seg_ccov,
    seg_flow_width,
    seg_slope,
    seg_tave_gw,
    albedo,
    maxiter_sntemp,
):
    """Compute equilibrium temperature using full energy balance.

    This is the equilb function from PRMS.

    Args:
        t_o: Initial temperature (degC)
        svi: Vegetation shade index
        seg_inflow: Segment inflow (CFS)
        seginc_swrad: Incident shortwave radiation
        seg_humid: Segment humidity
        seg_elev: Segment elevation
        seg_potet: Segment potential ET
        seg_shade: Segment shade fraction
        seg_ccov: Cloud cover fraction
        seg_flow_width: Flow width
        seg_slope: Segment slope
        seg_tave_gw: Groundwater temperature
        albedo: Albedo
        maxiter_sntemp: Maximum iterations

    Returns:
        te: Equilibrium temperature (degC)
        ak1: First-order thermal exchange coefficient
        ak2: Second-order thermal exchange coefficient
    """
    # Local Variables
    taabs = float(t_o + ZERO_C)

    vp_sat = 6.108 * np.exp(17.26939 * t_o / (t_o + 237.3))

    # Convert units and set up parameters
    q_init = max(seg_inflow * CFS_TO_CMS, NEARZERO)

    sw_power = 11.63 / 24.0 * float(seginc_swrad)

    # If humidity is 1.0, there is a divide by zero below
    foo = min(seg_humid, 0.99)

    # Compute atmospheric pressure based on segment elevation
    press = 1013.0 - (0.1055 * seg_elev)

    bow_coeff = (0.00061 * press) / (vp_sat * (1.0 - foo))
    evap = float(seg_potet * MPS_CONVERT)

    # Heat flux components
    # Ha: atmospheric-emitted longwave radiation
    ha = (
        (3.354939e-8 + 2.74995e-9 * np.sqrt(foo * vp_sat))
        * (1.0 - seg_shade)
        * (1.0 + (0.17 * (seg_ccov**2)))
    ) * (taabs**4)

    # Hf: heat dissipated from potential energy by friction
    hf = 9805.0 * (q_init / seg_flow_width) * seg_slope

    # Hs: net flux from shortwave solar radiation
    hs = (1.0 - seg_shade) * sw_power * (1.0 - albedo)

    # Hv: longwave radiation emitted by riparian vegetation
    hv = 5.24e-8 * svi * (taabs**4)

    # Determine equilibrium coefficients
    del_ht = 2.36e06
    ltnt_ht = 2495.0e06

    b = bow_coeff * evap * (ltnt_ht + (del_ht * t_o)) + AKZ - (del_ht * evap)
    c = bow_coeff * del_ht * evap
    d = (ha + hv + hf + hs) + (
        ltnt_ht * evap * ((bow_coeff * t_o) - 1.0) + (seg_tave_gw * AKZ)
    )

    # Determine equilibrium temperature & 1st order thermal exchange coef
    ted = t_o
    ted, ak1d = _teak1(A, b, c, d, ted, maxiter_sntemp)

    # Determine 2nd order thermal exchange coefficient
    hnet = (A * ((t_o + ZERO_C) ** 4)) + (b * t_o) - (c * (t_o**2.0)) - d
    delt = t_o - ted

    if abs(delt) < NEARZERO:
        ak2d = 0.0
    else:
        ak2d = ((delt * ak1d) - hnet) / (delt**2)

    return ted, ak1d, ak2d


@nb.jit(nopython=True)
def _twavg(
    qup,
    t0,
    qlat,
    tl_avg,
    te,
    ak1,
    ak2,
    seg_flow_width,
    seg_length,
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
        seg_flow_width: Flow width
        seg_length: Segment length

    Returns:
        tw: Average water temperature (degC)
    """
    # Determine equation parameters
    q_init = float(qup * CFS_TO_CMS)
    ql = float(qlat)
    width = seg_flow_width
    length = seg_length

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


@nb.jit(nopython=True)
def _lat_inflow(
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

    Args:
        seg_lateral_inflow: Total lateral inflow to segment (cfs)
        seginc_sroff: Surface runoff component (cfs)
        seginc_ssflow: Subsurface flow component (cfs)
        seginc_gwflow: Groundwater flow component (cfs)
        melt_temp: Snowmelt temperature (degC)
        tave_gw: Groundwater temperature (degC)
        tave_air: Air temperature (degC)
        tave_ss: Subsurface temperature (degC)
        melt: Snowmelt (inches)
        rain: Rainfall (inches)

    Returns:
        tl_avg: Weighted average lateral inflow temperature (degC)
        qlat: Lateral inflow (cms)
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
        tl_avg = weight_roff * troff + weight_ss * tss + weight_gw * tave_gw

    return tl_avg, qlat


@nb.jit(nopython=True)
def _compute_mixed_inlet_temp(
    upstream_flow, lateral_flow, seg_tave_upstream, seg_tave_lat
):
    """Compute mixed inlet temperature from upstream and lateral sources.

    Args:
        upstream_flow: Flow from upstream segments (cfs)
        lateral_flow: Lateral inflow (cfs)
        seg_tave_upstream: Upstream temperature (degC)
        seg_tave_lat: Lateral flow temperature (degC)

    Returns:
        Mixed inlet temperature (degC)
    """
    upstream_ready = upstream_flow > 0.0 and not np.isnan(seg_tave_upstream)
    lateral_ready = lateral_flow > 0.0 and not np.isnan(seg_tave_lat)

    if not upstream_ready and not lateral_ready:
        return np.nan
    elif upstream_ready and not lateral_ready:
        return seg_tave_upstream
    elif lateral_ready and not upstream_ready:
        return seg_tave_lat
    else:
        # Both sources present - compute weighted average
        return (
            seg_tave_upstream * upstream_flow + seg_tave_lat * lateral_flow
        ) / (upstream_flow + lateral_flow)
