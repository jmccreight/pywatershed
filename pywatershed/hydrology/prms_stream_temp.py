import pathlib as pl
from pathlib import Path
from typing import Literal, Union
from warnings import warn

import networkx as nx
import numba as nb
import numpy as np

from ..base.adapter import adaptable
from ..base.conservative_process import ConservativeProcess
from ..base.control import Control
from ..constants import cfs_to_cms, nearzero, zero
from ..parameters import Parameters
from .prms_stream_shade import (
    PRMSStreamShade,
    PRMSStreamShadeConstant,
    PRMSStreamShadeDynamic,
)

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


class PRMSStreamTempHumidityCBH(ConservativeProcess):
    """PRMS stream temperature with time-varying humidity input.

    PRMSStreamTemp and PRMSStreamTempHumidityCBH model stream temperature
    following PRMS 5.2.1.1. The class :class:`PRMSStreamTemp` covers the
    situation where humidity is supplied by the parameter files
    (``strmtemp_humidity_flag=1``) and :class:`PRMSStreamTempHumidityCBH`
    covers the case of when humidity is supplied by time-varying input files
    (NetCDF CBH equivalents, e.g. ``strmtemp_humidity_flag=0``. Note there is
    not currently an implementation for the PRMS case
    ``strmtemp_humidity_flag=2``). This documentation provides general
    background for both classes.

    The stream temperature classes use:

    - :class:`PRMSHydraulicGeometryFull` and
      :class:`PRMSHydraulicGeometryWidthOnly` as upstream processes which
      provide the hydraulic geometry variables (and are renamed seg_flow_*
      compared to the PRMS seg_* variables)
    - :class:`PRMSStreamShadeConstant` and :class:`PRMSStreamShadeDynamic`
      as models of stream shade. These are to be passed to or "composed" into
      the stream temperature on initialization.

    See the example notebooks
    `examples/01_multi-process_models.ipynb
    <https://github.com/DOI-USGS/pywatershed/blob/develop/examples/01_multi-process_models.ipynb>`__
    and
    `examples/02_prms_legacy_models.ipynb
    <https://github.com/DOI-USGS/pywatershed/blob/develop/examples/02_prms_legacy_models.ipynb>`__
    for worked examples.

    This implementation is based on PRMS 5.2.1.1 with theoretical documentation
    given by:

    `Markstrom, Steven L. P2S -- Coupled simulation with the
    Precipitation-Runoff Modeling System (PRMS) and the Stream Temperature
    Network (SNTemp) Models.
    No. 2012-1116. US Geological Survey, 2012.
    <https://pubs.usgs.gov/publication/ofr20121116>`__

    `Sanders, M.J., Markstrom, S.L., Regan, R.S., and Atkinson, R.D., 2017,
    Documentation of a daily mean stream temperature module — An enhancement to
    the Precipitation-Runoff Modeling System: U.S. Geological Survey Techniques
    and Methods, book 6, chap. D4, 18 p. <https://doi.org/10.3133/tm6D4>`__

    The stream temperature module computes daily mean water temperature for
    each stream segment using an energy balance approach. The module accounts
    for:

    - Solar radiation (shortwave)
    - Longwave radiation (atmospheric and vegetation)
    - Convection and evaporation
    - Conduction with the streambed
    - Temperature of inflows (upstream, lateral, groundwater, subsurface)
    - Shading from riparian vegetation and topography

    HRU quantities are aggregated to segments internally, computing segment-
    level meteorological variables (seg_tave_air, seg_humid, seg_ccov,
    seg_melt, seg_rain) from HRU inputs. This follows the Fortran PRMS
    stream_temp.f90 logic (lines ~699-850).

    Args:
        control: a Control object
        discretization: a discretization of class Parameters
        parameters: a parameter object of class Parameters
        seg_outflow: Streamflow leaving each segment (from PRMSChannel)
        seg_lateral_inflow: Lateral inflow entering each segment
            (from PRMSChannel)
        swrad: Solar radiation for each HRU (from PRMSAtmosphere)
        potet: Potential ET for each HRU (from PRMSAtmosphere)
        sroff: Surface runoff for each HRU (from PRMSRunoff)
        ssres_flow: Subsurface flow for each HRU (from PRMSSoilzone)
        gwres_flow: Groundwater flow for each HRU (from PRMSGroundwater)
        tavgc: Average air temperature for each HRU in Celsius
            (from PRMSAtmosphere)
        snowmelt: Snowmelt for each HRU (from PRMSSnow)
        hru_rain: Rainfall for each HRU (from PRMSAtmosphere)
        soltab_potsw: Potential shortwave radiation table for current day
            (from PRMSSolarGeometry or Adapter)
        humidity_hru: Humidity for each HRU (from CBH via Adapter).
            Values are relative humidity in percent (0-100).
        seg_flow_width: Flow-dependent width from PRMSHydraulicGeometryFull
        seg_flow_depth: Flow-dependent depth from PRMSHydraulicGeometryFull
        seg_flow_area: Flow-dependent cross-sectional area from
            PRMSHydraulicGeometryFull
        seg_flow_velocity: Flow-dependent velocity from
            PRMSHydraulicGeometryFull
        stream_shade: PRMSStreamShade instance (Dynamic or Constant).
            If provided, stream_shade_class and stream_shade_parameters
            are ignored.
        stream_shade_class: Class to use for stream shade computation
            (e.g., PRMSStreamShadeDynamic, PRMSStreamShadeConstant).
            Only used if stream_shade is None.
        stream_shade_parameters: Parameters for stream shade computation.
            Can be a Parameters object or a Path to a parameter file.
            Only used if stream_shade is None and stream_shade_class is
            provided.

        imbalance_behavior: one of ["defer", None, "warn", "error"]
        verbose: Print extra information or not?
        use_vectorized_shade: Use vectorized shade computation for all
            segments at once (default True)
        track_energy_fluxes: Track energy flux terms for budget analysis

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
        tavgc: adaptable,
        snowmelt: adaptable,
        hru_rain: adaptable,
        soltab_potsw: adaptable,
        humidity_hru: adaptable,
        seg_flow_width: adaptable = None,
        seg_flow_depth: adaptable = None,
        seg_flow_area: adaptable = None,
        seg_flow_velocity: adaptable = None,
        stream_shade: Union[PRMSStreamShade, None] = None,
        stream_shade_class: Union[type, None] = None,
        stream_shade_parameters: Union[Parameters, Path, None] = None,
        calc_method: Literal["numba", "numpy", None] = None,
        imbalance_behavior: Literal["defer", None, "warn", "error"] = "defer",
        input_aliases: dict = None,
        verbose: bool = False,
        use_vectorized_shade: bool = True,
        track_energy_fluxes: bool = True,
        atmos_exchange_factor: float = 1.0,
    ) -> None:
        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
            imbalance_behavior=imbalance_behavior,
            input_aliases=input_aliases,
        )
        self.name = "PRMSStreamTempHumidityCBH"

        # Handle stream_shade initialization before _set_inputs
        # This resolves the stream_shade from multiple possible input styles
        stream_shade = self._init_stream_shade(
            stream_shade=stream_shade,
            stream_shade_class=stream_shade_class,
            stream_shade_parameters=stream_shade_parameters,
            parameters=parameters,
            discretization=discretization,
        )

        # Update locals with resolved stream_shade for _set_inputs
        # Remove the class/parameters options since they've been resolved
        input_locals = locals().copy()
        input_locals["stream_shade"] = stream_shade
        input_locals.pop("stream_shade_class", None)
        input_locals.pop("stream_shade_parameters", None)

        self._set_inputs(input_locals)
        self._set_options(input_locals)

        # Store atmospheric exchange factor for sensitivity testing
        self._atmos_exchange_factor = atmos_exchange_factor

        # Just the names without the catergorization, in one list.
        self._energy_flux_vars = [
            item
            for vals in self.get_energy_budget_terms().values()
            for item in vals
        ]

        self._set_budget(basis="unit", quantity="energy")
        self._initialize_stream_temp()
        self._init_calc_method()

        # Consistency checks for energy flux tracking
        if not self._track_energy_fluxes:
            # Check 1: imbalance_behavior must be None if not tracking
            # energy fluxes
            if imbalance_behavior is not None:
                msg = (
                    "Inconsistent options: track_energy_fluxes=False "
                    f"requires imbalance_behavior=None, but "
                    f"imbalance_behavior={imbalance_behavior!r}"
                )
                raise ValueError(msg)

            # Check 2: Set energy flux variables to None if not tracking
            for var in self._energy_flux_vars:
                if hasattr(self, var):
                    setattr(self, var, None)

        else:
            # Initialize arrays for storing energy flux terms during
            # calculation
            self._hs_terms = np.zeros(self.nsegment)
            self._ha_terms = np.zeros(self.nsegment)
            self._hf_terms = np.zeros(self.nsegment)
            self._hv_terms = np.zeros(self.nsegment)
            self._evap_terms = np.zeros(self.nsegment)
            self._t_abs4_terms = np.zeros(self.nsegment)
            self._upstream_flows = np.zeros(self.nsegment)
            self._lateral_flows = np.zeros(self.nsegment)

        return

    @staticmethod
    def get_dimensions() -> tuple:
        return ("nhru", "nsegment", "nmonth", "ndoy")

    @staticmethod
    def get_parameters() -> tuple:
        return (
            "doy",
            "hru_segment",
            "hru_area",
            "hru_slope",
            "tosegment",
            "albedo",
            "lat_temp_adj",
            "seg_length",
            "seg_slope",
            "seg_lat",
            "seg_elev",
            "seg_close",
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
            "seg_flow_width",
            "seg_flow_depth",
            "seg_flow_area",
            "seg_flow_velocity",
            "swrad",
            "potet",
            "sroff",
            "ssres_flow",
            "gwres_flow",
            "tavgc",
            "snowmelt",
            "hru_rain",
            "soltab_potsw",
            "humidity_hru",
        )

    @staticmethod
    def get_init_values() -> dict:
        return {
            "seg_tave_water": -99.9,
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
            # Segment-level meteorological variables (computed from HRU inputs)
            "seg_tave_air": 0.0,
            "seg_humid": 0.0,
            "seg_ccov": 0.0,
            "seg_melt": 0.0,
            "seg_rain": 0.0,
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
            Dictionary with inputs, outputs, exchanges, and storage_changes
            for energy budget.

        Notes:
            Energy fluxes are computed in Watts (J/s). The budget tracks:
            - Advective heat transport (upstream, lateral, outflow)
            - Surface energy exchange (solar, longwave, evaporation)
            - Internal sources (friction, groundwater conduction)
            - Bi-directional exchanges that can be gains or losses

            Storage changes are empty because the kinematic wave assumption
            means water storage is constant - only temperature (and thus
            heat content) changes, which is captured by the balance of
            inputs and outputs.

            Exchanges are net fluxes that can be positive (heat gain) or
            negative (heat loss) depending on temperature gradients:
            - convective_exchange: depends on air vs water temperature
            While it seems that the following could be exchanges, their current
            formulation restricts their sign and they have been categorized
            to match that pre-determined sign:
            - longwave_vegetation: net longwave exchange with vegetation
            - groundwater_conduction: depends on groundwater vs water temp
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
            "exchanges": [
                "convective_exchange",  # Sensible heat exchange (W, ±)
            ],
            "outputs": [
                "heat_outflow",  # Advective heat leaving (W)
                "longwave_emission",  # LW radiation emitted (W)
                "longwave_vegetation",  # LW from vegetation (W)
                "evaporative_cooling",  # Latent heat loss (W)
            ],
            "storage_changes": [
                # Empty - kinematic wave assumption (no storage change)
            ],
        }

    def _init_stream_shade(
        self,
        stream_shade: Union[PRMSStreamShade, None],
        stream_shade_class: Union[type, None],
        stream_shade_parameters: Union[Parameters, Path, None],
        parameters: Parameters,
        discretization: Parameters,
    ) -> PRMSStreamShade:
        """Initialize stream shade from various input options.

        Three cases are handled:
        1. stream_shade is provided directly - use it as-is
        2. stream_shade_class and optionally stream_shade_parameters are
           provided - instantiate the class with the parameters
        3. All are None - default to PRMSStreamShadeConstant using
           parameters from self._params

        Args:
            stream_shade: Pre-instantiated PRMSStreamShade object
            stream_shade_class: Class to instantiate for shade computation
            stream_shade_parameters: Parameters for shade class (Parameters
                object or Path)
            parameters: The main parameters for PRMSStreamTemp
            discretization: Discretization parameters

        Returns:
            PRMSStreamShade instance
        """
        # Case 1: stream_shade provided directly
        if stream_shade is not None:
            if not isinstance(stream_shade, PRMSStreamShade):
                raise TypeError(
                    f"stream_shade must be a PRMSStreamShade instance, "
                    f"got {type(stream_shade)}"
                )
            return stream_shade

        # Case 2: stream_shade_class provided
        if stream_shade_class is not None:
            if not issubclass(stream_shade_class, PRMSStreamShade):
                raise TypeError(
                    f"stream_shade_class must be a subclass of "
                    f"PRMSStreamShade, got {stream_shade_class}"
                )

            # Load parameters if provided as path
            if stream_shade_parameters is not None:
                if isinstance(stream_shade_parameters, (str, Path)):
                    shade_params = Parameters.from_netcdf(
                        stream_shade_parameters
                    )
                else:
                    shade_params = stream_shade_parameters
            else:
                # Use the main parameters if no separate shade params provided
                shade_params = parameters

            return stream_shade_class(shade_params, discretization)

        # zero is the PRMS default value for this option
        stream_temp_shade_flag = self.control.options.get(
            "stream_temp_shade_flag", 0
        )

        if stream_temp_shade_flag == 1:
            stream_shade_class = PRMSStreamShadeConstant
        elif stream_temp_shade_flag != 0:
            raise ValueError(
                f"Invalid value for {stream_temp_shade_flag=} option"
            )
        else:
            stream_shade_class = PRMSStreamShadeDynamic

        # Case 3: All None - default to PRMSStreamShadeConstant
        # Try to instantiate using the main parameters
        try:
            return stream_shade_class(parameters, discretization)
        except (KeyError, AttributeError) as e:
            raise ValueError(
                "stream_shade not provided and could not initialize "
                f"{stream_shade_class} from parameters. Either provide "
                "stream_shade, or stream_shade_class with "
                "stream_shade_parameters, or ensure parameters contain "
                f"the required shade parameters. Original error: {e}"
            ) from e

    def initialize_netcdf(
        self,
        output_dir: Union[str, pl.Path] = None,
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
        (
            output_dir,
            output_vars,
            separate_files,
        ) = self._reconcile_nc_args_w_control_opts(
            output_dir, output_vars, separate_files
        )

        # Set output_vars appropriately based on tracking
        if output_vars is None:
            if self._track_energy_fluxes:
                # Include all variables
                output_vars = self.get_variables()
            else:
                # Exclude energy flux variables
                output_vars = [
                    vv
                    for vv in self.get_variables()
                    if vv not in self._energy_flux_vars
                ]
        else:
            # Check if energy flux variables are requested when not tracking
            output_vars = list(set(self.get_variables()) & set(output_vars))
            if not self._track_energy_fluxes:
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

    def _set_initial_conditions(self) -> None:
        # Initialize state variables
        # seg_tave_water starts as -99.9 from get_init_values
        # Segments with no upstream HRUs will be set to NaN ("never has flow")
        # in _initialize_stream_temp
        # Don't override seg_tave_water here - it's already -99.9
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

        # Initialize circular buffers with -99.9 (invalid marker) to
        # match Fortran PRMS
        # Values will be filled as air temps are added over time
        self.gw_silo[:, :] = -99.9
        self.ss_silo[:, :] = -99.9
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

        # Validate and correct seg_length (Fortran lines 428-435)
        # Check before conversion to km
        errors = []
        for i in range(self.nsegment):
            if self.seg_length[i] < NEARZERO:
                errors.append(
                    f"seg_length too small for segment {i}: "
                    f"{self.seg_length[i]}"
                )
            elif self.seg_length[i] < 1.0 and self._verbose:
                warn(
                    f"seg_length < 1.0 meters for segment {i}: "
                    f"{self.seg_length[i]}"
                )
        if errors:
            raise ValueError("\n".join(errors))

        # Convert seg_length from meters to kilometers (Fortran line 441)
        self.seg_length_km = self.seg_length / 1000.0

        # Validate and correct seg_slope (Fortran lines 436-440)
        # New minimum is 0.0000001 (was 0.0001 in old version)
        for i in range(self.nsegment):
            if self.seg_slope[i] < 0.0000001:
                if self._verbose:
                    warn(
                        f"seg_slope < 0.0000001 for segment {i}, "
                        f"setting to 0.0000001 (was {self.seg_slope[i]})"
                    )
                self.seg_slope[i] = 0.0000001

        # Note: lat_temp_adj units are degrees Celsius (not decimal fraction)
        # This matches Fortran stream_temp.f90 line 202

        # Compute hru_cossl from hru_slope (matches Fortran/PRMSSolarGeometry)
        self._hru_cossl = np.cos(np.arctan(self.hru_slope))

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

        # Handle seg_close parameter (new in PRMS 5.2.1.1)
        # Matches Fortran stream_temp.f90 lines 540-576
        # seg_close is now a parameter that can be:
        #   -1: auto-set to segment_up (with warning)
        #   >0: user-specified closest segment for segments without HRUs

        # Check if seg_close is specified or should default
        # (Fortran line 542-545)
        if self.seg_close[0] == -1:
            # Not specified - use segment_up with warning
            self.seg_close = np.copy(segment_up)
            self.seg_close_flag = 0
            if self._verbose:
                warn(
                    "seg_close parameter not specified (value=-1), "
                    "setting to segment_up or other segment if no segment_up"
                )
        else:
            # User specified - validate that segments without HRUs have
            # valid seg_close
            self.seg_close_flag = 1
            errors = []
            for i in range(self.nsegment):
                if self.segment_hruarea[i] <= NEARZERO:
                    if self.seg_close[i] == 0:
                        errors.append(
                            f"segment {i} does not have associated HRUs "
                            f"but seg_close is 0"
                        )
            if errors:
                raise ValueError("\n".join(errors))

        # For auto-assigned seg_close (seg_close_flag=0), handle special
        # cases (Fortran lines 563-576)
        if self.seg_close_flag == 0:
            for jj in range(self.nsegment):
                i = self.segment_order[jj]

                if self.segment_hruarea[i] <= NEARZERO:
                    # If segment_up is 0, need to find alternative
                    if self.segment_up[i] == 0:
                        if jj > 0:
                            # Use previous segment in route order
                            self.seg_close[i] = self.segment_order[jj - 1]
                            if self._verbose:
                                warn(
                                    f"segment_up = 0 without associated HRU "
                                    f"for segment {i}, will use previous "
                                    f"segment in route order"
                                )
                        else:
                            raise ValueError(
                                f"Cannot set associated segment for segment "
                                f"without associated HRU for segment {i}. "
                                f"Must specify seg_close for this case."
                            )

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

                # If no upstream HRUs, mark as never having flow (NaN)
                # Valid segments are already -99.9 from get_init_values
                if not has_upstream_hrus:
                    self.seg_tave_water[i] = np.nan

        # Note: seg_length conversion to km is done earlier as seg_length_km

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
        self._outflow_mask = np.full((len(self.tosegment)), False)
        for iseg in range(self.nsegment):
            toseg = self.tosegment[iseg] - 1  # Convert to 0-based
            if toseg >= 0:  # -1 means outlet in 0-based indexing
                connectivity.append((iseg, toseg))
            else:
                self._outflow_mask[iseg] = True

        # Use NetworkX for topological sort
        if self.nsegment > 1 and len(connectivity) > 0:
            graph = nx.DiGraph()
            graph.add_edges_from(connectivity)
            segment_order = list(nx.topological_sort(graph))
        else:
            segment_order = list(range(self.nsegment))

        # if the domain contains links with no upstream or
        # downstream reaches, we just throw these back at the
        # top of the order since networkx wont handle such nonsense
        wh_mask_set = set(np.where(self._outflow_mask)[0])
        seg_ord_set = set(segment_order)
        mask_not_seg_ord = list(wh_mask_set - seg_ord_set)
        if len(mask_not_seg_ord):
            segment_order = mask_not_seg_ord + segment_order
            for pp in mask_not_seg_ord:
                assert ((self.tosegment[pp] - 1) == -1) and (
                    pp not in (self.tosegment - 1)
                )

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

    def _init_calc_method(self):
        """Initialize calculation method (numpy or numba)."""
        if self._calc_method is None:
            self._calc_method = "numba"

        avail_methods = ["numpy", "numba"]
        if self._calc_method.lower() not in avail_methods:
            msg = (
                f"Invalid calc_method={self._calc_method} for {self.name}. "
                f"Setting calc_method to 'numba' for {self.name}"
            )
            warn(msg)
            self._calc_method = "numba"

        if self._calc_method.lower() == "numba":
            numba_msg = f"{self.name} jit compiling with numba"
            print(numba_msg, flush=True)

            # Wrap methods with numba jit compilation
            self._update_running_avg_temp = nb.njit(
                self._update_running_avg_temp
            )
            self._compute_lateral_temp = nb.njit(self._compute_lateral_temp)
            self._compute_water_temp = nb.njit(self._compute_water_temp)

        # else: methods are already available without wrapping

        return

    def _advance_variables(self) -> None:
        """Advance variables from current to previous timestep."""
        # No explicit time advancement needed for this module
        return

    def _calculate(self, time_length) -> None:
        """Calculate stream temperature for all segments."""

        # Get current month (1-based) and day of year
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
        # PRMSHydraulicGeometryFull process, not computed here

        # Compute running average temperatures for groundwater and subsurface
        # Skip segments marked as never having flow (matches Fortran checks)
        self._update_running_avg_temp(
            self.segment_order,
            self.seg_tave_water,
            self.seginc_swrad,
            self.seg_tave_air,
            self.gw_tau,
            self.gw_index,
            self.gw_silo,
            self.gw_sum,
            self.seg_tave_gw,
            self.ss_tau,
            self.ss_index,
            self.ss_silo,
            self.ss_sum,
            self.seg_tave_ss,
        )

        # Compute lateral flow temperatures
        # Skip segments marked as never having flow or data
        self._compute_lateral_temp(
            self.segment_order,
            self.seg_tave_water,
            self.seginc_swrad,
            self.seg_lateral_inflow,
            self.seginc_sroff,
            self.seginc_ssflow,
            self.seginc_gwflow,
            self.seg_melt,
            self.seg_rain,
            self.seg_tave_gw,
            self.seg_tave_air,
            self.seg_tave_ss,
            self.melt_temp,
            self.lat_temp_adj,
            self.control.current_month,
            self.seg_tave_lat,
        )

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
        # Don't reset segments marked as NaN (never have flow)
        # Valid segments get reset to -99.9 before calculation
        for jj in self.segment_order:
            if not np.isnan(self.seg_tave_water[jj]):
                self.seg_tave_water[jj] = -99.9

        # Note: Assumes vectorized shade is being used
        self._compute_water_temp(
            self.segment_order,
            self.seg_tave_water,
            self.seginc_swrad,
            self.seg_outflow,
            self.upstream_count,
            self.upstream_idx,
            self.seg_lateral_inflow,
            self.seg_tave_upstream,
            seg_svi_all,
            self._seg_inflow,
            self.seg_tave_lat,
            self.seginc_swrad,
            self.seg_humid,
            self.seg_elev,
            self.seg_potet,
            self.seg_shade,
            self.seg_ccov,
            self.seg_flow_width,
            self.seg_slope,
            self.seg_tave_gw,
            self.seg_length_km,
            self.albedo,
            int(self.maxiter_sntemp),
            self._track_energy_fluxes,
            self._hs_terms if self._track_energy_fluxes else np.zeros(1),
            self._ha_terms if self._track_energy_fluxes else np.zeros(1),
            self._hf_terms if self._track_energy_fluxes else np.zeros(1),
            self._hv_terms if self._track_energy_fluxes else np.zeros(1),
            self._evap_terms if self._track_energy_fluxes else np.zeros(1),
            self._t_abs4_terms if self._track_energy_fluxes else np.zeros(1),
            self._upstream_flows if self._track_energy_fluxes else np.zeros(1),
            self._lateral_flows if self._track_energy_fluxes else np.zeros(1),
            self._atmos_exchange_factor,
        )

        # Compute energy fluxes for all segments (vectorized)
        if self._track_energy_fluxes:
            self._compute_energy_fluxes_vectorized()

        return

    def _compute_segment_aggregates(self) -> None:
        """Compute segment aggregate variables from HRU inputs.

        This implements the aggregation calculations from PRMS stream_temp.f90
        around line 699-850. Computes:
        - seginc_sroff, seginc_ssflow, seginc_gwflow, seginc_swrad (flow/rad)
        - seg_tave_air, seg_melt, seg_rain, seg_ccov (met vars, non-humidity)
        - seg_humid via _compute_seg_humid_cbh_numba (CBH humidity path)
        """
        # Compute all segment aggregates except humidity
        _compute_segment_aggregates_numba(
            self.nhru,
            self.nsegment,
            self.hru_segment,
            self.hru_area,
            self.sroff,
            self.ssres_flow,
            self.gwres_flow,
            self.swrad,
            self.segment_hruarea,
            self.segment_up,
            self.tosegment,
            self.seginc_sroff,
            self.seginc_ssflow,
            self.seginc_gwflow,
            self.seginc_swrad,
            self.tavgc,
            self.snowmelt,
            self.hru_rain,
            self.soltab_potsw,
            self._hru_cossl,
            self.segment_order,
            self.seg_close,
            self.seg_tave_air,
            self.seg_melt,
            self.seg_rain,
            self.seg_ccov,
        )

        # Compute seg_humid from CBH humidity_hru input (flag==0 path)
        _compute_seg_humid_cbh_numba(
            self.nhru,
            self.nsegment,
            self.hru_segment,
            self.hru_area,
            self.humidity_hru,
            self.segment_hruarea,
            self.segment_order,
            self.seg_close,
            self.seg_humid,
        )

        return

    def _compute_seg_potet(self) -> None:
        """Compute seg_potet using stream_temp.f90 logic.

        This matches stream_temp.f90 lines 807-848, using segment_order
        and seg_close.
        """
        # Use numba-optimized function for performance
        _compute_seg_potet_numba(
            self.nhru,
            self.nsegment,
            self.hru_segment,
            self.hru_area,
            self.potet,
            self.segment_order,
            self.segment_hruarea,
            self.seg_close,
            self.seg_potet,
        )

        return

    # Note: _compute_hydraulic_geometry removed - now handled by
    # PRMSHydraulicGeometryFull process

    def _update_running_avg_temp_single(
        self, seg_idx: int, comp_type: str
    ) -> None:
        """Update running average temperature for groundwater or
        subsurface (single segment).

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

    def _compute_lateral_temp_single(
        self, seg_idx: int, nowmonth: int
    ) -> None:
        """Compute lateral flow temperature for a segment (single segment).

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

    def _compute_water_temp_single(self, seg_idx: int, svi: float) -> None:
        """Compute water temperature for a segment (single segment).

        Args:
            seg_idx: Segment index
            svi: Vegetation shade index
        """
        # Skip segments marked as permanently invalid (NaN = never has flow)
        # (matches Fortran line 887, 894)
        if np.isnan(self.seg_tave_water[seg_idx]):
            # Never has flow - skip all calculations
            return

        if self.seginc_swrad[seg_idx] < -99.0:
            # Never has data - mark and skip
            self.seg_tave_water[seg_idx] = np.nan
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
        result = self._equilb(seg_idx, t_in, svi)
        te, ak1, ak2 = result[0], result[1], result[2]

        # Store energy flux terms if tracking (for vectorized computation)
        if self._track_energy_fluxes:
            self._hs_terms[seg_idx] = result[3]
            self._ha_terms[seg_idx] = result[4]
            self._hf_terms[seg_idx] = result[5]
            self._hv_terms[seg_idx] = result[6]
            self._evap_terms[seg_idx] = result[7]
            self._t_abs4_terms[seg_idx] = result[8]
            self._upstream_flows[seg_idx] = upstream_flow
            self._lateral_flows[seg_idx] = lateral_flow

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
        return _compute_mixed_inlet_temp(
            upstream_flow,
            lateral_flow,
            self.seg_tave_upstream[seg_idx],
            self.seg_tave_lat[seg_idx],
        )

    def _compute_energy_fluxes_vectorized(self) -> None:
        """Compute energy fluxes for all segments after temperature
        calculations.

        This method uses intermediate terms stored during the temperature
        calculation loop to avoid redundant computation. All terms are
        computed vectorially for efficiency.
        """
        # Surface areas (vectorized)
        surface_area = (
            self.seg_flow_width * self.seg_length_km * 1000.0
        )  # m² (convert km to m)

        # Convert flows to m³/s (vectorized)
        q_up_cms = self._upstream_flows * CFS_TO_CMS
        q_lat_cms = self._lateral_flows * CFS_TO_CMS
        q_out_cms = self.seg_outflow * CFS_TO_CMS

        # === INPUTS (Heat gains) ===

        # 1. Advective heat from upstream (W) - vectorized
        self.heat_upstream[:] = np.where(
            q_up_cms > NEARZERO,
            q_up_cms
            * self.seg_tave_upstream
            * SPECIFIC_HEAT_WATER
            * WATER_DENSITY,
            0.0,
        )

        # 2. Advective heat from lateral inflow (W) - vectorized
        self.heat_lateral[:] = np.where(
            q_lat_cms > NEARZERO,
            q_lat_cms
            * self.seg_tave_lat
            * SPECIFIC_HEAT_WATER
            * WATER_DENSITY,
            0.0,
        )

        # 3. Net shortwave solar radiation (W) - use stored terms
        self.solar_radiation[:] = self._hs_terms * surface_area

        # 4. Atmospheric longwave radiation (W) - use stored terms
        self.atmospheric_longwave[:] = self._ha_terms * surface_area

        # 5. Friction heating (W) - use stored terms
        self.friction_heat[:] = self._hf_terms * surface_area

        # 6. Groundwater conduction (W) - vectorized
        self.groundwater_conduction[:] = self.seg_tave_gw * AKZ * surface_area

        # === OUTPUTS (Heat losses) ===

        # 7. Advective heat leaving segment (W) - vectorized
        self.heat_outflow[:] = np.where(
            q_out_cms > NEARZERO,
            q_out_cms
            * self.seg_tave_water
            * SPECIFIC_HEAT_WATER
            * WATER_DENSITY,
            0.0,
        )

        # 8. Longwave radiation emitted by water surface (W)
        # Use stored T^4 terms
        self.longwave_emission[:] = A * self._t_abs4_terms * surface_area

        # 9. Longwave radiation from riparian vegetation (W)
        # Use stored terms
        self.longwave_vegetation[:] = self._hv_terms * surface_area

        # 10. Evaporative cooling (W) - use stored evap terms
        self.evaporative_cooling[:] = (
            self._evap_terms * LATENT_HEAT_VAPORIZATION * surface_area
        )

        # 11. Convective/sensible heat exchange (W) - vectorized residual
        # Total inputs
        total_inputs = (
            self.heat_upstream
            + self.heat_lateral
            + self.solar_radiation
            + self.atmospheric_longwave
            + self.friction_heat
            + self.groundwater_conduction
        )

        # Total outputs (excluding convection)
        total_outputs = (
            self.heat_outflow
            + self.longwave_emission
            + self.longwave_vegetation
            + self.evaporative_cooling
        )

        # If the other potential exchanges could change sign, we'd need
        # something like thisand the code below to calculate convective
        # exchange as the residual
        # total_exchanges = (
        #     self.groundwater_conduction + self.longwave_vegetation
        # )

        # Convection closes the balance (can be positive or negative)
        self.convective_exchange[:] = total_outputs - total_inputs
        # Sign flipped to be an exchange: positive = gain, negative = loss
        # self.convective_exchange[:] = (
        #     total_outputs - total_inputs - total_exchanges
        # )

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
            hs: Net shortwave solar radiation (W/m²)
            ha: Atmospheric longwave radiation (W/m²)
            hf: Friction heating (W/m²)
            hv: Vegetation longwave radiation (W/m²)
            evap: Evaporation rate (m/s)
            t_abs4: Temperature to 4th power (K^4)
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
        atmos_exchange_factor=None,
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
        if atmos_exchange_factor is None:
            atmos_exchange_factor = self._atmos_exchange_factor
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
            atmos_exchange_factor,
        )

    @staticmethod
    def _update_running_avg_temp(
        segment_order,
        seg_tave_water,
        seginc_swrad,
        seg_tave_air,
        gw_tau,
        gw_index,
        gw_silo,
        gw_sum,
        seg_tave_gw,
        ss_tau,
        ss_index,
        ss_silo,
        ss_sum,
        seg_tave_ss,
    ):
        """Update running average temperatures for groundwater and subsurface.

        This function can be optionally JIT-compiled with numba for
        performance.

        Args:
            segment_order: Order to process segments (immutable)
            seg_tave_water: Water temperature array (immutable, for skip
                checks)
            seginc_swrad: Solar radiation array (immutable, for skip checks)
            seg_tave_air: Air temperature array (immutable)
            gw_tau: Groundwater tau values (immutable)
            gw_index: Groundwater circular buffer indices (MUTATED)
            gw_silo: Groundwater circular buffer (MUTATED)
            gw_sum: Groundwater running sums (MUTATED)
            seg_tave_gw: Groundwater temperatures (MUTATED - output)
            ss_tau: Subsurface tau values (immutable)
            ss_index: Subsurface circular buffer indices (MUTATED)
            ss_silo: Subsurface circular buffer (MUTATED)
            ss_sum: Subsurface running sums (MUTATED)
            seg_tave_ss: Subsurface temperatures (MUTATED - output)
        """
        for jj in segment_order:
            # Skip if marked as permanently invalid (no HRUs
            # upstream/downstream)
            if seginc_swrad[jj] < -99.0:
                continue

            # Update groundwater running average
            idx_gw = gw_index[jj]
            tau_gw = int(gw_tau[jj])

            # Add new air temperature to silo
            gw_silo[jj, idx_gw] = seg_tave_air[jj]

            # Recompute sum and count valid entries (matches Fortran)
            # This handles spin-up period correctly
            at_sum = 0.0
            at_cnt = 0
            for j in range(tau_gw):
                if gw_silo[jj, j] > -98.0:
                    at_sum += gw_silo[jj, j]
                    at_cnt += 1

            # Compute average as sum / count
            if at_cnt > 0:
                seg_tave_gw[jj] = at_sum / at_cnt
            else:
                seg_tave_gw[jj] = 0.0

            # Update index
            if idx_gw < tau_gw - 1:
                gw_index[jj] = idx_gw + 1
            else:
                gw_index[jj] = 0

            # Update subsurface running average
            idx_ss = ss_index[jj]
            tau_ss = int(ss_tau[jj])

            # Add new air temperature to silo
            ss_silo[jj, idx_ss] = seg_tave_air[jj]

            # Recompute sum and count valid entries (matches Fortran)
            # This handles spin-up period correctly
            at_sum = 0.0
            at_cnt = 0
            for j in range(tau_ss):
                if ss_silo[jj, j] > -98.0:
                    at_sum += ss_silo[jj, j]
                    at_cnt += 1

            # Compute average as sum / count
            if at_cnt > 0:
                seg_tave_ss[jj] = at_sum / at_cnt
            else:
                seg_tave_ss[jj] = 0.0

            # Update index
            if idx_ss < tau_ss - 1:
                ss_index[jj] = idx_ss + 1
            else:
                ss_index[jj] = 0

    @staticmethod
    def _compute_lateral_temp(
        segment_order,
        seg_tave_water,
        seginc_swrad,
        seg_lateral_inflow,
        seginc_sroff,
        seginc_ssflow,
        seginc_gwflow,
        seg_melt,
        seg_rain,
        seg_tave_gw,
        seg_tave_air,
        seg_tave_ss,
        melt_temp,
        lat_temp_adj,
        nowmonth,
        seg_tave_lat,
    ):
        """Compute lateral flow temperatures for all segments.

        This function can be optionally JIT-compiled with numba for
        performance.

        Args:
            segment_order: Order to process segments (immutable)
            seg_tave_water: Water temperature array (immutable, for skip
                checks)
            seginc_swrad: Solar radiation array (immutable, for skip checks)
            seg_lateral_inflow: Lateral inflow array (immutable)
            seginc_sroff: Surface runoff array (immutable)
            seginc_ssflow: Subsurface flow array (immutable)
            seginc_gwflow: Groundwater flow array (immutable)
            seg_melt: Snowmelt array (immutable)
            seg_rain: Rainfall array (immutable)
            seg_tave_gw: Groundwater temperature array (immutable)
            seg_tave_air: Air temperature array (immutable)
            seg_tave_ss: Subsurface temperature array (immutable)
            melt_temp: Melt temperature constant (immutable)
            lat_temp_adj: Monthly lateral temperature adjustment array
                (immutable)
            nowmonth: Current month (immutable, 1-based)
            seg_tave_lat: Lateral temperature array (MUTATED - output)
        """
        for jj in segment_order:
            # Skip if marked as never having flow (NaN = never has flow)
            if np.isnan(seg_tave_water[jj]):
                continue
            # Skip if marked as permanently invalid
            if seginc_swrad[jj] < -99.0:
                continue

            # Get segment values
            sroff = seginc_sroff[jj]
            ssflow = seginc_ssflow[jj]
            gwflow = seginc_gwflow[jj]
            melt = seg_melt[jj]
            rain = seg_rain[jj]
            tave_gw = seg_tave_gw[jj]
            tave_air = seg_tave_air[jj]
            tave_ss = seg_tave_ss[jj]

            # Use lat_inflow function for detailed lateral temperature
            # calculation
            tl_avg, qlat = _lat_inflow(
                seg_lateral_inflow[jj],
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
                tl_avg += lat_temp_adj[nowmonth - 1, jj]

            # Ensure non-negative (also converts NaN to 0.0 to match Fortran)
            if np.isnan(tl_avg) or tl_avg < 0.0:
                tl_avg = 0.0

            seg_tave_lat[jj] = tl_avg

    @staticmethod
    def _compute_water_temp(
        segment_order,
        seg_tave_water,
        seginc_swrad,
        seg_outflow,
        upstream_count,
        upstream_idx,
        seg_lateral_inflow,
        seg_tave_upstream,
        seg_svi_all,
        seg_inflow,
        seg_tave_lat,
        seginc_swrad_data,
        seg_humid,
        seg_elev,
        seg_potet,
        seg_shade,
        seg_ccov,
        seg_flow_width,
        seg_slope,
        seg_tave_gw,
        seg_length,
        albedo,
        maxiter_sntemp,
        track_energy_fluxes,
        hs_terms,
        ha_terms,
        hf_terms,
        hv_terms,
        evap_terms,
        t_abs4_terms,
        upstream_flows,
        lateral_flows,
        atmos_exchange_factor=1.0,
    ):
        """Compute water temperature for all segments.

        This function can be optionally JIT-compiled with numba for
        performance.

        Args:
            segment_order: Order to process segments (immutable)
            seg_tave_water: Water temperature array (MUTATED - input/output)
            seginc_swrad: Solar radiation array (immutable, for skip checks)
            seg_outflow: Segment outflow array (immutable)
            upstream_count: Number of upstream segments for each segment
                (immutable)
            upstream_idx: Indices of upstream segments (immutable)
            seg_lateral_inflow: Lateral inflow array (immutable)
            seg_tave_upstream: Upstream temperature array (MUTATED - output)
            seg_svi_all: Pre-computed vegetation shade index array (immutable)
            seg_inflow: Segment inflow array (MUTATED - output)
            seg_tave_lat: Lateral temperature array (immutable)
            seginc_swrad_data: Solar radiation data for energy balance
                (immutable)
            seg_humid: Humidity array (immutable)
            seg_elev: Elevation array (immutable)
            seg_potet: Potential ET array (immutable)
            seg_shade: Shade fraction array (immutable)
            seg_ccov: Cloud cover array (immutable)
            seg_flow_width: Flow width array (immutable)
            seg_slope: Slope array (immutable)
            seg_tave_gw: Groundwater temperature array (immutable)
            seg_length: Segment length array (immutable)
            albedo: Albedo value (immutable)
            maxiter_sntemp: Maximum iterations for temperature solver
                (immutable)
            track_energy_fluxes: Whether to track energy flux components
                (immutable)
            hs_terms: Solar radiation terms (MUTATED if tracking - output)
            ha_terms: Atmospheric longwave terms (MUTATED if tracking - output)
            hf_terms: Friction heat terms (MUTATED if tracking - output)
            hv_terms: Vegetation longwave terms (MUTATED if tracking - output)
            evap_terms: Evaporation terms (MUTATED if tracking - output)
            t_abs4_terms: Temperature^4 terms (MUTATED if tracking - output)
            upstream_flows: Upstream flow values (MUTATED if tracking - output)
            lateral_flows: Lateral flow values (MUTATED if tracking - output)
            atmos_exchange_factor: Factor to amplify atmospheric exchange
                (immutable, default=1.0)
        """
        for jj in segment_order:
            # Skip segments marked as never having flow (NaN = never has flow)
            if np.isnan(seg_tave_water[jj]):
                continue

            # Compute upstream temperature and flow in one pass
            # (consolidates _compute_upstream_temp and _compute_inflow)
            flow_sum = 0.0
            temp_sum = 0.0
            upstream_flow = 0.0

            for kk in range(upstream_count[jj]):
                up_idx = upstream_idx[jj, kk]
                if not np.isnan(seg_tave_water[up_idx]):
                    flow = seg_outflow[up_idx]
                    temp_sum += seg_tave_water[up_idx] * flow
                    flow_sum += flow
                    upstream_flow += flow

            if flow_sum > 0.0:
                seg_tave_upstream[jj] = temp_sum / flow_sum
            else:
                seg_tave_upstream[jj] = NOFLOW_TEMP

            # Get svi from pre-computed array
            svi = seg_svi_all[jj]

            # Compute total inflow
            seg_inflow[jj] = upstream_flow + seg_lateral_inflow[jj]

            # Now compute water temperature (logic from _compute_water_temp)

            # Skip if marked as permanently invalid
            if seginc_swrad[jj] < -99.0:
                seg_tave_water[jj] = np.nan
                continue

            # Check for no-flow conditions
            if seg_outflow[jj] <= 0.0:
                seg_tave_water[jj] = NOFLOW_TEMP
                continue

            lateral_flow = seg_lateral_inflow[jj]
            qlat = lateral_flow * CFS_TO_CMS

            # Match Fortran: check qlat in CMS (not lateral_flow in CFS)
            if upstream_flow * CFS_TO_CMS <= NEARZERO and qlat <= NEARZERO:
                seg_tave_water[jj] = NOFLOW_TEMP
                continue

            # Compute mixed inlet temperature
            t_in = _compute_mixed_inlet_temp(
                upstream_flow,
                lateral_flow,
                seg_tave_upstream[jj],
                seg_tave_lat[jj],
            )

            if np.isnan(t_in):
                seg_tave_water[jj] = NOFLOW_TEMP
                continue

            # Compute equilibrium temperature using full energy balance
            result = _equilb(
                t_in,
                svi,
                seg_inflow[jj],
                seginc_swrad_data[jj],
                seg_humid[jj],
                seg_elev[jj],
                seg_potet[jj],
                seg_shade[jj],
                seg_ccov[jj],
                seg_flow_width[jj],
                seg_slope[jj],
                seg_tave_gw[jj],
                albedo,
                maxiter_sntemp,
            )

            te = result[0]
            ak1 = result[1]
            ak2 = result[2]

            # Store energy flux terms if tracking
            if track_energy_fluxes:
                hs_terms[jj] = result[3]
                ha_terms[jj] = result[4]
                hf_terms[jj] = result[5]
                hv_terms[jj] = result[6]
                evap_terms[jj] = result[7]
                t_abs4_terms[jj] = result[8]
                upstream_flows[jj] = upstream_flow
                lateral_flows[jj] = lateral_flow

            # Compute final temperature using twavg function
            qup = upstream_flow
            tl_avg = seg_tave_lat[jj]

            seg_tave_water[jj] = _twavg(
                qup,
                t_in,
                qlat,
                tl_avg,
                te,
                ak1,
                ak2,
                seg_flow_width[jj],
                seg_length[jj],
                atmos_exchange_factor,
            )


class PRMSStreamTemp(PRMSStreamTempHumidityCBH):
    """PRMS stream temperature with monthly segment humidity parameter.

    Corresponds to ``strmtemp_humidity_flag=1`` where humidity is supplied via
    the ``seg_humidity`` parameter (12 monthly values per segment) rather than
    a time-varying CBH input. All other behaviour is identical to
    :class:`PRMSStreamTempHumidityCBH`.

    Args:
        control: a Control object
        discretization: a discretization of class Parameters
        parameters: a parameter object of class Parameters — must include
            ``seg_humidity`` with shape ``(nmonth, nsegment)``
        seg_outflow: Streamflow leaving each segment (from PRMSChannel)
        seg_lateral_inflow: Lateral inflow entering each segment
        swrad: Solar radiation for each HRU
        potet: Potential ET for each HRU
        sroff: Surface runoff for each HRU
        ssres_flow: Subsurface flow for each HRU
        gwres_flow: Groundwater flow for each HRU
        tavgc: Average air temperature for each HRU in Celsius
        snowmelt: Snowmelt for each HRU
        hru_rain: Rainfall for each HRU
        soltab_potsw: Potential shortwave radiation table for current day
        seg_flow_width: Flow-dependent width from PRMSHydraulicGeometryFull
        seg_flow_depth: Flow-dependent depth from PRMSHydraulicGeometryFull
        seg_flow_area: Flow-dependent cross-sectional area
        seg_flow_velocity: Flow-dependent velocity
        stream_shade: PRMSStreamShade instance
        stream_shade_class: Class to use for stream shade computation
        stream_shade_parameters: Parameters for stream shade computation
        imbalance_behavior: one of ["defer", None, "warn", "error"]
        verbose: Print extra information or not?
        use_vectorized_shade: Use vectorized shade computation
        track_energy_fluxes: Track energy flux terms for budget analysis
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
        tavgc: adaptable,
        snowmelt: adaptable,
        hru_rain: adaptable,
        soltab_potsw: adaptable,
        seg_flow_width: adaptable = None,
        seg_flow_depth: adaptable = None,
        seg_flow_area: adaptable = None,
        seg_flow_velocity: adaptable = None,
        stream_shade: Union[PRMSStreamShade, None] = None,
        stream_shade_class: Union[type, None] = None,
        stream_shade_parameters: Union[Parameters, Path, None] = None,
        calc_method: Literal["numba", "numpy", None] = None,
        imbalance_behavior: Literal["defer", None, "warn", "error"] = "defer",
        input_aliases: dict = None,
        verbose: bool = False,
        use_vectorized_shade: bool = True,
        track_energy_fluxes: bool = True,
        atmos_exchange_factor: float = 1.0,
    ) -> None:
        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
            seg_outflow=seg_outflow,
            seg_lateral_inflow=seg_lateral_inflow,
            swrad=swrad,
            potet=potet,
            sroff=sroff,
            ssres_flow=ssres_flow,
            gwres_flow=gwres_flow,
            tavgc=tavgc,
            snowmelt=snowmelt,
            hru_rain=hru_rain,
            soltab_potsw=soltab_potsw,
            humidity_hru=None,
            seg_flow_width=seg_flow_width,
            seg_flow_depth=seg_flow_depth,
            seg_flow_area=seg_flow_area,
            seg_flow_velocity=seg_flow_velocity,
            stream_shade=stream_shade,
            stream_shade_class=stream_shade_class,
            stream_shade_parameters=stream_shade_parameters,
            calc_method=calc_method,
            imbalance_behavior=imbalance_behavior,
            input_aliases=input_aliases,
            verbose=verbose,
            use_vectorized_shade=use_vectorized_shade,
            track_energy_fluxes=track_energy_fluxes,
            atmos_exchange_factor=atmos_exchange_factor,
        )
        self.name = "PRMSStreamTemp"

    @staticmethod
    def get_dimensions() -> tuple:
        return ("nhru", "nsegment", "nmonth", "ndoy")

    @staticmethod
    def get_parameters() -> tuple:
        return (
            "doy",
            "hru_segment",
            "hru_area",
            "hru_slope",
            "tosegment",
            "albedo",
            "lat_temp_adj",
            "seg_length",
            "seg_slope",
            "seg_lat",
            "seg_elev",
            "seg_close",
            "ss_tau",
            "gw_tau",
            "melt_temp",
            "maxiter_sntemp",
            "stream_tave_init",
            "seg_humidity",
        )

    @staticmethod
    def get_inputs() -> tuple:
        return (
            "seg_outflow",
            "seg_lateral_inflow",
            "seg_flow_width",
            "seg_flow_depth",
            "seg_flow_area",
            "seg_flow_velocity",
            "swrad",
            "potet",
            "sroff",
            "ssres_flow",
            "gwres_flow",
            "tavgc",
            "snowmelt",
            "hru_rain",
            "soltab_potsw",
        )

    def _compute_segment_aggregates(self) -> None:
        """Compute segment aggregates using monthly seg_humidity parameter.

        Calls _compute_segment_aggregates_numba for all non-humidity variables,
        then sets seg_humid directly from the seg_humidity parameter for the
        current month (strmtemp_humidity_flag=1 path).
        """
        _compute_segment_aggregates_numba(
            self.nhru,
            self.nsegment,
            self.hru_segment,
            self.hru_area,
            self.sroff,
            self.ssres_flow,
            self.gwres_flow,
            self.swrad,
            self.segment_hruarea,
            self.segment_up,
            self.tosegment,
            self.seginc_sroff,
            self.seginc_ssflow,
            self.seginc_gwflow,
            self.seginc_swrad,
            self.tavgc,
            self.snowmelt,
            self.hru_rain,
            self.soltab_potsw,
            self._hru_cossl,
            self.segment_order,
            self.seg_close,
            self.seg_tave_air,
            self.seg_melt,
            self.seg_rain,
            self.seg_ccov,
        )

        # Set seg_humid from monthly parameter.
        # seg_humidity has dims (nmonth, nsegment); Nowmonth is 1-based.
        # Matches PRMS stream_temp.f90 flag==1 path:
        #   Seg_humid(i) = Seg_humidity(i, Nowmonth)
        nowmonth = self.control.current_month
        self.seg_humid[:] = self.seg_humidity[nowmonth - 1, :]


@nb.jit(nopython=True)
def _compute_seg_humid_cbh_numba(
    nhru,
    nsegment,
    hru_segment,
    hru_area,
    humidity_hru,
    segment_hruarea,
    segment_order,
    seg_close,
    seg_humid,
):
    """Compute seg_humid from CBH humidity_hru input.

    Implements the flag==0 path from PRMS 5.2.1.1 stream_temp.f90:
    - Reset seg_humid to 0 (ELSE branch, reinstated fix)
    - Accumulate area-weighted humidity from HRUs, converting percent->fraction
    - Normalise by segment HRU area in a single segment_order pass, copying
      from seg_close for segments with no HRUs (matching PRMS single-pass).

    Args:
        nhru: Number of HRUs (immutable)
        nsegment: Number of segments (immutable)
        hru_segment: HRU to segment mapping (immutable, 1-based)
        hru_area: HRU areas in acres (immutable)
        humidity_hru: HRU relative humidity in percent 0-100 (immutable)
        segment_hruarea: Total HRU area per segment (immutable)
        segment_order: Order to process segments (immutable)
        seg_close: Closest segment with HRUs for each segment (immutable)
        seg_humid: Segment humidity in decimal fraction (MUTATED - output)
    """
    # Reset before accumulation — matches corrected PRMS 5.2.1.1 ELSE branch.
    # See humidity_bug_prms_fix.md "Secondary Bug" for full rationale.
    seg_humid[:] = 0.0

    # Accumulate area-weighted humidity from HRUs.
    # *0.01 applied during accumulation (percent -> decimal fraction) so units
    # are consistent throughout (stream_temp.f90 line 807):
    #   Seg_humid(i) = Seg_humid(i) + Humidity_hru(j) * 0.01 * harea
    for j in range(nhru):
        seg_idx = hru_segment[j]
        if seg_idx > 0:
            i = seg_idx - 1
            seg_humid[i] += humidity_hru[j] * 0.01 * hru_area[j]

    # Normalise in a single segment_order pass (matches PRMS single-pass).
    # For no-HRU segments, copy from seg_close in the same pass so that the
    # ordering matches PRMS exactly (stream_temp.f90 lines 829-844).
    for jj in range(nsegment):
        i = segment_order[jj]
        if segment_hruarea[i] > NEARZERO:
            seg_humid[i] /= segment_hruarea[i]
        else:
            seg_humid[i] = seg_humid[seg_close[i]]


@nb.jit(nopython=True)
def _compute_segment_aggregates_numba(
    nhru,
    nsegment,
    hru_segment,
    hru_area,
    sroff,
    ssres_flow,
    gwres_flow,
    swrad,
    segment_hruarea,
    segment_up,
    tosegment,
    seginc_sroff,
    seginc_ssflow,
    seginc_gwflow,
    seginc_swrad,
    # Meteorological aggregation inputs (excluding humidity)
    tavgc,
    snowmelt,
    hru_rain,
    soltab_potsw,
    hru_cossl,
    segment_order,
    seg_close,
    # Output arrays for meteorological variables (excluding seg_humid)
    seg_tave_air,
    seg_melt,
    seg_rain,
    seg_ccov,
):
    """Compute segment aggregate variables from HRU inputs.

    Humidity (seg_humid) is handled separately by _compute_seg_humid_cbh_numba
    (flag==0) or set directly from a parameter (flag==1).

    Args:
        nhru: Number of HRUs (immutable)
        nsegment: Number of segments (immutable)
        hru_segment: HRU to segment mapping (immutable, 1-based)
        hru_area: HRU areas (immutable)
        sroff: Surface runoff from HRUs (immutable)
        ssres_flow: Subsurface flow from HRUs (immutable)
        gwres_flow: Groundwater flow from HRUs (immutable)
        swrad: Solar radiation from HRUs (immutable)
        segment_hruarea: Total HRU area per segment (immutable)
        segment_up: Upstream segment indices (immutable, 0-based)
        tosegment: Downstream segment indices (immutable, 1-based)
        seginc_sroff: Segment surface runoff (MUTATED - output)
        seginc_ssflow: Segment subsurface flow (MUTATED - output)
        seginc_gwflow: Segment groundwater flow (MUTATED - output)
        seginc_swrad: Segment solar radiation (MUTATED - output)
        tavgc: HRU average temperature in Celsius (immutable)
        snowmelt: HRU snowmelt (immutable)
        hru_rain: HRU rainfall (immutable)
        soltab_potsw: Potential shortwave radiation for current day (immutable)
        hru_cossl: Cosine of HRU slope (immutable)
        segment_order: Order to process segments (immutable)
        seg_close: Closest segment with HRUs for each segment (immutable)
        seg_tave_air: Segment air temperature (MUTATED - output)
        seg_melt: Segment snowmelt (MUTATED - output)
        seg_rain: Segment rainfall (MUTATED - output)
        seg_ccov: Segment cloud cover (MUTATED - output)
    """
    # Initialize segment aggregate variables
    seginc_sroff[:] = 0.0
    seginc_ssflow[:] = 0.0
    seginc_gwflow[:] = 0.0
    seginc_swrad[:] = 0.0

    # Initialize meteorological segment variables
    seg_tave_air[:] = 0.0
    seg_melt[:] = 0.0
    seg_rain[:] = 0.0
    seg_ccov[:] = 0.0

    # Constants (from PRMS_SET_TIME)
    # Cfs_conv converts acre-inches/day to cfs
    # FT2_PER_ACRE / INCHES_PER_FOOT / SECS_PER_DAY
    # = 43560 / 12 / 86400
    cfs_conv = 43560.0 / 12.0 / 86400.0

    # Aggregate HRU values to segments
    for j in range(nhru):
        seg_idx = hru_segment[j]

        # Check if HRU contributes to a segment (seg_idx > 0)
        # hru_segment is 1-based, so valid segments are > 0
        if seg_idx > 0:
            # Convert to 0-based index
            i = seg_idx - 1
            harea = hru_area[j]

            # Convert from inches to cfs (area * inches/day * cfs_conv)
            tocfs = harea * cfs_conv

            # Accumulate flow components (converted to cfs)
            seginc_sroff[i] += sroff[j] * tocfs
            seginc_ssflow[i] += ssres_flow[j] * tocfs
            seginc_gwflow[i] += gwres_flow[j] * tocfs

            # Accumulate area-weighted radiation
            seginc_swrad[i] += swrad[j] * harea

            # Compute cloud cover for this HRU (stream_temp.f90 line 760-778)
            # ccov = 1.0 - (swrad / soltab_potsw * hru_cossl)
            potsw = soltab_potsw[j]
            if potsw <= 10.0:
                ccov = 1.0 - (swrad[j] / 10.0 * hru_cossl[j])
            else:
                ccov = 1.0 - (swrad[j] / potsw * hru_cossl[j])

            # Clamp ccov to [0, 1]
            if ccov < NEARZERO:
                ccov = 0.0
            elif ccov > 1.0:
                ccov = 1.0

            # Accumulate area-weighted meteorological variables
            seg_tave_air[i] += tavgc[j] * harea
            seg_ccov[i] += ccov * harea
            seg_melt[i] += snowmelt[j] * harea
            seg_rain[i] += hru_rain[j] * harea

    # Process seginc_swrad in numerical order first (matches original logic)
    # Divide radiation by segment HRU area to get averages
    # Process in numerical order to match routing.f90 (line 741-810)
    for i in range(nsegment):
        if segment_hruarea[i] > NEARZERO:
            seginc_swrad[i] /= segment_hruarea[i]

        else:
            # Segment has no HRUs - search upstream then downstream
            # (matches routing.f90 line 746-805)
            # Search upstream first (routing.f90 line 749-772)
            this_seg = i
            found = False
            max_iter = nsegment  # Prevent infinite loops
            iter_count = 0

            while not found and iter_count < max_iter:
                iter_count += 1

                # segment_up contains 0-based indices where 0 means
                # "no upstream". However, if segment 0 is upstream, we can't
                # distinguish it from "no upstream". The original code assumes
                # segment 0 is never upstream of anything
                upstream_seg = segment_up[this_seg]

                # Check if headwater (no upstream)
                if upstream_seg == 0 and this_seg != 0:
                    # No upstream segment exists
                    break
                elif upstream_seg == 0 and this_seg == 0:
                    # this_seg is segment 0, which has no upstream
                    break

                # Move to upstream segment (already 0-based)
                this_seg = upstream_seg

                if segment_hruarea[this_seg] > NEARZERO:
                    # Found segment with HRUs - compute average from
                    # accumulated value
                    seginc_swrad[i] = (
                        seginc_swrad[this_seg] / segment_hruarea[this_seg]
                    )
                    found = True
                    break

            # If not found upstream, search downstream
            # (routing.f90 line 776-800)
            if not found:
                this_seg = i
                iter_count = 0

                while not found and iter_count < max_iter:
                    iter_count += 1

                    # tosegment is 1-based, 0 means no downstream
                    downstream_seg = tosegment[this_seg]

                    # Check if terminal segment (no downstream)
                    if downstream_seg == 0:
                        break

                    # Move to downstream segment (convert 1-based to 0-based)
                    this_seg = downstream_seg - 1

                    if segment_hruarea[this_seg] > NEARZERO:
                        # Found segment with HRUs - compute average from
                        # accumulated value
                        seginc_swrad[i] = (
                            seginc_swrad[this_seg] / segment_hruarea[this_seg]
                        )
                        found = True
                        break

            # If still not found, set to invalid marker
            # (routing.f90 line 803-805)
            if not found:
                seginc_swrad[i] = -99.9

    # Process meteorological variables in single pass (matches Fortran).
    # Process in segment_order - for segments without HRUs, copy from
    # seg_close. Note: seg_close may point to upstream/downstream segments
    # not yet processed in some edge cases, but this matches the Fortran
    # single-pass behavior (stream_temp.f90 lines 820-844).
    for jj in range(nsegment):
        i = segment_order[jj]

        if segment_hruarea[i] > NEARZERO:
            # Segment has HRUs - compute area-weighted averages
            seg_tave_air[i] /= segment_hruarea[i]
            seg_ccov[i] /= segment_hruarea[i]
            seg_melt[i] /= segment_hruarea[i]
            seg_rain[i] /= segment_hruarea[i]
        else:
            # Segment has no HRUs - use values from seg_close
            # (stream_temp.f90 lines 832-844)
            close_seg = seg_close[i]
            seg_tave_air[i] = seg_tave_air[close_seg]
            seg_ccov[i] = seg_ccov[close_seg]
            seg_melt[i] = seg_melt[close_seg]
            seg_rain[i] = seg_rain[close_seg]


@nb.jit(nopython=True)
def _compute_seg_potet_numba(
    nhru,
    nsegment,
    hru_segment,
    hru_area,
    potet,
    segment_order,
    segment_hruarea,
    seg_close,
    seg_potet,
):
    """Compute seg_potet using stream_temp.f90 logic.

    This numba-optimized function replaces _compute_seg_potet.

    Args:
        nhru: Number of HRUs (immutable)
        nsegment: Number of segments (immutable)
        hru_segment: HRU to segment mapping (immutable, 1-based)
        hru_area: HRU areas (immutable)
        potet: Potential ET from HRUs (immutable)
        segment_order: Order to process segments (immutable)
        segment_hruarea: Total HRU area per segment (immutable)
        seg_close: Closest segment with HRUs for each segment (immutable)
        seg_potet: Segment potential ET (MUTATED - output)
    """
    # Initialize
    seg_potet[:] = 0.0

    # Accumulate from HRUs
    for j in range(nhru):
        seg_idx = hru_segment[j]
        if seg_idx > 0:
            i = seg_idx - 1
            seg_potet[i] += potet[j] * hru_area[j]

    # Process in segment_order
    for jj in range(nsegment):
        i = segment_order[jj]

        if segment_hruarea[i] > NEARZERO:
            seg_potet[i] /= segment_hruarea[i]

        else:
            # Segment has no HRUs - use seg_close
            close_seg = seg_close[i]
            seg_potet[i] = seg_potet[close_seg]


@nb.jit(nopython=True)
def _teak1(a_coef, b_coef, c_coef, d_coef, teq, maxiter_sntemp):
    """Solve for equilibrium temperature using Newton-Raphson iteration.

    This is the teak1 function from PRMS.

    Args:
        a_coef: Coefficient (immutable)
        b_coef: Coefficient (immutable)
        c_coef: Coefficient (immutable)
        d_coef: Coefficient (immutable)
        teq: Initial guess for equilibrium temperature (immutable)
        maxiter_sntemp: Maximum iterations (immutable)

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
        t_o: Initial temperature (immutable, degC)
        svi: Vegetation shade index (immutable)
        seg_inflow: Segment inflow (immutable, CFS)
        seginc_swrad: Incident shortwave radiation (immutable)
        seg_humid: Segment humidity (immutable)
        seg_elev: Segment elevation (immutable)
        seg_potet: Segment potential ET (immutable)
        seg_shade: Segment shade fraction (immutable)
        seg_ccov: Cloud cover fraction (immutable)
        seg_flow_width: Flow width (immutable)
        seg_slope: Segment slope (immutable)
        seg_tave_gw: Groundwater temperature (immutable)
        albedo: Albedo (immutable)
        maxiter_sntemp: Maximum iterations (immutable)

    Returns:
        te: Equilibrium temperature (degC)
        ak1: First-order thermal exchange coefficient
        ak2: Second-order thermal exchange coefficient
        hs: Net shortwave solar radiation (W/m²)
        ha: Atmospheric longwave radiation (W/m²)
        hf: Friction heating (W/m²)
        hv: Vegetation longwave radiation (W/m²)
        evap: Evaporation rate (m/s)
        t_abs4: Temperature to 4th power (K^4)
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
    # Note: Fortran uses Seg_humid directly, not foo (clamped for bow_coeff)
    ha = (
        (3.354939e-8 + 2.74995e-9 * np.sqrt(seg_humid * vp_sat))
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

    return ted, ak1d, ak2d, hs, ha, hf, hv, evap, taabs**4


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
    atmos_exchange_factor=1.0,
):
    """Compute average water temperature with lateral inflows.

    This is the twavg function from PRMS.

    Args:
        qup: Upstream flow (immutable, cfs)
        t0: Inlet temperature (immutable, degC)
        qlat: Lateral flow (immutable, cms)
        tl_avg: Lateral flow temperature (immutable, degC)
        te: Equilibrium temperature (immutable, degC)
        ak1: First-order thermal exchange coefficient (immutable)
        ak2: Second-order thermal exchange coefficient (immutable)
        seg_flow_width: Flow width (immutable)
        seg_length: Segment length (immutable)
        atmos_exchange_factor: Factor to amplify atmospheric exchange effects
            (immutable, default=1.0)

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
        b = (ak1 * atmos_exchange_factor * width) / 4182.0e03
        rexp = -1.0 * (b * length) / q_init
        r = np.exp(rexp)

    elif ql < 0.0:
        # Losing stream (should not happen in PRMS)
        tep = te
        b = (ql / length) + ((ak1 * atmos_exchange_factor * width) / 4182.0e03)
        rexp = (ql - (b * length)) / ql
        r = 1.0 + (ql / q_init)
        r = r**rexp

    elif ql > NEARZERO and q_init <= NEARZERO:
        tep = te
        b = (ak1 * atmos_exchange_factor * width) / 4182.0e03
        rexp = -1.0 * (b * length) / ql
        r = np.exp(rexp)

    else:
        b = (ql / length) + ((ak1 * atmos_exchange_factor * width) / 4182.0e03)
        tep = (
            ((ql / length) * tl_avg)
            + (((ak1 * atmos_exchange_factor * width) / (4182.0e03)) * te)
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

    if np.abs(denom) < NEARZERO:
        denom = np.sign(denom) * NEARZERO if denom != 0.0 else NEARZERO

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
        seg_lateral_inflow: Total lateral inflow to segment (immutable, cfs)
        seginc_sroff: Surface runoff component (immutable, cfs)
        seginc_ssflow: Subsurface flow component (immutable, cfs)
        seginc_gwflow: Groundwater flow component (immutable, cfs)
        melt_temp: Snowmelt temperature (immutable, degC)
        tave_gw: Groundwater temperature (immutable, degC)
        tave_air: Air temperature (immutable, degC)
        tave_ss: Subsurface temperature (immutable, degC)
        melt: Snowmelt (immutable, inches)
        rain: Rainfall (immutable, inches)

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
        upstream_flow: Flow from upstream segments (immutable, cfs)
        lateral_flow: Lateral inflow (immutable, cfs)
        seg_tave_upstream: Upstream temperature (immutable, degC)
        seg_tave_lat: Lateral flow temperature (immutable, degC)

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
