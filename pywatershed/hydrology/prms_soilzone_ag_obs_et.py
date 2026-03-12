"""PRMS Soilzone with agricultural area and iterative AET matching.

This module implements PRMSSoilzoneAg, a soil zone process that extends the
base PRMS soilzone with:

1. Dual area treatment (pervious and agricultural areas with separate
   accounting)
2. Iterative AET matching using observed AET data
3. Agricultural-specific parameters and irrigation estimation

Fortran Source Reference
------------------------
Based on GSFLOW 2.4.0 (currently unreleased) soilzone_ag.f90 with the following
simplifications:

**Implemented:**
- Dual soil moisture accounting (pervious and agricultural areas)
- Iterative loop to match observed AET via irrigation adjustment
- Preferential flow reservoir (same as base soilzone)
- Slow (gravity) reservoir with interflow and groundwater flow
- Agricultural-specific soil properties and cover types
- Transpiration and evaporation from soil zones

**Neglected functionality:**
- GLACIER HRU type logic
- GSFLOW_flag==ACTIVE logic (gravity reservoir interaction with MODFLOW)
- Cascade_flag > CASCADE_OFF logic (cascading flow between HRUs)
- compute_lateral==OFF logic (swales with lateral flow)
- Basin-wide aggregation variables (Basin_*)
- MODSIM_PRMS and MODSIM_PRMS_LOOSE model integration
- Frozen HRU logic (CFGI - Conditional Frozen Ground Index)

**Main Fortran Cross-References in code:**
- Main loop: szrun_ag() lines ~428-1219
- Iteration control: DO WHILE (keep_iterating==ACTIVE) lines ~442-1187
- HRU loop: DO k = 1, Active_hrus lines ~552-1178
- Pervious soil moisture: compute_soilmoist() lines ~752-758
- Agricultural soil moisture: compute_soilmoist() lines ~760-767
- Agricultural ET: compute_szactet() lines ~990-1005
- Pervious ET: compute_szactet() lines ~1011-1017
- Irrigation estimation: lines ~1127-1160
- Convergence check: lines ~1180-1187

Implementation Notes
--------------------
- The iteration loop adjusts `ag_irrigation_add` to minimize the difference
  between computed agricultural ET and observed AET
- Convergence criteria: unsatisfied_ag_et <= soilzone_aet_converge
- Maximum iterations: max_soilzone_ag_iter (default 10)
- After 20 iterations, doubles the irrigation increment to speed convergence
- Only iterates when iter_aet_flag=True and transp_on=True
- Agricultural area fraction (ag_frac) can be zero for non-agricultural HRUs

"""

import pathlib as pl
from typing import Literal
from warnings import warn

import numba as nb
import numpy as np

from ..base.adapter import adaptable
from ..base.conservative_process import ConservativeProcess
from ..base.control import Control
from ..constants import (
    HruType,
    nan,
    numba_num_threads,
    one,
    zero,
)
from ..parameters import Parameters

ONETHIRD = 1 / 3
TWOTHIRDS = 2 / 3


class PRMSSoilzoneAgObsET(ConservativeProcess):
    """PRMS soil zone with agricultural area and iterative AET matching.

    This is an agricultural variant of PRMSSoilzone based on GSFLOW 2.4.0
    soilzone_ag.f90. The key differences from the base PRMSSoilzone are:

    1. **Dual Area Treatment**: Each HRU is divided into pervious and
       agricultural/irrigated areas, each with separate soil moisture
       accounting.

    2. **Iterative AET Matching**: When observed actual ET (AET_external) is
       provided, the code iteratively adjusts irrigation to match the observed
       AET within a convergence tolerance.

    3. **Agricultural Parameters**: Additional parameters for agricultural soil
       properties, cover density, and irrigation thresholds.

    Implementation based on GSFLOW 2.4.0 soilzone_ag.f90 with theoretical
    documentation given in the PRMS-IV documentation:

    `Markstrom, S. L., Regan, R. S., Hay, L. E., Viger, R. J., Webb, R. M.,
    Payn, R. A., & LaFontaine, J. H. (2015). PRMS-IV, the
    precipitation-runoff modeling system, version 4. US Geological Survey
    Techniques and Methods, 6, B7.
    <https://pubs.usgs.gov/tm/6b7/pdf/tm6-b7.pdf>`__

    Args:
        control: a Control object
        discretization: a discretization of class Parameters
        parameters: a parameter object of class Parameters
        dprst_evap_hru: Evaporation from surface-depression storage for each
            HRU
        dprst_seep_hru: Seepage from surface-depression storage to associated
            GWR for each HRU
        hru_impervevap: HRU area-weighted average evaporation from impervious
            area for each HRU
        hru_intcpevap: HRU area-weighted average evaporation from the
            canopy for each HRU
        infil: Infiltration to the capillary and preferential-flow reservoirs,
            depth on pervious area
        infil_ag: Infiltration to the capillary reservoir for agricultural
            area, depth on HRU agricultural area
        sroff: Surface runoff to the stream network for each HRU
        sroff_vol: Surface runoff volume to the stream network for each HRU
        potet: Potential ET for each HRU
        transp_on: Flag indicating whether transpiration is occurring
            (0=no;1=yes)
        snow_evap: Evaporation and sublimation from snowpack on each HRU
        snowcov_area: Snow-covered area on each HRU prior to melt and
            sublimation unless snowpack
        ag_frac: Fraction of HRU that is agricultural/irrigated area
        aet_observed: Observed actual ET from CBH file for each HRU (when
            iter_aet_flag is True). Used to calculate AET_external.
        dprst_flag: use depression storage or not? None uses value in control
            file, which otherwise defaults to True.
        iter_aet_flag: Flag to enable iterative AET matching. If None,
            uses value from control.options["iter_aet_flag"] if available,
            otherwise defaults to False.
        imbalance_behavior: one of ["defer", None, "warn", "error"]
            with "defer" being the default and defering to
            control.options["imbalance_behavior"] when available. When
            control.options["imbalance_behavior"] is not avaiable,
            imbalance_behavior is set to "warn".
        calc_method: one of ["numpy", "numba"]. None defaults to "numpy".
            The "numba" option provides significant performance improvements,
            especially for the iterative AET matching loop. Parallelism is
            controlled via the NUMBA_NUM_THREADS environment variable.
        adjust_parameters: one of ["warn", "error", "no"]. Default is "warn",
            the code edits the parameters and issues a warning. If "error" is
            selected the the code issues warnings about all edited parameters
            before raising the error to give you information. If "no" is
            selected then no parameters are adjusted and there will be no
            warnings or errors.
        verbose: Print extra information or not?
        restart_read:
            May be boolean or a Pathlib.Path. If False, control.options
            will be examined for this key. If True, the working
            directory is searched for restart files. If a Pathlib.Path, this
            specifies an alternative directory to search for restart files.
        restart_write:
            As for restart_read but for writing. The directory in either
            case will be attempted to be created if it does not exist.
        restart_write_freq:
            If False, then control.options is examined for this key. The
            follwing values set the frequency of restart output with "y" for
            yearly, "m" for monthly, "d" for daily, or "f" for final.
    """

    def __init__(
        self,
        control: Control,
        discretization: Parameters,
        parameters: Parameters,
        dprst_evap_hru: adaptable,
        dprst_seep_hru: adaptable,
        hru_impervevap: adaptable,
        hru_intcpevap: adaptable,
        infil: adaptable,
        infil_ag: adaptable,
        sroff: adaptable,
        sroff_vol: adaptable,
        potet: adaptable,
        transp_on: adaptable,
        snow_evap: adaptable,
        snowcov_area: adaptable,
        ag_frac: adaptable,
        aet_observed: adaptable,
        dprst_flag: bool | None = None,
        iter_aet_flag: Literal[True, False, None] = None,
        imbalance_behavior: Literal["defer", None, "warn", "error"] = "defer",
        calc_method: Literal["numpy", None] = None,
        adjust_parameters: Literal["warn", "error", "no"] = "warn",
        verbose: bool | None = None,
        restart_read: pl.Path | bool = False,
        restart_write: pl.Path | bool = False,
        restart_write_freq: Literal["y", "m", "d", "f", False] = False,
    ):
        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
            restart_read=restart_read,
            restart_write=restart_write,
            restart_write_freq=restart_write_freq,
        )
        self.name = "PRMSSoilzoneAgObsET"

        self._set_inputs(locals())
        self._set_options(locals())

        # if iter_aet_flag not in control.options, _set_options  sets None, but
        # let's be more explicit.
        if self._iter_aet_flag is None:
            self._iter_aet_flag = False

        if self._dprst_flag is None:
            self._dprst_flag = True

        # Handle missing inputs when dprst_flag == False
        if not self._dprst_flag:
            raise ValueError("Not currently supporting non-dprst configs")

        # This uses options
        # if this uses inputs/dynamic parameters, they are not available until
        # advance, after model has solved inputs.

        self._set_budget()
        self._init_calc_method()

        return

    @staticmethod
    def get_dimensions() -> tuple:
        return ("nhru",)

    @staticmethod
    def get_parameters() -> tuple:
        return (
            "dprst_frac",
            "cov_type",
            "fastcoef_lin",
            "fastcoef_sq",
            "hru_area",
            "hru_in_to_cf",
            "hru_percent_imperv",
            "hru_type",
            "pref_flow_den",
            "pref_flow_infil_frac",
            "sat_threshold",
            "slowcoef_lin",
            "slowcoef_sq",
            "soil_moist_max",
            "soil_moist_init_frac",
            "soil_rechr_init_frac",
            "soil_rechr_max_frac",
            "soil_type",
            "soil2gw_max",
            "ssr2gw_exp",
            "ssr2gw_rate",
            "ssstor_init_frac",
            # Agricultural parameters
            "ag_soil_type",
            "ag_soil_moist_max",
            "ag_soil_moist_init_frac",
            "ag_soil_rechr_max_frac",
            "ag_soil_rechr_init_frac",
            "ag_cov_type",
            "ag_covden_sum",
            "ag_covden_win",
            "ag_soil2gw_max",
            "ag_soilwater_deficit_min",
            "max_soilzone_ag_iter",
            "soilzone_aet_converge",
        )

    @staticmethod
    def get_inputs() -> tuple:
        return (
            "dprst_evap_hru",
            "dprst_seep_hru",
            "hru_impervevap",
            "hru_intcpevap",
            "infil",
            "infil_ag",
            "sroff",
            "sroff_vol",
            "potet",
            "transp_on",
            "snow_evap",
            "snowcov_area",
            "ag_frac",
            "aet_observed",
        )

    @staticmethod
    def get_init_values() -> dict:
        return {
            # Mass budget input terms (whole HRU basis)
            "perv_infil_hru": zero,
            "ag_infil_hru": zero,
            "ag_irrigation_hru_source": zero,
            # Pervious area variables (whole HRU basis)
            "cap_infil_tot": zero,
            "cap_waterin": zero,
            "dunnian_flow": zero,
            "hru_actet": zero,
            "perv_actet": zero,
            "perv_actet_hru": zero,
            "potet_lower": zero,
            "potet_rechr": zero,
            "pref_flow": zero,
            "pref_flow_in": zero,
            "pref_flow_infil": zero,
            "pref_flow_max": zero,
            "pref_flow_stor": zero,
            "pref_flow_stor_change": zero,
            "pref_flow_stor_prev": nan,
            "pref_flow_thrsh": zero,
            "recharge": zero,
            "slow_flow": zero,
            "slow_stor": zero,
            "slow_stor_change": zero,
            "slow_stor_prev": nan,
            "soil_lower": zero,
            "soil_lower_change": zero,
            "soil_lower_change_hru": zero,
            "soil_lower_prev": zero,
            "soil_lower_ratio": zero,
            "soil_lower_max": nan,
            "soil_moist": nan,
            "soil_moist_tot": nan,
            "soil_rechr": zero,
            "soil_rechr_change": zero,
            "soil_rechr_change_hru": zero,
            "soil_rechr_prev": zero,
            "soil_saturated": zero,
            "soil_to_gw": zero,
            "soil_to_ssr": zero,
            "soil_zone_max": nan,
            "ssr_to_gw": zero,
            "ssres_flow": zero,
            "ssres_flow_vol": nan,
            "ssres_in": zero,
            "ssres_stor": nan,
            "swale_actet": zero,
            "unused_potet": zero,
            "perv_soil_to_gw": zero,
            "perv_soil_to_gvr": zero,
            # Agricultural area variables (whole HRU basis)
            "ag_cap_infil_tot": zero,
            "ag_soil_moist": zero,
            "ag_soil_moist_prev": zero,
            "ag_soil_moist_change": zero,
            "ag_soil_moist_change_hru": zero,
            "ag_soil_rechr": zero,
            "ag_soil_rechr_prev": zero,
            "ag_soil_rechr_change": zero,
            "ag_soil_rechr_change_hru": zero,
            "ag_soil_rechr_max": nan,
            "ag_soil_lower": nan,
            "ag_soil_lower_change": zero,
            "ag_soil_lower_change_hru": zero,
            "ag_soil_lower_stor_max": nan,
            "ag_actet": zero,
            "hru_ag_actet": zero,
            "ag_potet_rechr": zero,
            "ag_potet_lower": zero,
            "ag_soil_to_gw": zero,
            "ag_soil_to_gvr": zero,
            "ag_hortonian": zero,
            "ag_soil_saturated": zero,
            "unused_ag_et": zero,
            "ag_soilwater_deficit": zero,
            "ag_irrigation_add": zero,
            "ag_irrigation_add_vol": zero,
            "ag_aet_external_vol": zero,
            "AET_external": zero,
            # Redistribution tracking (for ag_frac changes)
            "ag_soil_moist_redistribution": zero,
            "ag_soil_rechr_redistribution": zero,
            "soil_rechr_redistribution": zero,
            "soil_lower_redistribution": zero,
            "slow_stor_redistribution": zero,
        }

    @staticmethod
    def get_restart_variables() -> list:
        return [
            "soil_moist",
            "soil_rechr",
            "slow_stor",
            "ag_soil_moist",
            "ag_soil_rechr",
            "soil_lower",
        ]

    @staticmethod
    def get_mass_budget_terms() -> dict:
        return {
            "inputs": [
                "perv_infil_hru",
                "ag_infil_hru",
                "ag_irrigation_hru_source",
            ],
            "outputs": [
                "perv_actet_hru",
                "hru_ag_actet",
                "perv_soil_to_gw",
                "ag_soil_to_gw",
                "ssr_to_gw",
                "slow_flow",
                "dunnian_flow",
                "pref_flow",
            ],
            "storage_changes": [
                "soil_rechr_change_hru",
                "soil_lower_change_hru",
                "slow_stor_change",
                "pref_flow_stor_change",
                "ag_soil_rechr_change_hru",
                "ag_soil_lower_change_hru",
            ],
        }

    def _set_initial_conditions(self):
        # Called in super before options are set
        pass

    def _init_diagnostic_vars(self) -> None:
        return

    def _initialize_soilzone_ag_data(self):
        """Initialize soilzone_ag data structures.

        Fortran reference: szinit_ag() and initialization in szdecl_ag()
        """

        # self.ag_cap_infil_tot = self.hru_percent_imperv * np.nan

        # Derived parameters for impervious and pervious areas
        self.hru_area_imperv = self.hru_percent_imperv * self.hru_area
        self.hru_area_perv = self.hru_area - self.hru_area_imperv

        # Agricultural area
        self.ag_area = self.ag_frac * self.hru_area

        # Adjust pervious area to exclude agricultural area
        wh_active = np.where(self.hru_type != HruType.INACTIVE.value)
        if self._dprst_flag:
            dprst_area_max = self.dprst_frac * self.hru_area
            self.hru_area_perv[wh_active] = (
                self.hru_area_perv[wh_active]
                - dprst_area_max[wh_active]
                - self.ag_area[wh_active]
            )
        else:
            self.hru_area_perv[wh_active] = (
                self.hru_area_perv[wh_active] - self.ag_area[wh_active]
            )

        # Recompute pervious fraction
        self.hru_frac_perv = np.zeros(self.nhru)
        wh_active = np.where(self.hru_type != HruType.INACTIVE.value)
        self.hru_frac_perv[wh_active] = (
            self.hru_area_perv[wh_active] / self.hru_area[wh_active]
        )

        # Pervious soil parameters
        self.soil_rechr_max = self.soil_rechr_max_frac * self.soil_moist_max

        # Agricultural soil parameters
        self.ag_soil_rechr_max = (
            self.ag_soil_rechr_max_frac * self.ag_soil_moist_max
        )

        # self._snow_free = one - self.snowcov_area

        # Edit parameters for inactive/lake HRUs
        wh_inactive_or_lake = np.where(
            (self.hru_type == HruType.INACTIVE.value)
            | (self.hru_type == HruType.LAKE.value)
        )
        self._sat_threshold = self.sat_threshold.copy()
        self._sat_threshold[wh_inactive_or_lake] = zero

        wh_not_land = np.where(self.hru_type != HruType.LAND.value)
        self._pref_flow_den = self.pref_flow_den.copy()
        self._pref_flow_den[wh_not_land] = zero

        # Validate pref_flow_infil_frac
        if (self.pref_flow_infil_frac.min() < zero).any() or (
            self.pref_flow_infil_frac.max() > one
        ).any():
            msg = (
                "Values of pref_flow_infil_frac outside of [0,1]. "
                "If values are all -1, you must set pref_flow_infil_frac to "
                "pref_flow_den in the parameter file yourself."
            )
            raise ValueError(msg)

        # Initialize soil moisture states
        if not self._restart_read:
            # Pervious area
            self.soil_moist[:] = (
                self.soil_moist_init_frac * self.soil_moist_max
            )
            self.soil_rechr[:] = (
                self.soil_rechr_init_frac * self.soil_rechr_max
            )

            # Agricultural area
            self.ag_soil_moist[:] = (
                self.ag_soil_moist_init_frac * self.ag_soil_moist_max
            )
            self.ag_soil_rechr[:] = (
                self.ag_soil_rechr_init_frac * self.ag_soil_rechr_max
            )

        # Initialize subsurface reservoir storage
        if not self._restart_read:
            self.ssres_stor = self.ssstor_init_frac * self._sat_threshold
            wh_inactive_or_lake = np.where(
                (self.hru_type == HruType.INACTIVE.value)
                | (self.hru_type == HruType.LAKE.value)
            )
            self.ssres_stor[wh_inactive_or_lake] = zero
            del self.ssstor_init_frac

        # Ensure LAKE and INACTIVE HRUs have zero agricultural area
        wh_no_ag = np.where(
            (self.hru_type == HruType.LAKE.value)
            | (self.hru_type == HruType.INACTIVE.value)
        )
        for vv in ["ag_frac", "ag_area"]:
            # GSFLOW sets the values on the mask to zero, but we will NOT
            # edit parameters. So this could be a point of divergence in
            # solution, though it will raise an error here.
            assert (self[vv][wh_no_ag] == zero).all()

        self.ag_soil_moist[wh_no_ag] = zero
        self.ag_soil_rechr[wh_no_ag] = zero

        # Calculate agricultural soil lower storage max
        for ihru in range(self.nhru):
            self.ag_soil_lower_stor_max[ihru] = (
                self.ag_soil_moist_max[ihru] - self.ag_soil_rechr_max[ihru]
            )

        # Parameter validation and adjustment
        throw_error = False
        mask = self.soil_moist_max < 1.0e-5
        if mask.any() and self._adjust_parameters != "no":
            if self._adjust_parameters == "error":
                throw_error = True
            msg = (
                "soil_moist_max < 1.0e-5, set to 1.0e-5 at indices: "
                f"{np.where(mask)[0]}"
            )
            warn(msg, UserWarning)
            self.soil_moist_max[mask] = 1.0e-5

        mask = self.ag_soil_moist_max < 1.0e-5
        if mask.any() and self._adjust_parameters != "no":
            if self._adjust_parameters == "error":
                throw_error = True
            msg = (
                "ag_soil_moist_max < 1.0e-5, set to 1.0e-5 at indices: "
                f"{np.where(mask)[0]}"
            )
            warn(msg, UserWarning)
            self.ag_soil_moist_max[mask] = 1.0e-5

        if throw_error:
            raise ValueError("Parameter adjustment errors, see warnings above")

        # Additional parameter validation similar to base soilzone
        mask = self.soil_rechr_max < 1.0e-5
        if mask.any() and self._adjust_parameters != "no":
            if self._adjust_parameters == "error":
                throw_error = True
            msg = (
                "soil_rechr_max < 1.0e-5, set to 1.0e-5 at indices: "
                f"{np.where(mask)[0]}"
            )
            warn(msg, UserWarning)
            self.soil_rechr_max[mask] = 1.0e-5

        mask = self.ag_soil_rechr_max < 1.0e-5
        if mask.any() and self._adjust_parameters != "no":
            if self._adjust_parameters == "error":
                throw_error = True
            msg = (
                "ag_soil_rechr_max < 1.0e-5, set to 1.0e-5 at indices: "
                f"{np.where(mask)[0]}"
            )
            warn(msg, UserWarning)
            self.ag_soil_rechr_max[mask] = 1.0e-5

        mask = self.soil_rechr_max > self.soil_moist_max
        if mask.any() and self._adjust_parameters != "no":
            if self._adjust_parameters == "error":
                throw_error = True
            msg = (
                "soil_rechr_max > soil_moist_max, "
                "soil_rechr_max set to soil_moist_max at indices: "
                f"{np.where(mask)[0]}"
            )
            warn(msg, UserWarning)
            self.soil_rechr_max = np.where(
                mask, self.soil_moist_max, self.soil_rechr_max
            )

        mask = self.ag_soil_rechr_max > self.ag_soil_moist_max
        if mask.any() and self._adjust_parameters != "no":
            if self._adjust_parameters == "error":
                throw_error = True
            msg = (
                "ag_soil_rechr_max > ag_soil_moist_max, "
                "ag_soil_rechr_max set to ag_soil_moist_max at indices: "
                f"{np.where(mask)[0]}"
            )
            warn(msg, UserWarning)
            self.ag_soil_rechr_max = np.where(
                mask, self.ag_soil_moist_max, self.ag_soil_rechr_max
            )

        mask = self.soil_rechr > self.soil_rechr_max
        if mask.any() and self._adjust_parameters != "no":
            if self._adjust_parameters == "error":
                throw_error = True
            msg = (
                "soil_rechr_init > soil_rechr_max, "
                "setting soil_rechr_init to soil_rechr_max at indices: "
                f"{np.where(mask)[0]}"
            )
            warn(msg, UserWarning)
            self.soil_rechr = np.where(
                mask, self.soil_rechr_max, self.soil_rechr
            )

        mask = self.ag_soil_rechr > self.ag_soil_rechr_max
        if mask.any() and self._adjust_parameters != "no":
            if self._adjust_parameters == "error":
                throw_error = True
            msg = (
                "ag_soil_rechr_init > ag_soil_rechr_max, "
                "setting ag_soil_rechr_init to ag_soil_rechr_max at indices: "
                f"{np.where(mask)[0]}"
            )
            warn(msg, UserWarning)
            self.ag_soil_rechr = np.where(
                mask, self.ag_soil_rechr_max, self.ag_soil_rechr
            )

        mask = self.soil_moist > self.soil_moist_max
        if mask.any() and self._adjust_parameters != "no":
            if self._adjust_parameters == "error":
                throw_error = True
            msg = (
                "soil_moist_init > soil_moist_max, "
                "setting soil_moist to soil_moist max at indices: "
                f"{np.where(mask)[0]}"
            )
            warn(msg, UserWarning)
            self.soil_moist = np.where(
                mask, self.soil_moist_max, self.soil_moist
            )

        mask = self.ag_soil_moist > self.ag_soil_moist_max
        if mask.any() and self._adjust_parameters != "no":
            if self._adjust_parameters == "error":
                throw_error = True
            msg = (
                "ag_soil_moist_init > ag_soil_moist_max, "
                "setting ag_soil_moist to ag_soil_moist max at indices: "
                f"{np.where(mask)[0]}"
            )
            warn(msg, UserWarning)
            self.ag_soil_moist = np.where(
                mask, self.ag_soil_moist_max, self.ag_soil_moist
            )

        mask = self.soil_rechr > self.soil_moist
        if mask.any() and self._adjust_parameters != "no":
            if self._adjust_parameters == "error":
                throw_error = True
            msg = (
                "soil_rechr > soil_moist, "
                "setting soil_rechr to soil_moist at indices: "
                f"{np.where(mask)[0]}"
            )
            warn(msg, UserWarning)
            self.soil_rechr = np.where(mask, self.soil_moist, self.soil_rechr)

        mask = self.ag_soil_rechr > self.ag_soil_moist
        if mask.any() and self._adjust_parameters != "no":
            if self._adjust_parameters == "error":
                throw_error = True
            msg = (
                "ag_soil_rechr > ag_soil_moist, "
                "setting ag_soil_rechr to ag_soil_moist at indices: "
                f"{np.where(mask)[0]}"
            )
            warn(msg, UserWarning)
            self.ag_soil_rechr = np.where(
                mask, self.ag_soil_moist, self.ag_soil_rechr
            )

        mask = self.ssres_stor > self._sat_threshold
        if mask.any() and self._adjust_parameters != "no":
            if self._adjust_parameters == "error":
                throw_error = True
            msg = (
                "ssres_stor > _sat_threshold, "
                "setting ssres_stor to _sat_threshold at indices: "
                f"{np.where(mask)[0]}"
            )
            warn(msg, UserWarning)
            self.ssres_stor = np.where(
                mask, self._sat_threshold, self.ssres_stor
            )

        if throw_error:
            raise ValueError(
                "Some parameter values were edited and an error was requested."
                " See warnings for additional details."
            )

        # Set up preferential flow thresholds and max
        wh_swale = np.where(self.hru_type == HruType.SWALE.value)
        self.pref_flow_thrsh[wh_swale] = self._sat_threshold[wh_swale]

        wh_land = np.where(self.hru_type == HruType.LAND.value)
        self.pref_flow_thrsh[wh_land] = self._sat_threshold[wh_land] * (
            one - self._pref_flow_den[wh_land]
        )
        self.pref_flow_max[wh_land] = (
            self._sat_threshold[wh_land] - self.pref_flow_thrsh[wh_land]
        )

        # Initialize slow_stor and pref_flow_stor from ssres_stor
        if not self._restart_read:
            wh_land_or_swale = np.where(
                (self.hru_type == HruType.LAND.value)
                | (self.hru_type == HruType.SWALE.value)
            )
            self.slow_stor[wh_land_or_swale] = np.minimum(
                self.ssres_stor[wh_land_or_swale],
                self.pref_flow_thrsh[wh_land_or_swale],
            )
            self.pref_flow_stor[wh_land_or_swale] = (
                self.ssres_stor[wh_land_or_swale]
                - self.slow_stor[wh_land_or_swale]
            )

        # Set flags for preferential flow and soil to gw
        self._pref_flow_flag = (self._pref_flow_den > zero).any()
        self._soil2gw_flag = (self.soil2gw_max > zero).any()

        # Calculate derived soil zone variables
        self.soil_zone_max = (
            self._sat_threshold
            + self.soil_moist_max * self.hru_frac_perv
            + self.ag_soil_moist_max * self.ag_frac
        )
        self.soil_moist_tot = (
            self.ssres_stor
            + self.soil_moist * self.hru_frac_perv
            + self.ag_soil_moist * self.ag_frac
        )

        if not self._restart_read:
            self.soil_lower = self.soil_moist - self.soil_rechr
        self.soil_lower_max = self.soil_moist_max - self.soil_rechr_max

        wh_soil_lower_stor = np.where(self.soil_lower_max > zero)
        self.soil_lower_ratio[wh_soil_lower_stor] = (
            self.soil_lower[wh_soil_lower_stor]
            / self.soil_lower_max[wh_soil_lower_stor]
        )

        # Iteration counter for non-convergence tracking
        self._iter_nonconverge = 0

        # Initialize _prev variables for timestep 0 (when not reading restart)
        if not self._restart_read:
            self.soil_rechr_prev[:] = self.soil_rechr
            self.soil_lower_prev[:] = self.soil_lower
            self.ag_soil_moist_prev[:] = self.ag_soil_moist
            self.ag_soil_rechr_prev[:] = self.ag_soil_rechr
            self.pref_flow_stor_prev[:] = self.pref_flow_stor
            self.slow_stor_prev[:] = self.slow_stor

        return

    def _init_calc_method(self):
        """Initialize calculation method."""
        if self._calc_method is None:
            self._calc_method = "numpy"

        avail_methods = ["numpy", "numba"]

        if self._calc_method.lower() not in avail_methods:
            msg = (
                f"Invalid calc_method={self._calc_method} for {self.name}. "
                f"Setting calc_method to 'numpy' for {self.name}"
            )
            warn(msg)
            self._calc_method = "numpy"

        if self._calc_method.lower() == "numba":
            import numba as nb

            numba_msg = f"{self.name} jit compiling with numba "
            nb_parallel = (numba_num_threads is not None) and (
                numba_num_threads > 1
            )
            if nb_parallel:
                numba_msg += f"and using {numba_num_threads} threads"
            else:
                numba_msg += "in sequential mode"
            print(numba_msg, flush=True)

            # JIT compile the module-level function with appropriate parallel
            # setting

            # Clear any existing compilation
            if hasattr(_calculate_soilzone_ag_numba, "_cache"):
                _calculate_soilzone_ag_numba._cache.clear()

            # Compile with the right parallel setting
            self._calculate_soilzone_ag_numba_compiled = nb.njit(
                parallel=nb_parallel
            )(_calculate_soilzone_ag_numba)

            self._calculate_soilzone_ag = self._calculate_numba

        else:
            self._calculate_soilzone_ag = self._calculate_numpy

        return

    def _advance_variables(self) -> None:
        """Save previous timestep values.

        Fortran reference: variables saved at start of szrun_ag()
        """
        self.pref_flow_stor_prev[:] = self.pref_flow_stor
        self.soil_rechr_prev[:] = self.soil_rechr
        self.soil_lower_prev[:] = self.soil_lower
        self.slow_stor_prev[:] = self.slow_stor
        # Agricultural soil moisture previous values
        self.ag_soil_moist_prev[:] = self.ag_soil_moist
        self.ag_soil_rechr_prev[:] = self.ag_soil_rechr
        return

    def _update_ag_areas(self):
        """Update ag_area and related quantities when ag_frac changes.

        This method recalculates derived areas that depend on ag_frac,
        which may change dynamically via an AdapterDynamicParameter.

        When ag_frac changes, the behavior follows Fortran logic in
        dynamic_soil_param_read.f90 lines 503-528:
        - If ag_frac DECREASES: keep ag_soil_moist unchanged, send excess
          water volume to slow_stor (GVR storage)
        - If ag_frac INCREASES: scale down ag_soil_moist to conserve depth
          (ag_soil_moist = ag_soil_moist * old_frac / new_frac)
        - If ag_frac goes to ZERO: transfer water to slow_stor

        Fortran reference: dynamic_soil_param_read.f90
        """
        # Store old values for scaling calculations
        old_ag_frac = (
            self.ag_area / self.hru_area
        )  # old ag_frac from old ag_area
        old_hru_area_perv = self.hru_area_perv.copy()

        # Recalculate agricultural area
        new_ag_area = self.ag_frac * self.hru_area

        # Recalculate pervious area to exclude agricultural area
        new_hru_area_perv = self.hru_area - self.hru_area_imperv
        wh_active = np.where(self.hru_type != HruType.INACTIVE.value)
        if self._dprst_flag:
            dprst_area_max = self.dprst_frac * self.hru_area
            new_hru_area_perv[wh_active] = (
                new_hru_area_perv[wh_active]
                - dprst_area_max[wh_active]
                - new_ag_area[wh_active]
            )
        else:
            new_hru_area_perv[wh_active] = (
                new_hru_area_perv[wh_active] - new_ag_area[wh_active]
            )

        # Track water to transfer to slow_stor
        to_slow_stor = np.zeros(self.nhru)

        # Handle ag_frac changes following Fortran logic
        # Use np.isclose to avoid false positives from floating-point noise
        for ihru in range(self.nhru):
            if np.isclose(self.ag_frac[ihru], old_ag_frac[ihru]):
                # No change
                continue

            if self.ag_soil_moist[ihru] > 0.0:
                if self.ag_frac[ihru] > 0.0:
                    if self.ag_frac[ihru] < old_ag_frac[ihru]:
                        # ag_frac DECREASED: keep ag_soil_moist unchanged,
                        # send excess water volume to slow_stor
                        # Fortran: to_slow_stor =
                        #     Ag_soil_moist(i) * (Ag_frac(i) - frac_ag)
                        excess_volume = self.ag_soil_moist[ihru] * (
                            old_ag_frac[ihru] - self.ag_frac[ihru]
                        )
                        to_slow_stor[ihru] = excess_volume
                        # ag_soil_moist and ag_soil_rechr remain unchanged
                    elif old_ag_frac[ihru] < self.ag_frac[ihru]:
                        # ag_frac INCREASED: scale down ag_soil_moist
                        # Fortran: Ag_soil_moist(i) =
                        #     Ag_soil_moist(i)*Ag_frac(i)/frac_ag
                        scale = old_ag_frac[ihru] / self.ag_frac[ihru]
                        self.ag_soil_moist[ihru] = (
                            self.ag_soil_moist[ihru] * scale
                        )
                        self.ag_soil_rechr[ihru] = (
                            self.ag_soil_rechr[ihru] * scale
                        )
                else:
                    # ag_frac went to ZERO: transfer all water to slow_stor
                    # Fortran: tmp = Ag_soil_moist(i)*Ag_frac(i)
                    to_slow_stor[ihru] = (
                        self.ag_soil_moist[ihru] * old_ag_frac[ihru]
                    )
                    self.ag_soil_moist[ihru] = 0.0
                    self.ag_soil_rechr[ihru] = 0.0

        # Add excess water to slow_stor (GVR storage)
        # This matches Fortran behavior of sending excess to gravity reservoir
        self.slow_stor[:] = self.slow_stor + to_slow_stor

        # Scale pervious soil moisture when pervious area changes
        # Fortran: Soil_moist(i) = Soil_moist(i)*tmp where
        # tmp = Hru_perv(i)/hruperv
        # Use np.isclose to avoid false positives from floating-point noise
        wh_perv_changed = np.where(
            (~np.isclose(old_hru_area_perv, new_hru_area_perv))
            & (new_hru_area_perv > 0.0)
        )
        if len(wh_perv_changed[0]) > 0:
            scale_perv = (
                old_hru_area_perv[wh_perv_changed]
                / new_hru_area_perv[wh_perv_changed]
            )
            self.soil_moist[wh_perv_changed] = (
                self.soil_moist[wh_perv_changed] * scale_perv
            )
            self.soil_rechr[wh_perv_changed] = (
                self.soil_rechr[wh_perv_changed] * scale_perv
            )
            # Note: soil_lower is NOT updated here - it will be recomputed
            # as soil_moist - soil_rechr in the main HRU loop

        # Handle case where pervious area goes to zero
        wh_perv_to_zero = np.where(
            (old_hru_area_perv > 0.0) & np.isclose(new_hru_area_perv, 0.0)
        )
        if len(wh_perv_to_zero[0]) > 0:
            self.soil_moist[wh_perv_to_zero] = 0.0
            self.soil_rechr[wh_perv_to_zero] = 0.0
            # Note: soil_lower will be recomputed in the main HRU loop

        # Update the stored area values
        self.ag_area[:] = new_ag_area
        self.hru_area_perv[:] = new_hru_area_perv

        # Recompute pervious fraction
        self.hru_frac_perv[wh_active] = (
            self.hru_area_perv[wh_active] / self.hru_area[wh_active]
        )

        # Recalculate soil_zone_max since it depends on hru_frac_perv and
        # ag_frac
        self.soil_zone_max[:] = (
            self._sat_threshold
            + self.soil_moist_max * self.hru_frac_perv
            + self.ag_soil_moist_max * self.ag_frac
        )

        return

    def _calculate(self, simulation_time):
        """Main calculation routine with iterative AET matching.

        Fortran reference: szrun_ag()

        This method implements the iterative loop that adjusts irrigation
        to match observed AET when iter_aet_flag is True.
        """
        if self.control._itime_step == 0:
            self._initialize_soilzone_ag_data()

        # Update ag_area and related quantities in case ag_frac changed
        self._update_ag_areas()

        # After ag_frac changes, recompute soil_lower from the scaled values
        # (Note: _update_ag_areas scales soil_moist and soil_rechr but doesn't
        # update soil_lower, so we need to do it here)
        self.soil_lower[:] = self.soil_moist - self.soil_rechr

        # Calculate how much water was redistributed due to ag_frac changes.
        #
        # MASS BUDGET APPROACH:
        # When ag_frac changes (e.g., on Jan 1 annually), water is
        # redistributed:
        # - Agricultural soil water may be transferred to slow_stor or scaled
        # - Pervious soil water is scaled to conserve volume over new area
        # These redistributions are area-accounting adjustments, not hydrologic
        # processes, so they must be excluded from mass budget calculations.
        #
        # To maintain correct mass balance while preserving the semantic
        # meaning of "_prev" variables (= truly previous timestep values), we:
        # 1. Keep _prev values as they were at end of previous timestep
        # 2. Track how much was redistributed in these _redistribution
        #    variables
        # 3. Calculate _change = current - prev - redistribution
        #
        # This gives us:
        # - Correct _prev semantics (match Fortran postprocessed _prev values)
        # - Correct mass budget (_change excludes area redistributions)
        #
        # NOTE: The _change variables calculated this way will NOT match simple
        # Fortran postprocessing (current - previous) because Fortran includes
        # the redistributions. This is intentional - the mass budget validates
        # these variables, not external comparison.
        self.ag_soil_moist_redistribution[:] = (
            self.ag_soil_moist - self.ag_soil_moist_prev
        )
        self.ag_soil_rechr_redistribution[:] = (
            self.ag_soil_rechr - self.ag_soil_rechr_prev
        )
        self.soil_rechr_redistribution[:] = (
            self.soil_rechr - self.soil_rechr_prev
        )
        self.soil_lower_redistribution[:] = (
            self.soil_lower - self.soil_lower_prev
        )
        self.slow_stor_redistribution[:] = self.slow_stor - self.slow_stor_prev

        # Store initial values for iteration (Fortran: It0 variables)
        it0_soil_moist = self.soil_moist.copy()
        it0_soil_rechr = self.soil_rechr.copy()
        it0_ssres_stor = self.ssres_stor.copy()
        it0_slow_stor = self.slow_stor.copy()
        it0_ag_soil_moist = self.ag_soil_moist.copy()
        it0_ag_soil_rechr = self.ag_soil_rechr.copy()
        if self._pref_flow_flag:
            it0_pref_flow_stor = self.pref_flow_stor.copy()

        # Initialize iteration variables
        keep_iterating = True
        soil_iter = 1

        # Initialize ObsET variables only when iter_aet_flag is True
        if self._iter_aet_flag:
            self.ag_irrigation_add[:] = zero
            self.ag_irrigation_add_vol[:] = zero
            self.ag_aet_external_vol[:] = zero

            # Compute and validate AET_external from observed values
            # (per climate_hru_debug.f90 lines 113-127)
            # Copy observed values to working variables
            self.AET_external[:] = self.aet_observed

            # Where AET < 0 (but not -1.0 missing value) for ag HRUs, set to 0
            mask = (
                (self.AET_external < 0.0)
                & (self.AET_external != -1.0)
                & (self.ag_frac > 0.0)
            )
            if np.any(mask) and self._verbose:
                warn(
                    f"AET_external < 0.0 for {mask.sum()} HRUs; setting to 0.0"
                )
            self.AET_external[mask] = 0.0

            # Use actual ObsET arrays
            aet_external_arr = self.AET_external
            ag_irrigation_add_arr = self.ag_irrigation_add
        else:
            # Create dummy zero arrays for non-iterative case
            aet_external_arr = np.zeros(self.nhru, dtype=float)
            ag_irrigation_add_arr = np.zeros(self.nhru, dtype=float)

        # Fortran: DO WHILE ( keep_iterating==ACTIVE )
        while keep_iterating:
            # Restore initial conditions for iteration
            if soil_iter > 1:
                self.soil_moist[:] = it0_soil_moist
                self.soil_rechr[:] = it0_soil_rechr
                self.ssres_stor[:] = it0_ssres_stor
                self.slow_stor[:] = it0_slow_stor
                self.ag_soil_moist[:] = it0_ag_soil_moist
                self.ag_soil_rechr[:] = it0_ag_soil_rechr
                if self._pref_flow_flag:
                    self.pref_flow_stor[:] = it0_pref_flow_stor

            result = self._calculate_soilzone_ag(
                soil_iter=soil_iter,
                iter_aet_flag=self._iter_aet_flag,
                pref_flow_flag=self._pref_flow_flag,
                snow_free=1.0 - self.snowcov_area,
                compute_soilmoist=self._compute_soilmoist,
                compute_szactet=self._compute_szactet,
                compute_interflow=self._compute_interflow,
                compute_gwflow=self._compute_gwflow,
                # Parameters
                cov_type=self.cov_type,
                ag_cov_type=self.ag_cov_type,
                fastcoef_lin=self.fastcoef_lin,
                fastcoef_sq=self.fastcoef_sq,
                hru_frac_perv=self.hru_frac_perv,
                hru_type=self.hru_type,
                ag_frac=self.ag_frac,
                ag_area=self.ag_area,
                hru_area=self.hru_area,
                hru_area_perv=self.hru_area_perv,
                pref_flow_den=self._pref_flow_den,
                pref_flow_infil_frac=self.pref_flow_infil_frac,
                sat_threshold=self._sat_threshold,
                slowcoef_lin=self.slowcoef_lin,
                slowcoef_sq=self.slowcoef_sq,
                soil_moist_max=self.soil_moist_max,
                soil_rechr_max=self.soil_rechr_max,
                soil_type=self.soil_type,
                ag_soil_type=self.ag_soil_type,
                ag_soil_moist_max=self.ag_soil_moist_max,
                ag_soil_rechr_max=self.ag_soil_rechr_max,
                soil2gw_max=self.soil2gw_max,
                ag_soil2gw_max=self.ag_soil2gw_max,
                ssr2gw_exp=self.ssr2gw_exp,
                ssr2gw_rate=self.ssr2gw_rate,
                ag_soilwater_deficit_min=self.ag_soilwater_deficit_min,
                soilzone_aet_converge=self.soilzone_aet_converge,
                # Inputs
                dprst_evap_hru=self.dprst_evap_hru,
                dprst_seep_hru=self.dprst_seep_hru,
                hru_impervevap=self.hru_impervevap,
                hru_intcpevap=self.hru_intcpevap,
                infil=self.infil,
                infil_ag=self.infil_ag,
                potet=self.potet,
                transp_on=self.transp_on,
                snow_evap=self.snow_evap,
                snowcov_area=self.snowcov_area,
                aet_external=aet_external_arr,
                ag_irrigation_add=ag_irrigation_add_arr,
                # State variables
                soil_moist=self.soil_moist,
                soil_rechr=self.soil_rechr,
                ag_soil_moist=self.ag_soil_moist,
                ag_soil_rechr=self.ag_soil_rechr,
                pref_flow_stor=self.pref_flow_stor,
                slow_stor=self.slow_stor,
                ssres_stor=self.ssres_stor,
                # Previous timestep values for storage changes
                pref_flow_stor_prev=self.pref_flow_stor_prev,
                soil_rechr_prev=self.soil_rechr_prev,
                soil_lower_prev=self.soil_lower_prev,
                slow_stor_prev=self.slow_stor_prev,
                ag_soil_moist_prev=self.ag_soil_moist_prev,
                ag_soil_rechr_prev=self.ag_soil_rechr_prev,
                # Output variables
                soil_to_gw=self.soil_to_gw,
                soil_to_ssr=self.soil_to_ssr,
                perv_soil_to_gw=self.perv_soil_to_gw,
                perv_soil_to_gvr=self.perv_soil_to_gvr,
                ag_soil_to_gw=self.ag_soil_to_gw,
                ag_soil_to_gvr=self.ag_soil_to_gvr,
                ssr_to_gw=self.ssr_to_gw,
                slow_flow=self.slow_flow,
                ssres_flow=self.ssres_flow,
                potet_rechr=self.potet_rechr,
                potet_lower=self.potet_lower,
                ag_potet_rechr=self.ag_potet_rechr,
                ag_potet_lower=self.ag_potet_lower,
                cap_waterin=self.cap_waterin,
                cap_infil_tot=self.cap_infil_tot,
                ag_cap_infil_tot=self.ag_cap_infil_tot,
                # ag_water_in=self.ag_water_in,
                pref_flow_in=self.pref_flow_in,
                pref_flow_infil=self.pref_flow_infil,
                pref_flow=self.pref_flow,
                perv_actet=self.perv_actet,
                perv_actet_hru=self.perv_actet_hru,
                ag_actet=self.ag_actet,
                hru_ag_actet=self.hru_ag_actet,
                hru_actet=self.hru_actet,
                soil_lower=self.soil_lower,
                ag_soil_lower=self.ag_soil_lower,
                dunnian_flow=self.dunnian_flow,
                pref_flow_max=self.pref_flow_max,
                pref_flow_thrsh=self.pref_flow_thrsh,
                soil_lower_max=self.soil_lower_max,
                ag_soil_lower_stor_max=self.ag_soil_lower_stor_max,
                soil_zone_max=self.soil_zone_max,
                soil_moist_tot=self.soil_moist_tot,
                ssres_in=self.ssres_in,
                recharge=self.recharge,
                unused_potet=self.unused_potet,
                unused_ag_et=self.unused_ag_et,
                ag_hortonian=self.ag_hortonian,
                ag_soil_saturated=self.ag_soil_saturated,
                ag_soilwater_deficit=self.ag_soilwater_deficit,
                swale_actet=self.swale_actet,
                soil_saturated=self.soil_saturated,
                sroff=self.sroff,
                soil_lower_ratio=self.soil_lower_ratio,
                pref_flow_stor_change=self.pref_flow_stor_change,
                soil_lower_change=self.soil_lower_change,
                soil_lower_change_hru=self.soil_lower_change_hru,
                soil_rechr_change=self.soil_rechr_change,
                soil_rechr_change_hru=self.soil_rechr_change_hru,
                slow_stor_change=self.slow_stor_change,
                ag_soil_moist_change=self.ag_soil_moist_change,
                ag_soil_moist_change_hru=self.ag_soil_moist_change_hru,
                ag_soil_rechr_change=self.ag_soil_rechr_change,
                ag_soil_rechr_change_hru=self.ag_soil_rechr_change_hru,
                ag_soil_lower_change=self.ag_soil_lower_change,
                ag_soil_lower_change_hru=self.ag_soil_lower_change_hru,
                ag_soil_moist_redistribution=self.ag_soil_moist_redistribution,
                ag_soil_rechr_redistribution=self.ag_soil_rechr_redistribution,
                soil_rechr_redistribution=self.soil_rechr_redistribution,
                soil_lower_redistribution=self.soil_lower_redistribution,
                slow_stor_redistribution=self.slow_stor_redistribution,
            )

            # Unpack results
            (
                add_estimated_irrigation,
                num_hrus_ag_iter,
                unsatisfied_big,
            ) = result

            # Update keep_iterating based on result
            if not add_estimated_irrigation:
                keep_iterating = False

            # Check convergence
            soil_iter += 1
            if not self._iter_aet_flag or not self.transp_on.any():
                keep_iterating = False
            if soil_iter > self.max_soilzone_ag_iter:
                keep_iterating = False

        # Store final iteration count
        soil_iter -= 1

        # Report convergence status
        if self._iter_aet_flag and self._verbose:
            if soil_iter == self.max_soilzone_ag_iter:
                self._iter_nonconverge += 1
                msg = (
                    f"WARNING: ag AET did not converge at "
                    f"{simulation_time.isoformat()}, "
                    f"largest unsatisfied AET: {unsatisfied_big:.4f}, "
                    f"iterations: {soil_iter}, "
                    f"non-convergence count: {self._iter_nonconverge}"
                )
                warn(msg, UserWarning)

        # Calculate volume-based outputs
        self.sroff_vol[:] = self.sroff * self.hru_in_to_cf
        self.ssres_flow_vol[:] = self.ssres_flow * self.hru_in_to_cf

        # Calculate ObsET volume outputs only when iter_aet_flag is True
        if self._iter_aet_flag:
            self.ag_irrigation_add_vol[:] = (
                self.ag_irrigation_add * self.ag_area
            )
            self.ag_aet_external_vol[:] = self.ag_actet * self.ag_area

        # Calculate mass budget input terms (all on HRU basis)
        self.perv_infil_hru[:] = self.infil * self.hru_frac_perv
        self.ag_infil_hru[:] = self.infil_ag * self.ag_frac
        # Only calculate ag_irrigation_hru_source when iter_aet_flag is set
        # (PRMSSoilzoneAg doesn't have this variable in get_variables)
        if self._iter_aet_flag:
            self.ag_irrigation_hru_source[:] = (
                self.ag_irrigation_add * self.ag_frac
            )
        elif hasattr(self, "ag_irrigation_hru_source"):
            self.ag_irrigation_hru_source[:] = zero

        return

    @staticmethod
    def _calculate_numpy(
        soil_iter,
        iter_aet_flag,
        pref_flow_flag,
        snow_free,
        compute_soilmoist,
        compute_szactet,
        compute_interflow,
        compute_gwflow,
        # Parameters
        cov_type,
        ag_cov_type,
        fastcoef_lin,
        fastcoef_sq,
        hru_frac_perv,
        hru_type,
        ag_frac,
        ag_area,
        hru_area,
        hru_area_perv,
        pref_flow_den,
        pref_flow_infil_frac,
        sat_threshold,
        slowcoef_lin,
        slowcoef_sq,
        soil_moist_max,
        soil_rechr_max,
        soil_type,
        ag_soil_type,
        ag_soil_moist_max,
        ag_soil_rechr_max,
        soil2gw_max,
        ag_soil2gw_max,
        ssr2gw_exp,
        ssr2gw_rate,
        ag_soilwater_deficit_min,
        soilzone_aet_converge,
        # Inputs
        dprst_evap_hru,
        dprst_seep_hru,
        hru_impervevap,
        hru_intcpevap,
        infil,
        infil_ag,
        potet,
        transp_on,
        snow_evap,
        snowcov_area,
        aet_external,
        ag_irrigation_add,
        # State variables
        soil_moist,
        soil_rechr,
        ag_soil_moist,
        ag_soil_rechr,
        pref_flow_stor,
        slow_stor,
        ssres_stor,
        # Previous timestep values for storage changes
        pref_flow_stor_prev,
        soil_rechr_prev,
        soil_lower_prev,
        slow_stor_prev,
        ag_soil_moist_prev,
        ag_soil_rechr_prev,
        # Output variables
        soil_to_gw,
        soil_to_ssr,
        perv_soil_to_gw,
        perv_soil_to_gvr,
        ag_soil_to_gw,
        ag_soil_to_gvr,
        ssr_to_gw,
        slow_flow,
        ssres_flow,
        potet_rechr,
        potet_lower,
        ag_potet_rechr,
        ag_potet_lower,
        cap_waterin,
        cap_infil_tot,
        ag_cap_infil_tot,
        # ag_water_in,
        pref_flow_in,
        pref_flow_infil,
        pref_flow,
        perv_actet,
        perv_actet_hru,
        ag_actet,
        hru_ag_actet,
        hru_actet,
        soil_lower,
        ag_soil_lower,
        dunnian_flow,
        pref_flow_max,
        pref_flow_thrsh,
        soil_lower_max,
        ag_soil_lower_stor_max,
        soil_zone_max,
        soil_moist_tot,
        ssres_in,
        recharge,
        unused_potet,
        unused_ag_et,
        ag_hortonian,
        ag_soil_saturated,
        ag_soilwater_deficit,
        swale_actet,
        soil_saturated,
        sroff,
        soil_lower_ratio,
        pref_flow_stor_change,
        soil_lower_change,
        soil_lower_change_hru,
        soil_rechr_change,
        soil_rechr_change_hru,
        slow_stor_change,
        ag_soil_moist_change,
        ag_soil_moist_change_hru,
        ag_soil_rechr_change,
        ag_soil_rechr_change_hru,
        ag_soil_lower_change,
        ag_soil_lower_change_hru,
        ag_soil_moist_redistribution,
        ag_soil_rechr_redistribution,
        soil_rechr_redistribution,
        soil_lower_redistribution,
        slow_stor_redistribution,
    ) -> tuple:
        """Numpy-based calculation for agricultural soilzone.

        Fortran reference: szrun_ag() main HRU loop

        This implements the soil water balance for both pervious and
        agricultural areas of each HRU.
        """
        nhru = len(hru_type)

        # Initialize outputs for this iteration
        add_estimated_irrigation = False
        num_hrus_ag_iter = 0
        unsatisfied_big = 0.0

        # Initialize per-timestep variables
        soil_to_gw[:] = zero
        soil_to_ssr[:] = zero
        perv_soil_to_gw[:] = zero
        perv_soil_to_gvr[:] = zero
        ag_soil_to_gw[:] = zero
        ag_soil_to_gvr[:] = zero
        ssr_to_gw[:] = zero
        slow_flow[:] = zero
        ssres_flow[:] = zero
        potet_rechr[:] = zero
        potet_lower[:] = zero
        ag_potet_rechr[:] = zero
        ag_potet_lower[:] = zero
        ag_actet[:] = zero
        hru_ag_actet[:] = zero
        ag_soil_saturated[:] = zero
        ag_hortonian[:] = zero
        unused_ag_et[:] = zero
        ag_soilwater_deficit[:] = zero
        ag_cap_infil_tot[:] = zero
        # ag_water_in[:] = zero
        ag_soil_lower[:] = zero
        swale_actet[:] = zero
        soil_saturated[:] = zero
        dunnian_flow[:] = zero
        pref_flow_infil[:] = zero
        pref_flow_in[:] = zero
        pref_flow[:] = zero

        # Fortran: DO k = 1, Active_hrus; i = Hru_route_order(k)
        # Here we process all HRUs in order
        for ihru in range(nhru):
            # Skip inactive HRUs (Fortran skips via Active_hrus)
            if hru_type[ihru] == HruType.INACTIVE.value:
                continue

            # Initial AET from impervious, interception, and snow
            # Fortran: hruactet = Hru_impervevap(i) + Hru_intcpevap(i) + Snow_evap(i)  # noqa
            hruactet = (
                hru_impervevap[ihru] + hru_intcpevap[ihru] + snow_evap[ihru]
            )
            hruactet += dprst_evap_hru[ihru]

            # Handle LAKE HRUs separately
            if hru_type[ihru] == HruType.LAKE.value:
                # Lakes don't have soil zone processes
                # Fortran: Unused_potet(i) = Potet(i) - hruactet
                unused_potet[ihru] = potet[ihru] - hruactet
                hru_actet[ihru] = hruactet
                continue

            # Get area fractions
            perv_frac = hru_frac_perv[ihru]
            agfrac = ag_frac[ihru]
            perv_area = hru_area_perv[ihru]
            agarea = ag_area[ihru]

            perv_on_flag = perv_area > 0.0
            ag_on_flag = agarea > 0.0

            # Available PET after impervious, interception, snow evap
            # Fortran: avail_potet = Potet(i) - hruactet
            avail_potet = potet[ihru] - hruactet
            if avail_potet < 0.0:
                avail_potet = 0.0
                hruactet = potet[ihru]

            # Initialize flow components
            dunnianflw = 0.0
            dunnianflw_pfr = 0.0
            # interflow = 0.0

            # ****** Add infiltration to soil
            # Fortran: capwater_maxin = Infil(i)
            capwater_maxin = infil[ihru]
            ag_water_maxin = infil_ag[ihru]

            # Add irrigation if iterating (Fortran line ~591)
            if iter_aet_flag:
                ag_water_maxin = ag_water_maxin + ag_irrigation_add[ihru]

            # Compute preferential flow (if enabled)
            # Fortran: IF ( Pref_flag == ACTIVE ) THEN
            prefflow = 0.0
            if pref_flow_flag:
                if pref_flow_infil_frac[ihru] > 0.0:
                    ag_pref_flow_maxin = 0.0
                    cap_pref_flow_maxin = 0.0

                    if ag_water_maxin > 0.0:
                        ag_pref_flow_maxin = (
                            ag_water_maxin * pref_flow_infil_frac[ihru]
                        )
                        ag_water_maxin = ag_water_maxin - ag_pref_flow_maxin
                        ag_pref_flow_maxin = ag_pref_flow_maxin * agfrac

                    if capwater_maxin > 0.0:
                        cap_pref_flow_maxin = (
                            capwater_maxin * pref_flow_infil_frac[ihru]
                        )
                        capwater_maxin = capwater_maxin - cap_pref_flow_maxin
                        cap_pref_flow_maxin = cap_pref_flow_maxin * perv_frac

                    pref_flow_maxin = cap_pref_flow_maxin + ag_pref_flow_maxin

                    # Add to preferential flow storage
                    pref_flow_stor[ihru] = (
                        pref_flow_stor[ihru] + pref_flow_maxin
                    )
                    dunnianflw_pfr = max(
                        0.0, pref_flow_stor[ihru] - pref_flow_max[ihru]
                    )

                    if dunnianflw_pfr > 0.0:
                        pref_flow_stor[ihru] = pref_flow_max[ihru]

                    pref_flow_infil[ihru] = pref_flow_maxin - dunnianflw_pfr

            # Set cap_infil_tot and ag_cap_infil_tot BEFORE compute_soilmoist
            # (Fortran lines 737-744, before line 753)
            if perv_on_flag:
                cap_infil_tot[ihru] = capwater_maxin * perv_frac
            if ag_on_flag:
                ag_cap_infil_tot[ihru] = ag_water_maxin * agfrac

            # ****** Compute soil moisture for pervious area
            # Fortran: CALL compute_soilmoist (lines ~752-758)
            # Note: compute_soilmoist modifies infil (capwater_maxin) in
            # Fortran
            perv_soil_to_gvr[ihru] = 0.0
            if perv_on_flag:
                if capwater_maxin + soil_moist[ihru] > 0.0:
                    (
                        capwater_maxin,
                        soil_moist[ihru],
                        soil_rechr[ihru],
                        perv_soil_to_gw[ihru],
                        perv_soil_to_gvr[ihru],
                    ) = compute_soilmoist(
                        perv_frac,
                        soil_moist_max[ihru],
                        soil_rechr_max[ihru],
                        soil2gw_max[ihru],
                        capwater_maxin,
                        soil_moist[ihru],
                        soil_rechr[ihru],
                        perv_soil_to_gw[ihru],
                        perv_soil_to_gvr[ihru],
                    )

            # ****** Compute soil moisture for agricultural area
            # Fortran: CALL compute_soilmoist for ag (lines ~760-767)
            # Note: compute_soilmoist modifies infil (ag_water_maxin) in
            # Fortran
            if ag_on_flag:
                if ag_water_maxin + ag_soil_moist[ihru] > 0.0:
                    (
                        ag_water_maxin,
                        ag_soil_moist[ihru],
                        ag_soil_rechr[ihru],
                        ag_soil_to_gw[ihru],
                        ag_soil_to_gvr[ihru],
                    ) = compute_soilmoist(
                        agfrac,
                        ag_soil_moist_max[ihru],
                        ag_soil_rechr_max[ihru],
                        # NOTE: Fortran uses soil2gw_max here, not
                        # ag_soil2gw_max (likely a bug in Fortran line 763)
                        soil2gw_max[ihru],
                        ag_water_maxin,
                        ag_soil_moist[ihru],
                        ag_soil_rechr[ihru],
                        ag_soil_to_gw[ihru],
                        ag_soil_to_gvr[ihru],
                    )
                    # ag_water_in[ihru] = ag_water_maxin

            # Combine soil to gw and ssr
            # Fortran: Soil_to_gw(i) = perv_soil_to_gw(i) + ag_soil_to_gw(i)
            soil_to_gw[ihru] = perv_soil_to_gw[ihru] + ag_soil_to_gw[ihru]
            soil_to_ssr[ihru] = perv_soil_to_gvr[ihru] + ag_soil_to_gvr[ihru]

            cap_waterin[ihru] = capwater_maxin * perv_frac

            # ****** Compute slow interflow and ssr_to_gw
            # Fortran: compute_interflow and compute_gwflow (simplified, no
            # GSFLOW)
            availh2o = slow_stor[ihru] + soil_to_ssr[ihru]
            topfr = 0.0

            # Compute for non-swale HRUs
            # Fortran: compute_lateral==ACTIVE
            if hru_type[ihru] != HruType.SWALE.value:
                if pref_flow_flag:
                    topfr = max(0.0, availh2o - pref_flow_thrsh[ihru])

                ssresin = soil_to_ssr[ihru] - topfr
                slow_stor[ihru] = availh2o - topfr

                # Compute slow contribution to interflow
                if slow_stor[ihru] > 0.0:
                    slow_stor[ihru], slow_flow[ihru] = compute_interflow(
                        slowcoef_lin[ihru],
                        slowcoef_sq[ihru],
                        ssresin,
                        slow_stor[ihru],
                        slow_flow[ihru],
                    )
            else:
                # Swale - no lateral flow
                slow_stor[ihru] = availh2o

            # Compute flow to groundwater
            if slow_stor[ihru] > 0.0 and ssr2gw_rate[ihru] > 0.0:
                ssr_to_gw[ihru], slow_stor[ihru] = compute_gwflow(
                    ssr2gw_rate[ihru],
                    ssr2gw_exp[ihru],
                    slow_stor[ihru],
                )

            # Compute contribution to Dunnian flow from PFR
            # Fortran: IF ( Pref_flag==ACTIVE ) THEN (lines ~909-938)
            dunnianflw_gvr = 0.0
            if pref_flow_flag:
                if pref_flow_max[ihru] > 0.0:
                    availh2o = pref_flow_stor[ihru] + topfr
                    dunnianflw_gvr = max(0.0, availh2o - pref_flow_max[ihru])
                    if dunnianflw_gvr > 0.0:
                        topfr = topfr - dunnianflw_gvr
                        if topfr < 0.0:
                            topfr = 0.0

                    pref_flow_in[ihru] = pref_flow_infil[ihru] + topfr
                    pref_flow_stor[ihru] = pref_flow_stor[ihru] + topfr

                    if pref_flow_stor[ihru] > 0.0:
                        pref_flow_stor[ihru], prefflow = compute_interflow(
                            fastcoef_lin[ihru],
                            fastcoef_sq[ihru],
                            pref_flow_in[ihru],
                            pref_flow_stor[ihru],
                            prefflow,
                        )
            elif not (pref_flow_max[ihru] > 0.0):
                if hru_type[ihru] != HruType.SWALE.value:
                    dunnianflw_gvr = topfr

            # ****** Compute actual evapotranspiration
            # Fortran: compute_szactet (lines ~952-1023)
            pervactet = 0.0
            agactet = 0.0
            ag_hruactet = 0.0
            unsatisfied_ag_et = 0.0

            # Agricultural ET
            if ag_on_flag:
                if iter_aet_flag:
                    # Use observed AET as target
                    ag_AETtarget = aet_external[ihru]
                else:
                    # Use PET as target
                    ag_AETtarget = potet[ihru]

                # Subtract only canopy interception ET
                # Fortran assumes impervious, snow, and dprst evap not in ag
                # fraction
                # Fortran: ag_avail_targetAET = ag_AETtarget - Hru_intcpevap(i)
                ag_avail_targetAET = ag_AETtarget - hru_intcpevap[ihru]
                if ag_avail_targetAET < 0.0:
                    ag_avail_targetAET = 0.0

                # if ihru == 39:
                #     breakpoint()

                if ag_avail_targetAET > 0.0:
                    # Set ag_soil_saturated flag if ag soil moisture ratio >
                    # 0.9999
                    # Fortran reference: compute_szactet line ~1448
                    # Must check BEFORE ET is subtracted from ag_soil_moist
                    # Only set when Et_type > 1 (evap or transp active)
                    # Fortran Et_type logic:
                    #   Et_type = 1 if Avail_potet < NEARZERO
                    #   Et_type = 1 if Transp_on==OFF and Snow_free < 0.01
                    #   Et_type = 3 if Transp_on==ON and Cov_type > BARESOIL
                    #   Et_type = 1 if Transp_on==ON and Cov_type==BARESOIL and
                    #       Snow_free < 0.01
                    #   Et_type = 2 if Transp_on==ON and Cov_type==BARESOIL and
                    #       Snow_free >= 0.01
                    # So Et_type > 1 when: (transp_on AND cov_type > 0) OR
                    # snow_free >= 0.01
                    ag_et_type_gt_1 = (
                        transp_on[ihru] and ag_cov_type[ihru] > 0
                    ) or snow_free[ihru] >= 0.01
                    if ag_et_type_gt_1:
                        ag_pcts = ag_soil_moist[ihru] / ag_soil_moist_max[ihru]
                        if ag_pcts > 0.9999:
                            ag_soil_saturated[ihru] = 1

                    # Call compute_szactet for ag area
                    (
                        ag_soil_moist[ihru],
                        ag_soil_rechr[ihru],
                        _,
                        ag_potet_rechr[ihru],
                        ag_potet_lower[ihru],
                        agactet,
                    ) = compute_szactet(
                        transp_on[ihru],
                        ag_cov_type[ihru],
                        ag_soil_type[ihru],
                        ag_soil_moist_max[ihru],
                        ag_soil_rechr_max[ihru],
                        snow_free[ihru],
                        ag_soil_moist[ihru],
                        ag_soil_rechr[ihru],
                        ag_avail_targetAET,
                        ag_potet_rechr[ihru],
                        ag_potet_lower[ihru],
                    )

                    ag_hruactet = agactet * agfrac

                unsatisfied_ag_et = ag_avail_targetAET - agactet
                unused_ag_et[ihru] = unsatisfied_ag_et

                # Add back canopy interception (GSFLOW 2.4.1 behavior)
                # Fortran: Ag_actet(i) = agactet + Hru_intcpevap(i)
                ag_actet[ihru] = agactet + hru_intcpevap[ihru]

            # Pervious ET
            avail_potet = potet[ihru] - hruactet - ag_hruactet
            if avail_potet < 0.0:
                avail_potet = 0.0

            if soil_moist[ihru] > 0.0 and avail_potet > 0.0:
                # Set soil_saturated flag if soil moisture ratio > 0.9999
                # Fortran reference: compute_szactet line ~1448
                # Must check BEFORE ET is subtracted from soil_moist
                # Only set when Et_type > 1 (evap or transp active)
                # Fortran Et_type logic:
                #   Et_type = 1 if Avail_potet < NEARZERO
                #   Et_type = 1 if Transp_on==OFF and Snow_free < 0.01
                #   Et_type = 3 if Transp_on==ON and Cov_type > BARESOIL
                #   Et_type = 1 if Transp_on==ON and Cov_type==BARESOIL and
                #       Snow_free < 0.01
                #   Et_type = 2 if Transp_on==ON and Cov_type==BARESOIL and
                #       Snow_free >= 0.01
                # So Et_type > 1 when: (transp_on AND cov_type > 0) OR
                #       snow_free >= 0.01
                # IF ( pcts>0.9999 ) Soil_saturated = 1
                et_type_gt_1 = (
                    transp_on[ihru] and cov_type[ihru] > 0
                ) or snow_free[ihru] >= 0.01
                if et_type_gt_1:
                    pcts = soil_moist[ihru] / soil_moist_max[ihru]
                    if pcts > 0.9999:
                        soil_saturated[ihru] = 1

                (
                    soil_moist[ihru],
                    soil_rechr[ihru],
                    _,
                    potet_rechr[ihru],
                    potet_lower[ihru],
                    pervactet,
                ) = compute_szactet(
                    transp_on[ihru],
                    cov_type[ihru],
                    soil_type[ihru],
                    soil_moist_max[ihru],
                    soil_rechr_max[ihru],
                    snow_free[ihru],
                    soil_moist[ihru],
                    soil_rechr[ihru],
                    avail_potet,
                    potet_rechr[ihru],
                    potet_lower[ihru],
                )

            # Combine ET components
            perv_actet_hru[ihru] = pervactet * perv_frac
            hru_ag_actet[ihru] = ag_hruactet
            hru_actet[ihru] = hruactet + perv_actet_hru[ihru] + ag_hruactet
            perv_actet[ihru] = pervactet

            # Soil moisture accounting
            # Fortran: Soil_lower(i) = Soil_moist(i) - Soil_rechr(i)
            soil_lower[ihru] = soil_moist[ihru] - soil_rechr[ihru]

            # Compute interflow and excess flow
            # Fortran: IF ( compute_lateral==ACTIVE ) THEN (lines ~1009-1048)
            if hru_type[ihru] != HruType.SWALE.value:
                # interflow = slow_flow[ihru] + prefflow
                dunnianflw = dunnianflw_gvr + dunnianflw_pfr
                dunnian_flow[ihru] = dunnianflw

                # Treat pref_flow as interflow
                ssres_flow[ihru] = slow_flow[ihru]
                if pref_flow_flag:
                    if pref_flow_max[ihru] > 0.0:
                        pref_flow[ihru] = prefflow
                        ssres_flow[ihru] = ssres_flow[ihru] + prefflow

                # Treat dunnianflw as surface runoff
                sroff[ihru] = sroff[ihru] + dunnian_flow[ihru]
                ssres_stor[ihru] = slow_stor[ihru]
                if pref_flow_flag:
                    ssres_stor[ihru] = ssres_stor[ihru] + pref_flow_stor[ihru]

            else:
                # Swale HRU
                availh2o = slow_stor[ihru] - sat_threshold[ihru]
                swale_actet[ihru] = 0.0
                if availh2o > 0.0:
                    unsatisfied_et = potet[ihru] - hru_actet[ihru]
                    if unsatisfied_et > 0.0:
                        availh2o = min(availh2o, unsatisfied_et)
                        swale_actet[ihru] = availh2o
                        hru_actet[ihru] = hru_actet[ihru] + swale_actet[ihru]
                        slow_stor[ihru] = slow_stor[ihru] - swale_actet[ihru]

                ssres_stor[ihru] = slow_stor[ihru]

            # Soil lower ratio
            if soil_lower_max[ihru] > 0.0:
                soil_lower_ratio[ihru] = (
                    soil_lower[ihru] / soil_lower_max[ihru]
                )
                # Cap ratio at 1.0 if excess is due to rounding error (< 1e-4)
                # This handles accumulated floating-point differences between
                # Python (double precision) and Fortran (single precision)
                if soil_lower_ratio[ihru] > 1.0:
                    excess = soil_lower_ratio[ihru] - 1.0
                    if excess < 1e-4:
                        # Cap only small excesses due to floating-point error
                        soil_lower_ratio[ihru] = 1.0
                        soil_lower[ihru] = soil_lower_max[ihru]
                    # else:
                    #     # Fortran prints error but does NOT cap when excess
                    #     # >= 1e-4. This can happen in reanalysis settings
                    #     # with dynamic parameters (Fortran STOP is commented)
                    #     warnings.warn(
                    #         f"HRU {ihru}: soil_lower exceeds soil_lower_max "
                    #         f"by {excess:.2e} (ratio = "
                    #         f"{soil_lower_ratio[ihru]:.6f}). "
                    #         f"This may indicate a mass balance error.",
                    #         UserWarning,
                    #     )

            ssres_in[ihru] = soil_to_ssr[ihru]
            if pref_flow_flag:
                ssres_in[ihru] = ssres_in[ihru] + pref_flow_infil[ihru]

            unused_potet[ihru] = potet[ihru] - hru_actet[ihru]

            # Total soil moisture
            soil_moist_tot[ihru] = (
                ssres_stor[ihru]
                + soil_moist[ihru] * perv_frac
                + ag_soil_moist[ihru] * agfrac
            )

            # Recharge
            recharge[ihru] = soil_to_gw[ihru] + ssr_to_gw[ihru]
            recharge[ihru] = recharge[ihru] + dprst_seep_hru[ihru]

            # Agricultural soil lower
            if ag_on_flag:
                ag_soil_lower[ihru] = ag_soil_moist[ihru] - ag_soil_rechr[ihru]

                # Check for irrigation need (Fortran lines ~1127-1160)
                if iter_aet_flag and transp_on[ihru]:
                    if unsatisfied_ag_et > soilzone_aet_converge:
                        ag_soilwater_deficit[ihru] = (
                            ag_soil_moist_max[ihru] - ag_soil_moist[ihru]
                        ) / ag_soil_moist_max[ihru]

                        if (
                            ag_soilwater_deficit[ihru]
                            > ag_soilwater_deficit_min[ihru]
                        ):
                            unsatisfied_max = unsatisfied_ag_et
                            if unsatisfied_ag_et > ag_soil_moist_max[ihru]:
                                unsatisfied_max = ag_soil_moist_max[ihru]
                            else:
                                # Speed up convergence after 20 iterations
                                if soil_iter > 20:
                                    unsatisfied_max = (
                                        unsatisfied_max + unsatisfied_ag_et
                                    )
                                add_estimated_irrigation = True
                                num_hrus_ag_iter += 1

                            ag_irrigation_add[ihru] = (
                                ag_irrigation_add[ihru] + unsatisfied_max
                            )

                            if unsatisfied_max > unsatisfied_big:
                                unsatisfied_big = unsatisfied_max

            # if ihru == 439:
            #     asdf

        # End HRU loop

        # Calculate storage changes for mass budget
        # Fortran reference: Not explicitly in szrun_ag, but needed for budget
        #
        # Subtract redistributions to get only hydrologic process changes.
        # This ensures mass balance is correct by excluding area-accounting
        # adjustments that occur when ag_frac changes dynamically.
        pref_flow_stor_change[:] = pref_flow_stor - pref_flow_stor_prev
        soil_lower_change[:] = (
            soil_lower - soil_lower_prev - soil_lower_redistribution
        )
        soil_rechr_change[:] = (
            soil_rechr - soil_rechr_prev - soil_rechr_redistribution
        )
        slow_stor_change[:] = (
            slow_stor - slow_stor_prev - slow_stor_redistribution
        )

        # Convert to HRU basis (multiply by area fractions)
        soil_lower_change_hru[:] = soil_lower_change * hru_frac_perv
        soil_rechr_change_hru[:] = soil_rechr_change * hru_frac_perv

        # Agricultural soil moisture storage changes
        # Subtract redistributions to get only hydrologic process changes
        ag_soil_moist_change[:] = (
            ag_soil_moist - ag_soil_moist_prev - ag_soil_moist_redistribution
        )
        ag_soil_rechr_change[:] = (
            ag_soil_rechr - ag_soil_rechr_prev - ag_soil_rechr_redistribution
        )
        ag_soil_lower_change[:] = ag_soil_moist_change - ag_soil_rechr_change

        # Convert to HRU basis (multiply by area fractions)
        ag_soil_moist_change_hru[:] = ag_soil_moist_change * ag_frac
        ag_soil_rechr_change_hru[:] = ag_soil_rechr_change * ag_frac
        ag_soil_lower_change_hru[:] = ag_soil_lower_change * ag_frac

        # Return iteration control variables
        return (add_estimated_irrigation, num_hrus_ag_iter, unsatisfied_big)

    def _calculate_numba(
        self,
        soil_iter,
        iter_aet_flag,
        pref_flow_flag,
        snow_free,
        compute_soilmoist,
        compute_szactet,
        compute_interflow,
        compute_gwflow,
        # Parameters
        cov_type,
        ag_cov_type,
        fastcoef_lin,
        fastcoef_sq,
        hru_frac_perv,
        hru_type,
        ag_frac,
        ag_area,
        hru_area,
        hru_area_perv,
        pref_flow_den,
        pref_flow_infil_frac,
        sat_threshold,
        slowcoef_lin,
        slowcoef_sq,
        soil_moist_max,
        soil_rechr_max,
        soil_type,
        ag_soil_type,
        ag_soil_moist_max,
        ag_soil_rechr_max,
        soil2gw_max,
        ag_soil2gw_max,
        ssr2gw_exp,
        ssr2gw_rate,
        ag_soilwater_deficit_min,
        soilzone_aet_converge,
        # Inputs
        dprst_evap_hru,
        dprst_seep_hru,
        hru_impervevap,
        hru_intcpevap,
        infil,
        infil_ag,
        potet,
        transp_on,
        snow_evap,
        snowcov_area,
        aet_external,
        ag_irrigation_add,
        # State variables
        soil_moist,
        soil_rechr,
        ag_soil_moist,
        ag_soil_rechr,
        pref_flow_stor,
        slow_stor,
        ssres_stor,
        # Previous timestep values for storage changes
        pref_flow_stor_prev,
        soil_rechr_prev,
        soil_lower_prev,
        slow_stor_prev,
        ag_soil_moist_prev,
        ag_soil_rechr_prev,
        # Output variables
        soil_to_gw,
        soil_to_ssr,
        perv_soil_to_gw,
        perv_soil_to_gvr,
        ag_soil_to_gw,
        ag_soil_to_gvr,
        ssr_to_gw,
        slow_flow,
        ssres_flow,
        potet_rechr,
        potet_lower,
        ag_potet_rechr,
        ag_potet_lower,
        cap_waterin,
        cap_infil_tot,
        ag_cap_infil_tot,
        # ag_water_in,
        pref_flow_in,
        pref_flow_infil,
        pref_flow,
        perv_actet,
        perv_actet_hru,
        ag_actet,
        hru_ag_actet,
        hru_actet,
        soil_lower,
        ag_soil_lower,
        dunnian_flow,
        pref_flow_max,
        pref_flow_thrsh,
        soil_lower_max,
        ag_soil_lower_stor_max,
        soil_zone_max,
        soil_moist_tot,
        ssres_in,
        recharge,
        unused_potet,
        unused_ag_et,
        ag_hortonian,
        ag_soil_saturated,
        ag_soilwater_deficit,
        swale_actet,
        soil_saturated,
        sroff,
        soil_lower_ratio,
        pref_flow_stor_change,
        soil_lower_change,
        soil_lower_change_hru,
        soil_rechr_change,
        soil_rechr_change_hru,
        slow_stor_change,
        ag_soil_moist_change,
        ag_soil_moist_change_hru,
        ag_soil_rechr_change,
        ag_soil_rechr_change_hru,
        ag_soil_lower_change,
        ag_soil_lower_change_hru,
        ag_soil_moist_redistribution,
        ag_soil_rechr_redistribution,
        soil_rechr_redistribution,
        soil_lower_redistribution,
        slow_stor_redistribution,
    ) -> tuple:
        """Numba-optimized calculation for agricultural soilzone.

        This is a wrapper that calls the numba-compiled function.
        """
        return self._calculate_soilzone_ag_numba_compiled(
            soil_iter,
            iter_aet_flag,
            pref_flow_flag,
            snow_free,
            # Parameters
            cov_type,
            ag_cov_type,
            fastcoef_lin,
            fastcoef_sq,
            hru_frac_perv,
            hru_type,
            ag_frac,
            ag_area,
            hru_area,
            hru_area_perv,
            pref_flow_den,
            pref_flow_infil_frac,
            sat_threshold,
            slowcoef_lin,
            slowcoef_sq,
            soil_moist_max,
            soil_rechr_max,
            soil_type,
            ag_soil_type,
            ag_soil_moist_max,
            ag_soil_rechr_max,
            soil2gw_max,
            ag_soil2gw_max,
            ssr2gw_exp,
            ssr2gw_rate,
            ag_soilwater_deficit_min,
            soilzone_aet_converge,
            # Inputs
            dprst_evap_hru,
            dprst_seep_hru,
            hru_impervevap,
            hru_intcpevap,
            infil,
            infil_ag,
            potet,
            transp_on,
            snow_evap,
            snowcov_area,
            aet_external,
            ag_irrigation_add,
            # State variables
            soil_moist,
            soil_rechr,
            ag_soil_moist,
            ag_soil_rechr,
            pref_flow_stor,
            slow_stor,
            ssres_stor,
            # Previous timestep values for storage changes
            pref_flow_stor_prev,
            soil_rechr_prev,
            soil_lower_prev,
            slow_stor_prev,
            ag_soil_moist_prev,
            ag_soil_rechr_prev,
            # Output variables
            soil_to_gw,
            soil_to_ssr,
            perv_soil_to_gw,
            perv_soil_to_gvr,
            ag_soil_to_gw,
            ag_soil_to_gvr,
            ssr_to_gw,
            slow_flow,
            ssres_flow,
            potet_rechr,
            potet_lower,
            ag_potet_rechr,
            ag_potet_lower,
            cap_waterin,
            cap_infil_tot,
            ag_cap_infil_tot,
            pref_flow_in,
            pref_flow_infil,
            pref_flow,
            perv_actet,
            perv_actet_hru,
            ag_actet,
            hru_ag_actet,
            hru_actet,
            soil_lower,
            ag_soil_lower,
            dunnian_flow,
            pref_flow_max,
            pref_flow_thrsh,
            soil_lower_max,
            ag_soil_lower_stor_max,
            soil_zone_max,
            soil_moist_tot,
            ssres_in,
            recharge,
            unused_potet,
            unused_ag_et,
            ag_hortonian,
            ag_soil_saturated,
            ag_soilwater_deficit,
            swale_actet,
            soil_saturated,
            sroff,
            soil_lower_ratio,
            pref_flow_stor_change,
            soil_lower_change,
            soil_lower_change_hru,
            soil_rechr_change,
            soil_rechr_change_hru,
            slow_stor_change,
            ag_soil_moist_change,
            ag_soil_moist_change_hru,
            ag_soil_rechr_change,
            ag_soil_rechr_change_hru,
            ag_soil_lower_change,
            ag_soil_lower_change_hru,
            ag_soil_moist_redistribution,
            ag_soil_rechr_redistribution,
            soil_rechr_redistribution,
            soil_lower_redistribution,
            slow_stor_redistribution,
        )

    @staticmethod
    def _compute_soilmoist(
        perv_frac,
        soil_moist_max,
        soil_rechr_max,
        soil2gw_max,
        infil,
        soil_moist,
        soil_rechr,
        soil_to_gw,
        soil_to_ssr,
    ) -> tuple:
        """Compute soil moisture accounting.

        Fortran reference: SUBROUTINE compute_soilmoist in soilzone.f90

        This is the same as in the base PRMSSoilzone class.
        """
        # PRMSIV Step 4 (eqn 1-125)
        soil_rechr = min(soil_rechr + infil, soil_rechr_max)

        # PRMSIV Step 5
        excess = soil_moist + infil
        soil_moist = min(excess, soil_moist_max)

        # PRMSIV Step 6
        excess = (excess - soil_moist_max) * perv_frac

        if excess > 0.0:
            if soil2gw_max > 0.0:
                soil_to_gw = min(soil2gw_max, excess)
                excess = excess - soil_to_gw

            if excess > (infil * perv_frac):
                infil = 0.0
            else:
                infil = infil - (excess / perv_frac)

            soil_to_ssr = max(0.0, excess)

        return (
            infil,
            soil_moist,
            soil_rechr,
            soil_to_gw,
            soil_to_ssr,
        )

    @staticmethod
    def _compute_interflow(
        coef_lin, coef_sq, ssres_in, storage, inter_flow
    ) -> tuple:
        """Compute interflow from subsurface reservoir.

        Fortran reference: SUBROUTINE compute_interflow in soilzone.f90

        This is the same as in the base PRMSSoilzone class.
        """
        if (coef_lin <= 0.0) and (ssres_in <= 0.0):
            c1 = coef_sq * storage
            inter_flow = storage * (c1 / (1.0 + c1))

        elif (coef_lin > 0.0) and (coef_sq <= 0.0):
            c2 = 1.0 - np.exp(-coef_lin)
            inter_flow = ssres_in * (1.0 - c2 / coef_lin) + storage * c2

        elif coef_sq > 0.0:
            c3 = np.sqrt(coef_lin**2.0 + 4.0 * coef_sq * ssres_in)
            sos = storage - ((c3 - coef_lin) / (2.0 * coef_sq))

            if c3 == 0.0:
                msg = (
                    "ERROR, in compute_interflow sos=0, "
                    "please contact code developers"
                )
                raise RuntimeError(msg)

            c1 = coef_sq * sos / c3
            c2 = 1.0 - np.exp(-c3)

            if (1.0 + c1 * c2) > 0.0:
                inter_flow = ssres_in + (
                    (sos * (1.0 + c1) * c2) / (1.0 + c1 * c2)
                )
            else:
                inter_flow = ssres_in
        else:
            inter_flow = 0.0

        if inter_flow < 0.0:
            inter_flow = 0.0
        elif inter_flow > storage:
            inter_flow = storage

        storage = storage - inter_flow
        return storage, inter_flow

    @staticmethod
    def _compute_gwflow(ssr2gw_rate, ssr2gw_exp, slow_stor) -> tuple:
        """Compute flow to groundwater.

        Fortran reference: SUBROUTINE compute_gwflow in soilzone.f90

        This is the same as in the base PRMSSoilzone class.
        """
        ssr_to_gw = max(0.0, ssr2gw_rate * slow_stor**ssr2gw_exp)
        ssr_to_gw = min(ssr_to_gw, slow_stor)
        slow_stor = slow_stor - ssr_to_gw
        return ssr_to_gw, slow_stor

    @staticmethod
    def _compute_szactet(
        transp_on,
        cov_type,
        soil_type,
        soil_moist_max,
        soil_rechr_max,
        snow_free,
        soil_moist,
        soil_rechr,
        avail_potet,
        potet_rechr,
        potet_lower,
    ) -> tuple:
        """Compute actual ET from soil zone.

        Fortran reference: SUBROUTINE compute_szactet in soilzone.f90

        This function determines the evapotranspiration type (Et_type) and
        computes actual ET based on soil moisture availability and soil type.

        Et_type Logic
        -------------
        The Et_type determines whether ET occurs and how it's computed:

        - Et_type = 1 (ET_DEFAULT): No ET computed (et = 0)
        - Et_type = 2 (EVAP_ONLY): Evaporation only (scaled by snow_free)
        - Et_type = 3 (EVAP_PLUS_TRANSP): Full evapotranspiration

        Et_type is determined by:

        1. If avail_potet < NEARZERO → Et_type = 1
        2. If transp_on == OFF:
           - If snow_free < 0.01 → Et_type = 1
           - Else → Et_type = 2
        3. If transp_on == ON and cov_type > BARESOIL → Et_type = 3
        4. If transp_on == ON and cov_type == BARESOIL:
           - If snow_free < 0.01 → Et_type = 1
           - Else → Et_type = 2

        Note: The soil_saturated flag check (pcts > 0.9999) is performed in
        the calling code BEFORE this function is called, and only when
        Et_type > 1. The condition for Et_type > 1 is:
        ``(transp_on AND cov_type > 0) OR snow_free >= 0.01``

        Returns
        -------
        tuple
            (soil_moist, soil_rechr, avail_potet, potet_rechr, potet_lower, et)
        """
        from ..constants import ETType, SoilType, nearzero

        # Determine type of evapotranspiration
        if avail_potet < nearzero:
            et_type = ETType.ET_DEFAULT.value
            avail_potet = 0.0

        elif not transp_on:
            if snow_free < 0.01:
                et_type = ETType.ET_DEFAULT.value
            else:
                et_type = ETType.EVAP_ONLY.value

        elif cov_type > 0:
            et_type = ETType.EVAP_PLUS_TRANSP.value

        elif snow_free < 0.01:
            et_type = ETType.ET_DEFAULT.value

        else:
            et_type = ETType.EVAP_ONLY.value

        # Compute ET based on type
        if et_type in [ETType.EVAP_ONLY.value, ETType.EVAP_PLUS_TRANSP.value]:
            pcts = soil_moist / soil_moist_max
            pctr = soil_rechr / soil_rechr_max
            potet_lower = avail_potet
            potet_rechr = avail_potet

            if soil_type == SoilType.SAND.value:
                if pcts < 0.25:
                    potet_lower = 0.5 * pcts * avail_potet
                if pctr < 0.25:
                    potet_rechr = 0.5 * pctr * avail_potet

            elif soil_type == SoilType.LOAM.value:
                if pcts < 0.5:
                    potet_lower = pcts * avail_potet
                if pctr < 0.5:
                    potet_rechr = pctr * avail_potet

            elif soil_type == SoilType.CLAY.value:
                if (pcts < TWOTHIRDS) and (pcts > ONETHIRD):
                    potet_lower = pcts * avail_potet
                elif pcts <= ONETHIRD:
                    potet_lower = 0.5 * pcts * avail_potet

                if (pctr < TWOTHIRDS) and (pctr > ONETHIRD):
                    potet_rechr = pctr * avail_potet
                elif pctr <= ONETHIRD:
                    potet_rechr = 0.5 * pctr * avail_potet

            # Soil moisture accounting
            if et_type == ETType.EVAP_ONLY.value:
                potet_rechr = potet_rechr * snow_free

            if potet_rechr > soil_rechr:
                potet_rechr = soil_rechr
                soil_rechr = 0.0
            else:
                soil_rechr = soil_rechr - potet_rechr

            if (et_type == ETType.EVAP_ONLY.value) or (
                potet_rechr >= potet_lower
            ):
                if potet_rechr > soil_moist:
                    potet_rechr = soil_moist
                    soil_moist = 0.0
                else:
                    soil_moist = soil_moist - potet_rechr

                et = potet_rechr

            elif potet_lower > soil_moist:
                et = soil_moist
                soil_moist = 0.0
            else:
                soil_moist = soil_moist - potet_lower
                et = potet_lower

            if soil_rechr > soil_moist:
                soil_rechr = soil_moist

        else:
            et = 0.0

        return (
            soil_moist,
            soil_rechr,
            avail_potet,
            potet_rechr,
            potet_lower,
            et,
        )


# ============================================================================
# Numba-optimized functions
# ============================================================================


@nb.jit(nopython=True)
def _compute_soilmoist_numba(
    perv_frac,
    soil_moist_max,
    soil_rechr_max,
    soil2gw_max,
    infil,
    soil_moist,
    soil_rechr,
    soil_to_gw,
    soil_to_ssr,
):
    """Numba-optimized soil moisture accounting."""
    # PRMSIV Step 4 (eqn 1-125)
    soil_rechr = min(soil_rechr + infil, soil_rechr_max)

    # PRMSIV Step 5
    excess = soil_moist + infil
    soil_moist = min(excess, soil_moist_max)

    # PRMSIV Step 6
    excess = (excess - soil_moist_max) * perv_frac

    if excess > 0.0:
        if soil2gw_max > 0.0:
            soil_to_gw = min(soil2gw_max, excess)
            excess = excess - soil_to_gw

        if excess > (infil * perv_frac):
            infil = 0.0
        else:
            infil = infil - (excess / perv_frac)

        soil_to_ssr = max(0.0, excess)

    return (
        infil,
        soil_moist,
        soil_rechr,
        soil_to_gw,
        soil_to_ssr,
    )


@nb.jit(nopython=True)
def _compute_interflow_numba(coef_lin, coef_sq, ssres_in, storage, inter_flow):
    """Numba-optimized interflow computation."""
    if (coef_lin <= 0.0) and (ssres_in <= 0.0):
        c1 = coef_sq * storage
        inter_flow = storage * (c1 / (1.0 + c1))

    elif (coef_lin > 0.0) and (coef_sq <= 0.0):
        c2 = 1.0 - np.exp(-coef_lin)
        inter_flow = ssres_in * (1.0 - c2 / coef_lin) + storage * c2

    elif coef_sq > 0.0:
        c3 = np.sqrt(coef_lin**2.0 + 4.0 * coef_sq * ssres_in)
        sos = storage - ((c3 - coef_lin) / (2.0 * coef_sq))

        if c3 == 0.0:
            # In numba, we can't raise RuntimeError, so return 0
            inter_flow = 0.0
        else:
            c1 = coef_sq * sos / c3
            c2 = 1.0 - np.exp(-c3)

            if (1.0 + c1 * c2) > 0.0:
                inter_flow = ssres_in + (
                    (sos * (1.0 + c1) * c2) / (1.0 + c1 * c2)
                )
            else:
                inter_flow = ssres_in
    else:
        inter_flow = 0.0

    if inter_flow < 0.0:
        inter_flow = 0.0
    elif inter_flow > storage:
        inter_flow = storage

    storage = storage - inter_flow
    return storage, inter_flow


@nb.jit(nopython=True)
def _compute_gwflow_numba(ssr2gw_rate, ssr2gw_exp, slow_stor):
    """Numba-optimized groundwater flow computation."""
    ssr_to_gw = max(0.0, ssr2gw_rate * slow_stor**ssr2gw_exp)
    ssr_to_gw = min(ssr_to_gw, slow_stor)
    slow_stor = slow_stor - ssr_to_gw
    return ssr_to_gw, slow_stor


@nb.jit(nopython=True)
def _compute_szactet_numba(
    transp_on,
    cov_type,
    soil_type,
    soil_moist_max,
    soil_rechr_max,
    snow_free,
    soil_moist,
    soil_rechr,
    avail_potet,
    potet_rechr,
    potet_lower,
):
    """Numba-optimized actual ET computation from soil zone."""
    nearzero = 1e-6

    # Determine type of evapotranspiration
    if avail_potet < nearzero:
        et_type = 1  # ET_DEFAULT
        avail_potet = 0.0
    elif not transp_on:
        if snow_free < 0.01:
            et_type = 1  # ET_DEFAULT
        else:
            et_type = 2  # EVAP_ONLY
    elif cov_type > 0:
        et_type = 3  # EVAP_PLUS_TRANSP
    elif snow_free < 0.01:
        et_type = 1  # ET_DEFAULT
    else:
        et_type = 2  # EVAP_ONLY

    # Compute ET based on type
    if et_type in [2, 3]:  # EVAP_ONLY or EVAP_PLUS_TRANSP
        pcts = soil_moist / soil_moist_max
        pctr = soil_rechr / soil_rechr_max
        potet_lower = avail_potet
        potet_rechr = avail_potet

        if soil_type == 1:  # SAND
            if pcts < 0.25:
                potet_lower = 0.5 * pcts * avail_potet
            if pctr < 0.25:
                potet_rechr = 0.5 * pctr * avail_potet

        elif soil_type == 2:  # LOAM
            if pcts < 0.5:
                potet_lower = pcts * avail_potet
            if pctr < 0.5:
                potet_rechr = pctr * avail_potet

        elif soil_type == 3:  # CLAY
            if (pcts < TWOTHIRDS) and (pcts > ONETHIRD):
                potet_lower = pcts * avail_potet
            elif pcts <= ONETHIRD:
                potet_lower = 0.5 * pcts * avail_potet

            if (pctr < TWOTHIRDS) and (pctr > ONETHIRD):
                potet_rechr = pctr * avail_potet
            elif pctr <= ONETHIRD:
                potet_rechr = 0.5 * pctr * avail_potet

        # Soil moisture accounting
        if et_type == 2:  # EVAP_ONLY
            potet_rechr = potet_rechr * snow_free

        if potet_rechr > soil_rechr:
            potet_rechr = soil_rechr
            soil_rechr = 0.0
        else:
            soil_rechr = soil_rechr - potet_rechr

        if (et_type == 2) or (potet_rechr >= potet_lower):  # EVAP_ONLY
            if potet_rechr > soil_moist:
                potet_rechr = soil_moist
                soil_moist = 0.0
            else:
                soil_moist = soil_moist - potet_rechr

            et = potet_rechr

        elif potet_lower > soil_moist:
            et = soil_moist
            soil_moist = 0.0
        else:
            soil_moist = soil_moist - potet_lower
            et = potet_lower

        if soil_rechr > soil_moist:
            soil_rechr = soil_moist

    else:
        et = 0.0

    return (
        soil_moist,
        soil_rechr,
        avail_potet,
        potet_rechr,
        potet_lower,
        et,
    )


def _calculate_soilzone_ag_numba(
    soil_iter,
    iter_aet_flag,
    pref_flow_flag,
    snow_free,
    # Parameters
    cov_type,
    ag_cov_type,
    fastcoef_lin,
    fastcoef_sq,
    hru_frac_perv,
    hru_type,
    ag_frac,
    ag_area,
    hru_area,
    hru_area_perv,
    pref_flow_den,
    pref_flow_infil_frac,
    sat_threshold,
    slowcoef_lin,
    slowcoef_sq,
    soil_moist_max,
    soil_rechr_max,
    soil_type,
    ag_soil_type,
    ag_soil_moist_max,
    ag_soil_rechr_max,
    soil2gw_max,
    ag_soil2gw_max,
    ssr2gw_exp,
    ssr2gw_rate,
    ag_soilwater_deficit_min,
    soilzone_aet_converge,
    # Inputs
    dprst_evap_hru,
    dprst_seep_hru,
    hru_impervevap,
    hru_intcpevap,
    infil,
    infil_ag,
    potet,
    transp_on,
    snow_evap,
    snowcov_area,
    aet_external,
    ag_irrigation_add,
    # State variables
    soil_moist,
    soil_rechr,
    ag_soil_moist,
    ag_soil_rechr,
    pref_flow_stor,
    slow_stor,
    ssres_stor,
    # Previous timestep values for storage changes
    pref_flow_stor_prev,
    soil_rechr_prev,
    soil_lower_prev,
    slow_stor_prev,
    ag_soil_moist_prev,
    ag_soil_rechr_prev,
    # Output variables
    soil_to_gw,
    soil_to_ssr,
    perv_soil_to_gw,
    perv_soil_to_gvr,
    ag_soil_to_gw,
    ag_soil_to_gvr,
    ssr_to_gw,
    slow_flow,
    ssres_flow,
    potet_rechr,
    potet_lower,
    ag_potet_rechr,
    ag_potet_lower,
    cap_waterin,
    cap_infil_tot,
    ag_cap_infil_tot,
    pref_flow_in,
    pref_flow_infil,
    pref_flow,
    perv_actet,
    perv_actet_hru,
    ag_actet,
    hru_ag_actet,
    hru_actet,
    soil_lower,
    ag_soil_lower,
    dunnian_flow,
    pref_flow_max,
    pref_flow_thrsh,
    soil_lower_max,
    ag_soil_lower_stor_max,
    soil_zone_max,
    soil_moist_tot,
    ssres_in,
    recharge,
    unused_potet,
    unused_ag_et,
    ag_hortonian,
    ag_soil_saturated,
    ag_soilwater_deficit,
    swale_actet,
    soil_saturated,
    sroff,
    soil_lower_ratio,
    pref_flow_stor_change,
    soil_lower_change,
    soil_lower_change_hru,
    soil_rechr_change,
    soil_rechr_change_hru,
    slow_stor_change,
    ag_soil_moist_change,
    ag_soil_moist_change_hru,
    ag_soil_rechr_change,
    ag_soil_rechr_change_hru,
    ag_soil_lower_change,
    ag_soil_lower_change_hru,
    ag_soil_moist_redistribution,
    ag_soil_rechr_redistribution,
    soil_rechr_redistribution,
    soil_lower_redistribution,
    slow_stor_redistribution,
):
    """Numba-optimized calculation for agricultural soilzone.

    This implements the soil water balance for both pervious and
    agricultural areas of each HRU with significant performance improvements
    through numba JIT compilation.
    """
    nhru = len(hru_type)

    # Initialize outputs for this iteration
    add_estimated_irrigation = False
    num_hrus_ag_iter = 0
    unsatisfied_big = 0.0

    # Initialize per-timestep variables
    soil_to_gw[:] = 0.0
    soil_to_ssr[:] = 0.0
    perv_soil_to_gw[:] = 0.0
    perv_soil_to_gvr[:] = 0.0
    ag_soil_to_gw[:] = 0.0
    ag_soil_to_gvr[:] = 0.0
    ssr_to_gw[:] = 0.0
    slow_flow[:] = 0.0
    ssres_flow[:] = 0.0
    potet_rechr[:] = 0.0
    potet_lower[:] = 0.0
    ag_potet_rechr[:] = 0.0
    ag_potet_lower[:] = 0.0
    ag_actet[:] = 0.0
    hru_ag_actet[:] = 0.0
    ag_soil_saturated[:] = 0.0
    ag_hortonian[:] = 0.0
    unused_ag_et[:] = 0.0
    ag_soilwater_deficit[:] = 0.0
    ag_cap_infil_tot[:] = 0.0
    ag_soil_lower[:] = 0.0
    swale_actet[:] = 0.0
    soil_saturated[:] = 0.0
    dunnian_flow[:] = 0.0
    pref_flow_infil[:] = 0.0
    pref_flow_in[:] = 0.0
    pref_flow[:] = 0.0

    # Arrays to track which HRUs need irrigation (for post-loop reduction)
    # Prevents race conditions when multiple threads try to update shared
    # variables
    hru_needs_irrigation = np.empty(nhru, dtype=np.int32)
    hru_needs_irrigation[:] = 0
    hru_unsatisfied_max = np.empty(nhru, dtype=np.float64)
    hru_unsatisfied_max[:] = 0.0

    # HRU loop - uses prange for parallel execution when numba_num_threads > 1
    for ihru in nb.prange(nhru):
        # Skip inactive HRUs
        if hru_type[ihru] == 0:  # INACTIVE
            continue

        # Initial AET from impervious, interception, and snow
        hruactet = hru_impervevap[ihru] + hru_intcpevap[ihru] + snow_evap[ihru]
        hruactet += dprst_evap_hru[ihru]

        # Handle LAKE HRUs separately
        if hru_type[ihru] == 2:  # LAKE
            unused_potet[ihru] = potet[ihru] - hruactet
            hru_actet[ihru] = hruactet
            continue

        # Get area fractions
        perv_frac = hru_frac_perv[ihru]
        agfrac = ag_frac[ihru]
        perv_area = hru_area_perv[ihru]
        agarea = ag_area[ihru]

        perv_on_flag = perv_area > 0.0
        ag_on_flag = agarea > 0.0

        # Available PET after impervious, interception, snow evap
        avail_potet = potet[ihru] - hruactet
        if avail_potet < 0.0:
            avail_potet = 0.0
            hruactet = potet[ihru]

        # Initialize flow components
        dunnianflw = 0.0
        dunnianflw_pfr = 0.0

        # Add infiltration to soil
        capwater_maxin = infil[ihru]
        ag_water_maxin = infil_ag[ihru]

        # Add irrigation if iterating
        if iter_aet_flag:
            ag_water_maxin = ag_water_maxin + ag_irrigation_add[ihru]

        # Compute preferential flow (if enabled)
        prefflow = 0.0
        if pref_flow_flag:
            if pref_flow_infil_frac[ihru] > 0.0:
                ag_pref_flow_maxin = 0.0
                cap_pref_flow_maxin = 0.0

                if ag_water_maxin > 0.0:
                    ag_pref_flow_maxin = (
                        ag_water_maxin * pref_flow_infil_frac[ihru]
                    )
                    ag_water_maxin = ag_water_maxin - ag_pref_flow_maxin
                    ag_pref_flow_maxin = ag_pref_flow_maxin * agfrac

                if capwater_maxin > 0.0:
                    cap_pref_flow_maxin = (
                        capwater_maxin * pref_flow_infil_frac[ihru]
                    )
                    capwater_maxin = capwater_maxin - cap_pref_flow_maxin
                    cap_pref_flow_maxin = cap_pref_flow_maxin * perv_frac

                pref_flow_maxin = cap_pref_flow_maxin + ag_pref_flow_maxin

                # Add to preferential flow storage
                pref_flow_stor[ihru] = pref_flow_stor[ihru] + pref_flow_maxin
                dunnianflw_pfr = max(
                    0.0, pref_flow_stor[ihru] - pref_flow_max[ihru]
                )

                if dunnianflw_pfr > 0.0:
                    pref_flow_stor[ihru] = pref_flow_max[ihru]

                pref_flow_infil[ihru] = pref_flow_maxin - dunnianflw_pfr

        # Set cap_infil_tot and ag_cap_infil_tot BEFORE compute_soilmoist
        if perv_on_flag:
            cap_infil_tot[ihru] = capwater_maxin * perv_frac
        if ag_on_flag:
            ag_cap_infil_tot[ihru] = ag_water_maxin * agfrac

        # Compute soil moisture for pervious area
        perv_soil_to_gvr[ihru] = 0.0
        if perv_on_flag:
            if capwater_maxin + soil_moist[ihru] > 0.0:
                (
                    capwater_maxin,
                    soil_moist[ihru],
                    soil_rechr[ihru],
                    perv_soil_to_gw[ihru],
                    perv_soil_to_gvr[ihru],
                ) = _compute_soilmoist_numba(
                    perv_frac,
                    soil_moist_max[ihru],
                    soil_rechr_max[ihru],
                    soil2gw_max[ihru],
                    capwater_maxin,
                    soil_moist[ihru],
                    soil_rechr[ihru],
                    perv_soil_to_gw[ihru],
                    perv_soil_to_gvr[ihru],
                )

        # Compute soil moisture for agricultural area
        if ag_on_flag:
            if ag_water_maxin + ag_soil_moist[ihru] > 0.0:
                (
                    ag_water_maxin,
                    ag_soil_moist[ihru],
                    ag_soil_rechr[ihru],
                    ag_soil_to_gw[ihru],
                    ag_soil_to_gvr[ihru],
                ) = _compute_soilmoist_numba(
                    agfrac,
                    ag_soil_moist_max[ihru],
                    ag_soil_rechr_max[ihru],
                    soil2gw_max[ihru],
                    ag_water_maxin,
                    ag_soil_moist[ihru],
                    ag_soil_rechr[ihru],
                    ag_soil_to_gw[ihru],
                    ag_soil_to_gvr[ihru],
                )

        # Combine soil to gw and ssr
        soil_to_gw[ihru] = perv_soil_to_gw[ihru] + ag_soil_to_gw[ihru]
        soil_to_ssr[ihru] = perv_soil_to_gvr[ihru] + ag_soil_to_gvr[ihru]

        cap_waterin[ihru] = capwater_maxin * perv_frac

        # Compute slow interflow and ssr_to_gw
        availh2o = slow_stor[ihru] + soil_to_ssr[ihru]
        topfr = 0.0

        # Compute for non-swale HRUs
        if hru_type[ihru] != 3:  # SWALE
            if pref_flow_flag:
                topfr = max(0.0, availh2o - pref_flow_thrsh[ihru])

            ssresin = soil_to_ssr[ihru] - topfr
            slow_stor[ihru] = availh2o - topfr

            # Compute slow contribution to interflow
            if slow_stor[ihru] > 0.0:
                slow_stor[ihru], slow_flow[ihru] = _compute_interflow_numba(
                    slowcoef_lin[ihru],
                    slowcoef_sq[ihru],
                    ssresin,
                    slow_stor[ihru],
                    slow_flow[ihru],
                )
        else:
            # Swale - no lateral flow
            slow_stor[ihru] = availh2o

        # Compute flow to groundwater
        if slow_stor[ihru] > 0.0 and ssr2gw_rate[ihru] > 0.0:
            ssr_to_gw[ihru], slow_stor[ihru] = _compute_gwflow_numba(
                ssr2gw_rate[ihru],
                ssr2gw_exp[ihru],
                slow_stor[ihru],
            )

        # Compute contribution to Dunnian flow from PFR
        dunnianflw_gvr = 0.0
        if pref_flow_flag:
            if pref_flow_max[ihru] > 0.0:
                availh2o = pref_flow_stor[ihru] + topfr
                dunnianflw_gvr = max(0.0, availh2o - pref_flow_max[ihru])
                if dunnianflw_gvr > 0.0:
                    topfr = topfr - dunnianflw_gvr
                    if topfr < 0.0:
                        topfr = 0.0

                pref_flow_in[ihru] = pref_flow_infil[ihru] + topfr
                pref_flow_stor[ihru] = pref_flow_stor[ihru] + topfr

                if pref_flow_stor[ihru] > 0.0:
                    pref_flow_stor[ihru], prefflow = _compute_interflow_numba(
                        fastcoef_lin[ihru],
                        fastcoef_sq[ihru],
                        pref_flow_in[ihru],
                        pref_flow_stor[ihru],
                        prefflow,
                    )
        elif not (pref_flow_max[ihru] > 0.0):
            if hru_type[ihru] != 3:  # SWALE
                dunnianflw_gvr = topfr

        # Compute actual evapotranspiration
        pervactet = 0.0
        agactet = 0.0
        ag_hruactet = 0.0
        unsatisfied_ag_et = 0.0

        # Agricultural ET
        if ag_on_flag:
            if iter_aet_flag:
                ag_AETtarget = aet_external[ihru]
            else:
                ag_AETtarget = potet[ihru]

            ag_avail_targetAET = ag_AETtarget - hru_intcpevap[ihru]
            if ag_avail_targetAET < 0.0:
                ag_avail_targetAET = 0.0

            if ag_avail_targetAET > 0.0:
                # Set ag_soil_saturated flag if ag soil moisture ratio > 0.9999
                ag_et_type_gt_1 = (
                    transp_on[ihru] and ag_cov_type[ihru] > 0
                ) or snow_free[ihru] >= 0.01
                if ag_et_type_gt_1:
                    ag_pcts = ag_soil_moist[ihru] / ag_soil_moist_max[ihru]
                    if ag_pcts > 0.9999:
                        ag_soil_saturated[ihru] = 1

                # Call compute_szactet for ag area
                (
                    ag_soil_moist[ihru],
                    ag_soil_rechr[ihru],
                    _,
                    ag_potet_rechr[ihru],
                    ag_potet_lower[ihru],
                    agactet,
                ) = _compute_szactet_numba(
                    transp_on[ihru],
                    ag_cov_type[ihru],
                    ag_soil_type[ihru],
                    ag_soil_moist_max[ihru],
                    ag_soil_rechr_max[ihru],
                    snow_free[ihru],
                    ag_soil_moist[ihru],
                    ag_soil_rechr[ihru],
                    ag_avail_targetAET,
                    ag_potet_rechr[ihru],
                    ag_potet_lower[ihru],
                )

                ag_hruactet = agactet * agfrac

            unsatisfied_ag_et = ag_avail_targetAET - agactet
            unused_ag_et[ihru] = unsatisfied_ag_et

            # Add back canopy interception
            ag_actet[ihru] = agactet + hru_intcpevap[ihru]

        # Pervious ET
        avail_potet = potet[ihru] - hruactet - ag_hruactet
        if avail_potet < 0.0:
            avail_potet = 0.0

        if soil_moist[ihru] > 0.0 and avail_potet > 0.0:
            # Set soil_saturated flag if soil moisture ratio > 0.9999
            et_type_gt_1 = (
                transp_on[ihru] and cov_type[ihru] > 0
            ) or snow_free[ihru] >= 0.01
            if et_type_gt_1:
                pcts = soil_moist[ihru] / soil_moist_max[ihru]
                if pcts > 0.9999:
                    soil_saturated[ihru] = 1

            (
                soil_moist[ihru],
                soil_rechr[ihru],
                _,
                potet_rechr[ihru],
                potet_lower[ihru],
                pervactet,
            ) = _compute_szactet_numba(
                transp_on[ihru],
                cov_type[ihru],
                soil_type[ihru],
                soil_moist_max[ihru],
                soil_rechr_max[ihru],
                snow_free[ihru],
                soil_moist[ihru],
                soil_rechr[ihru],
                avail_potet,
                potet_rechr[ihru],
                potet_lower[ihru],
            )

        # Combine ET components
        perv_actet_hru[ihru] = pervactet * perv_frac
        hru_ag_actet[ihru] = ag_hruactet
        hru_actet[ihru] = hruactet + perv_actet_hru[ihru] + ag_hruactet
        perv_actet[ihru] = pervactet

        # Soil moisture accounting
        soil_lower[ihru] = soil_moist[ihru] - soil_rechr[ihru]

        # Compute interflow and excess flow
        if hru_type[ihru] != 3:  # SWALE
            dunnianflw = dunnianflw_gvr + dunnianflw_pfr
            dunnian_flow[ihru] = dunnianflw

            # Treat pref_flow as interflow
            ssres_flow[ihru] = slow_flow[ihru]
            if pref_flow_flag:
                if pref_flow_max[ihru] > 0.0:
                    pref_flow[ihru] = prefflow
                    ssres_flow[ihru] = ssres_flow[ihru] + prefflow

            # Treat dunnianflw as surface runoff
            sroff[ihru] = sroff[ihru] + dunnian_flow[ihru]
            ssres_stor[ihru] = slow_stor[ihru]
            if pref_flow_flag:
                ssres_stor[ihru] = ssres_stor[ihru] + pref_flow_stor[ihru]

        else:
            # Swale HRU
            availh2o = slow_stor[ihru] - sat_threshold[ihru]
            swale_actet[ihru] = 0.0
            if availh2o > 0.0:
                unsatisfied_et = potet[ihru] - hru_actet[ihru]
                if unsatisfied_et > 0.0:
                    availh2o = min(availh2o, unsatisfied_et)
                    swale_actet[ihru] = availh2o
                    hru_actet[ihru] = hru_actet[ihru] + swale_actet[ihru]
                    slow_stor[ihru] = slow_stor[ihru] - swale_actet[ihru]

            ssres_stor[ihru] = slow_stor[ihru]

        # Soil lower ratio
        if soil_lower_max[ihru] > 0.0:
            soil_lower_ratio[ihru] = soil_lower[ihru] / soil_lower_max[ihru]
            # Cap ratio at 1.0 if excess is due to rounding error (< 1e-4)
            if soil_lower_ratio[ihru] > 1.0:
                excess = soil_lower_ratio[ihru] - 1.0
                if excess < 1e-4:
                    soil_lower_ratio[ihru] = 1.0
                    soil_lower[ihru] = soil_lower_max[ihru]

        ssres_in[ihru] = soil_to_ssr[ihru]
        if pref_flow_flag:
            ssres_in[ihru] = ssres_in[ihru] + pref_flow_infil[ihru]

        unused_potet[ihru] = potet[ihru] - hru_actet[ihru]

        # Total soil moisture
        soil_moist_tot[ihru] = (
            ssres_stor[ihru]
            + soil_moist[ihru] * perv_frac
            + ag_soil_moist[ihru] * agfrac
        )

        # Recharge
        recharge[ihru] = soil_to_gw[ihru] + ssr_to_gw[ihru]
        recharge[ihru] = recharge[ihru] + dprst_seep_hru[ihru]

        # Agricultural soil lower
        if ag_on_flag:
            ag_soil_lower[ihru] = ag_soil_moist[ihru] - ag_soil_rechr[ihru]

            # Check for irrigation need
            if iter_aet_flag and transp_on[ihru]:
                if unsatisfied_ag_et > soilzone_aet_converge:
                    ag_soilwater_deficit[ihru] = (
                        ag_soil_moist_max[ihru] - ag_soil_moist[ihru]
                    ) / ag_soil_moist_max[ihru]

                    if (
                        ag_soilwater_deficit[ihru]
                        > ag_soilwater_deficit_min[ihru]
                    ):
                        unsatisfied_max = unsatisfied_ag_et
                        if unsatisfied_ag_et > ag_soil_moist_max[ihru]:
                            unsatisfied_max = ag_soil_moist_max[ihru]
                        else:
                            # Speed up convergence after 20 iterations
                            if soil_iter > 20:
                                unsatisfied_max = (
                                    unsatisfied_max + unsatisfied_ag_et
                                )
                            # Mark this HRU as needing irrigation (thread-safe)
                            hru_needs_irrigation[ihru] = 1

                        ag_irrigation_add[ihru] = (
                            ag_irrigation_add[ihru] + unsatisfied_max
                        )

                        # Store for reduction after loop (thread-safe)
                        hru_unsatisfied_max[ihru] = unsatisfied_max

    # End HRU loop

    # Compute iteration control variables after parallel loop (avoid race
    # conditions)
    add_estimated_irrigation = False
    num_hrus_ag_iter = 0
    unsatisfied_big = 0.0

    for ihru in range(nhru):
        if hru_needs_irrigation[ihru] > 0:
            add_estimated_irrigation = True
            num_hrus_ag_iter += 1
        if hru_unsatisfied_max[ihru] > unsatisfied_big:
            unsatisfied_big = hru_unsatisfied_max[ihru]

    # Calculate storage changes for mass budget
    pref_flow_stor_change[:] = pref_flow_stor - pref_flow_stor_prev
    soil_lower_change[:] = (
        soil_lower - soil_lower_prev - soil_lower_redistribution
    )
    soil_rechr_change[:] = (
        soil_rechr - soil_rechr_prev - soil_rechr_redistribution
    )
    slow_stor_change[:] = slow_stor - slow_stor_prev - slow_stor_redistribution

    # Convert to HRU basis (multiply by area fractions)
    soil_lower_change_hru[:] = soil_lower_change * hru_frac_perv
    soil_rechr_change_hru[:] = soil_rechr_change * hru_frac_perv

    # Agricultural soil moisture storage changes
    ag_soil_moist_change[:] = (
        ag_soil_moist - ag_soil_moist_prev - ag_soil_moist_redistribution
    )
    ag_soil_rechr_change[:] = (
        ag_soil_rechr - ag_soil_rechr_prev - ag_soil_rechr_redistribution
    )
    ag_soil_lower_change[:] = ag_soil_moist_change - ag_soil_rechr_change

    # Convert to HRU basis (multiply by area fractions)
    ag_soil_moist_change_hru[:] = ag_soil_moist_change * ag_frac
    ag_soil_rechr_change_hru[:] = ag_soil_rechr_change * ag_frac
    ag_soil_lower_change_hru[:] = ag_soil_lower_change * ag_frac

    # Return iteration control variables
    return (add_estimated_irrigation, num_hrus_ag_iter, unsatisfied_big)
