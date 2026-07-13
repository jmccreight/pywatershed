"""PRMS Soilzone with agricultural area (without observed ET iteration).

This module implements PRMSSoilzoneAg, a simplified version of
PRMSSoilzoneAgObsET that does not require observed AET/PET inputs and does
not perform iterative AET matching.

For the full implementation with iterative AET matching capabilities, see
PRMSSoilzoneAgObsET.
"""

import pathlib as pl
from typing import Literal

from ..base.adapter import adaptable
from ..base.control import Control
from ..constants import nan, zero
from ..parameters import Parameters
from .prms_soilzone_ag_obs_et import PRMSSoilzoneAgObsET


class PRMSSoilzoneAg(PRMSSoilzoneAgObsET):
    """PRMS soil zone with agricultural area (no observed ET iteration).

    This is a simplified version of PRMSSoilzoneAgObsET that does not require
    observed actual ET (aet_observed) inputs. It performs agricultural soil
    zone calculations without iterative AET matching.

    For cases where you have observed ET data and want to iteratively match it
    by adjusting irrigation, use PRMSSoilzoneAgObsET instead.

    Args:
        control: a Control object
        discretization: a discretization of class Parameters
        parameters: a parameter object of class Parameters
        dprst_evap_hru: Evaporation from depression storage on each HRU
        dprst_seep_hru: Seepage from depression storage to groundwater on each
            HRU
        hru_impervevap: HRU area-weighted average evaporation from
            impervious area for each HRU
        hru_intcpevap: HRU area-weighted average evaporation from the
            canopy for each HRU
        infil: Infiltration to the capillary reservoir for pervious area, depth
            on HRU area
        infil_ag: Infiltration to the capillary reservoir for agricultural
            area, depth on HRU area
        sroff: Surface runoff to the stream network for each HRU
        sroff_vol: Surface runoff volume to the stream network for each HRU
        potet: Potential ET for each HRU
        transp_on: Flag indicating whether transpiration is occurring
            (0=no; 1=yes)
        snow_evap: Evaporation and sublimation from snowpack on each HRU
        snowcov_area: Snow-covered area on each HRU prior to melt and
            sublimation unless snowpack
        ag_frac: Fraction of HRU area that is agricultural/irrigated
        dprst_flag: Boolean flag to enable depression storage. Default is True.

        imbalance_behavior: one of ["defer", None, "warn", "error"] with
            "defer" being the default and deferring to
            control.options["imbalance_behavior"] when available.
        calc_method: one of ["numpy", "numba"]. None defaults to "numba".
        adjust_parameters: one of ["warn", "error", "no"]. Default is "warn".
        verbose: Print extra information or not?
        restart_read: May be boolean or a Pathlib.Path. See base class docs.
        restart_write: May be boolean or a Pathlib.Path. See base class docs.
        restart_write_freq: Frequency of restart file writing. See base class
            docs.
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
        dprst_flag: bool | None = None,
        imbalance_behavior: Literal["defer", None, "warn", "error"] = "defer",
        calc_method: Literal["numpy", None] = None,
        adjust_parameters: Literal["warn", "error", "no"] = "warn",
        input_aliases: dict = None,
        verbose: bool | None = None,
        restart_read: pl.Path | bool = False,
        restart_write: pl.Path | bool = False,
        restart_write_freq: Literal["y", "m", "d", "f", False] = False,
    ):

        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
            dprst_evap_hru=dprst_evap_hru,
            dprst_seep_hru=dprst_seep_hru,
            hru_impervevap=hru_impervevap,
            hru_intcpevap=hru_intcpevap,
            infil=infil,
            infil_ag=infil_ag,
            sroff=sroff,
            sroff_vol=sroff_vol,
            potet=potet,
            transp_on=transp_on,
            snow_evap=snow_evap,
            snowcov_area=snowcov_area,
            ag_frac=ag_frac,
            aet_observed=None,
            dprst_flag=dprst_flag,
            imbalance_behavior=imbalance_behavior,
            calc_method=calc_method,
            adjust_parameters=adjust_parameters,
            input_aliases=input_aliases,
            verbose=verbose,
            restart_read=restart_read,
            restart_write=restart_write,
            restart_write_freq=restart_write_freq,
        )

        self.name = "PRMSSoilzoneAg"

        self._iter_aet_flag = False
        if (
            "iter_aet_flag" in self.control.options
            and self.control.options["iter_aet_flag"]
        ):
            raise ValueError(
                "PRMSSoilzoneAg does not support iter_aet_flag=True. "
                "Use PRMSSoilzoneAgObsET if you need observed ET iteration."
            )

        return

    @staticmethod
    def get_inputs() -> tuple:
        """Return the input variable names required by this Process.

        Returns a tuple without aet_observed since this class does not require
        observed ET data.
        """
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
        )

    @staticmethod
    def get_variables() -> tuple:
        """Return the variable names output by this Process.

        Excludes AET_external and related variables that are only used in
        PRMSSoilzoneAgObsET (the observed ET iteration version).
        """
        return (
            # Mass budget input terms (whole HRU basis)
            "perv_infil_hru",
            "ag_infil_hru",
            # Pervious area variables (whole HRU basis)
            "cap_infil_tot",
            "cap_waterin",
            "dunnian_flow",
            "hru_actet",
            "perv_actet",
            "perv_actet_hru",
            "potet_lower",
            "potet_rechr",
            "pref_flow",
            "pref_flow_in",
            "pref_flow_infil",
            "pref_flow_max",
            "pref_flow_stor",
            "pref_flow_stor_change",
            "pref_flow_stor_prev",
            "pref_flow_thrsh",
            "recharge",
            "slow_flow",
            "slow_stor",
            "slow_stor_change",
            "slow_stor_prev",
            "soil_lower",
            "soil_lower_change",
            "soil_lower_change_hru",
            "soil_lower_prev",
            "soil_lower_ratio",
            "soil_lower_max",
            "soil_moist",
            "soil_moist_tot",
            "soil_rechr",
            "soil_rechr_change",
            "soil_rechr_change_hru",
            "soil_rechr_prev",
            "soil_saturated",
            "soil_to_gw",
            "soil_to_ssr",
            "soil_zone_max",
            "ssr_to_gw",
            "ssres_flow",
            "ssres_flow_vol",
            "ssres_in",
            "ssres_stor",
            "swale_actet",
            "unused_potet",
            "perv_soil_to_gw",
            "perv_soil_to_gvr",
            # Agricultural area variables (whole HRU basis)
            "ag_cap_infil_tot",
            "ag_soil_moist",
            "ag_soil_moist_prev",
            "ag_soil_moist_change",
            "ag_soil_moist_change_hru",
            "ag_soil_rechr",
            "ag_soil_rechr_prev",
            "ag_soil_rechr_change",
            "ag_soil_rechr_change_hru",
            "ag_soil_rechr_max",
            "ag_soil_lower",
            "ag_soil_lower_change",
            "ag_soil_lower_change_hru",
            "ag_soil_lower_stor_max",
            "ag_actet",
            "hru_ag_actet",
            "ag_potet_rechr",
            "ag_potet_lower",
            "ag_soil_to_gw",
            "ag_soil_to_gvr",
            "ag_hortonian",
            "ag_soil_saturated",
            "unused_ag_et",
            "ag_soilwater_deficit",
            # Redistribution tracking (for ag_frac changes)
            "ag_soil_moist_redistribution",
            "ag_soil_rechr_redistribution",
            "soil_rechr_redistribution",
            "soil_lower_redistribution",
            "slow_stor_redistribution",
        )

    @staticmethod
    def get_mass_budget_terms() -> dict:
        """Return mass budget terms for PRMSSoilzoneAg."""
        return {
            "inputs": [
                "perv_infil_hru",
                "ag_infil_hru",
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
        # Redistribution tracking (for ag_frac changes)
        "ag_soil_moist_redistribution": zero,
        "ag_soil_rechr_redistribution": zero,
        "soil_rechr_redistribution": zero,
        "soil_lower_redistribution": zero,
        "slow_stor_redistribution": zero,
    }
