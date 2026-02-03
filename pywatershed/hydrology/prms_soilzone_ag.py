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
from ..parameters import Parameters
from .prms_soilzone_ag_obs_et import PRMSSoilzoneAgObsET


class PRMSSoilzoneAg(PRMSSoilzoneAgObsET):
    """PRMS soil zone with agricultural area (no observed ET iteration).

    This is a simplified version of PRMSSoilzoneAgObsET that does not require
    observed actual ET (aet_observed) or potential ET (pet_observed) inputs.
    It performs agricultural soil zone calculations without iterative AET
    matching.

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
        iter_aet_flag: Must be False or None for this class (no observed ET
            iteration)
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
        iter_aet_flag: Literal[False, None] = None,
        imbalance_behavior: Literal["defer", None, "warn", "error"] = "defer",
        calc_method: Literal["numpy", None] = None,
        adjust_parameters: Literal["warn", "error", "no"] = "warn",
        verbose: bool | None = None,
        restart_read: pl.Path | bool = False,
        restart_write: pl.Path | bool = False,
        restart_write_freq: Literal["y", "m", "d", "f", False] = False,
    ):
        # Ensure iter_aet_flag is not True
        if iter_aet_flag is True:
            raise ValueError(
                "PRMSSoilzoneAg does not support iter_aet_flag=True. "
                "Use PRMSSoilzoneAgObsET if you need observed ET iteration."
            )

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
            pet_observed=None,
            dprst_flag=dprst_flag,
            iter_aet_flag=iter_aet_flag,
            imbalance_behavior=imbalance_behavior,
            calc_method=calc_method,
            adjust_parameters=adjust_parameters,
            verbose=verbose,
            restart_read=restart_read,
            restart_write=restart_write,
            restart_write_freq=restart_write_freq,
        )

        self.name = "PRMSSoilzoneAg"

        return

    @staticmethod
    def get_inputs() -> tuple:
        """Return the input variable names required by this Process.

        Returns a tuple without aet_observed and pet_observed since this class
        does not require observed ET data.
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
