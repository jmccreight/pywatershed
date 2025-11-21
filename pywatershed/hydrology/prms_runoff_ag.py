"""PRMS Surface Runoff with Agricultural Area Infiltration.

This module implements PRMSRunoffAg, which extends PRMSRunoff to compute
separate infiltration for agricultural areas in addition to pervious areas.

Fortran Source Reference
------------------------
Based on GSFLOW 2.4.0 srunoff.f90 with the following additions:

**Implemented:**
- Separate infiltration calculation for agricultural areas (infil_ag)
- Agricultural contributing area computation (ag_comp)
- Agricultural infiltration capacity check (check_capacity_ag)
- Parallel processing of pervious and agricultural areas

**Key Fortran Cross-References:**
- Main loop with AG_flag: Lines ~730-982 in srunoff()
- compute_infil_ag_glcr(): Lines ~1120-1241
- ag_comp() subroutine: Lines ~1284-1321
- check_capacity_ag(): Lines ~1409-1430

Implementation Notes
--------------------
- Agricultural infiltration is computed in parallel with pervious infiltration
- Uses antecedent ag_soil_moist or ag_soil_rechr to compute contributing area
- Agricultural area fraction (ag_frac) determines which calculations apply
- When ag_frac=0, behaves identically to base PRMSRunoff

"""

import pathlib as pl
from typing import Literal, Union

import numpy as np

from ..base.adapter import adaptable
from ..base.control import Control
from ..constants import HruType, nearzero, zero
from ..parameters import Parameters
from .prms_runoff import PRMSRunoff

RAIN = 0
SNOW = 1

BARESOIL = 0
GRASSES = 1

OFF = 0
ACTIVE = 1

LAND = HruType.LAND.value
LAKE = HruType.LAKE.value

dnearzero = 1.0e-15


class PRMSRunoffAg(PRMSRunoff):
    """PRMS surface runoff with agricultural area infiltration.

    A surface runoff representation from PRMS that computes separate
    infiltration for agricultural areas.

    Implementation based on GSFLOW 2.4.0 with theoretical documentation given
    in the PRMS-IV documentation:

    `Markstrom, S. L., Regan, R. S., Hay, L. E., Viger, R. J., Webb, R. M.,
    Payn, R. A., & LaFontaine, J. H. (2015). PRMS-IV, the
    precipitation-runoff modeling system, version 4. US Geological Survey
    Techniques and Methods, 6, B7.
    <https://pubs.usgs.gov/tm/6b7/pdf/tm6-b7.pdf>`__

    Args:
        control: a Control object
        discretization: a discretization of class Parameters
        parameters: a parameter object of class Parameters
        soil_lower_prev: Previous storage of lower reservoir for each HRU
        soil_rechr_prev: Previous storage of recharge reservoir for each HRU
        ag_soil_moist_prev: Previous ag soil moisture for each HRU
        ag_soil_rechr_prev: Previous ag recharge reservoir for each HRU
        ag_frac: Fraction of HRU that is agricultural/irrigated
        net_ppt: Precipitation (rain and/or snow) that falls through the
            canopy for each HRU
        net_rain: Rain that falls through canopy for each HRU
        net_snow: Snow that falls through canopy for each HRU
        potet: Potential ET for each HRU
        snowmelt: Snowmelt from snowpack on each HRU
        snow_evap: Evaporation and sublimation from snowpack on each HRU
        pkwater_equiv: Snowpack water equivalent on each HRU
        pptmix_nopack: Flag indicating that a mixed precipitation event has
            occurred with no snowpack
        snowcov_area: Snow-covered area on each HRU prior to melt and
            sublimation unless snowpack
        through_rain: Rain that passes through snow when no snow present
        hru_intcpevap: HRU area-weighted average evaporation from the
            canopy for each HRU
        intcp_changeover: Canopy throughfall caused by canopy density
            change from winter to summer
        dprst_flag: use depression storage or not? None uses value in control
            file, which otherwise defaults to True.
        imbalance_behavior: one of ["defer", None, "warn", "error"]
            with "defer" being the default and defering to
            control.options["imbalance_behavior"] when available. When
            control.options["imbalance_behavior"] is not avaiable,
            imbalance_behavior is set to "warn".
        calc_method: one of ["fortran", "numba", "numpy"]. None defaults to
            "numba".
        verbose: Print extra information or not?
        restart_read: Path or boolean for restart file reading
        restart_write: Path or boolean for restart file writing
        restart_write_freq: Frequency for restart file writing
    """

    def __init__(
        self,
        control: Control,
        discretization: Parameters,
        parameters: Parameters,
        soil_lower_prev: adaptable,
        soil_rechr_prev: adaptable,
        net_ppt: adaptable,
        net_rain: adaptable,
        net_snow: adaptable,
        potet: adaptable,
        snowmelt: adaptable,
        snow_evap: adaptable,
        pkwater_equiv: adaptable,
        pptmix_nopack: adaptable,
        snowcov_area: adaptable,
        through_rain: adaptable,
        hru_intcpevap: adaptable,
        intcp_changeover: adaptable,
        ag_soil_moist_prev: adaptable,
        ag_soil_rechr_prev: adaptable,
        dprst_flag: Union[bool, None] = None,
        imbalance_behavior: Literal["defer", None, "warn", "error"] = "defer",
        calc_method: Literal["numba", "numpy", None] = None,
        verbose: Union[bool, None] = None,
        restart_read: Union[pl.Path, bool] = False,
        restart_write: Union[pl.Path, bool] = False,
        restart_write_freq: Literal["y", "m", "d", "f", False] = False,
    ) -> None:
        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
            soil_lower_prev=soil_lower_prev,
            soil_rechr_prev=soil_rechr_prev,
            net_ppt=net_ppt,
            net_rain=net_rain,
            net_snow=net_snow,
            potet=potet,
            snowmelt=snowmelt,
            snow_evap=snow_evap,
            pkwater_equiv=pkwater_equiv,
            pptmix_nopack=pptmix_nopack,
            snowcov_area=snowcov_area,
            through_rain=through_rain,
            hru_intcpevap=hru_intcpevap,
            intcp_changeover=intcp_changeover,
            ag_soil_moist_prev=ag_soil_moist_prev,
            ag_soil_rechr_prev=ag_soil_rechr_prev,
            dprst_flag=dprst_flag,
            imbalance_behavior=imbalance_behavior,
            calc_method=calc_method,
            verbose=verbose,
            restart_read=restart_read,
            restart_write=restart_write,
            restart_write_freq=restart_write_freq,
        )

        self.name = "PRMSRunoffAg"

        self._set_inputs(locals())
        self._set_options(locals())

        if self._dprst_flag is None:
            self._dprst_flag = True

        # self._set_budget()
        # self._init_calc_method()

        # Rename? or move contents here?
        self.basin_init()

        if self._dprst_flag:
            self.dprst_init()

        if restart_read is not False or restart_write is not False:
            self.restart_read = False
            self.restart_write = False

        return

    @staticmethod
    def get_dimensions() -> tuple:
        return ("nhru",)

    @staticmethod
    def get_parameters() -> tuple:
        return (
            "hru_type",
            "hru_area",
            "hru_in_to_cf",
            "hru_percent_imperv",
            "imperv_stor_max",
            "carea_max",
            "smidx_coef",
            "smidx_exp",
            "soil_moist_max",
            "snowinfil_max",
            "dprst_depth_avg",
            "dprst_et_coef",
            "dprst_flow_coef",
            "dprst_frac",  # necessary?
            "dprst_frac_init",
            "dprst_frac_open",
            "dprst_seep_rate_clos",
            "dprst_seep_rate_open",
            "sro_to_dprst_imperv",
            "sro_to_dprst_perv",
            "va_open_exp",
            "va_clos_exp",
            "op_flow_thres",
            # soil_rechr_max",  # necessary?
            # Agricultural parameters
            "ag_frac",
            "ag_soil_moist_max",
            "ag_soil_rechr_max_frac",
        )

    @staticmethod
    def get_inputs() -> tuple:
        return (
            "soil_lower_prev",
            "soil_rechr_prev",
            "net_rain",
            "net_ppt",
            "net_snow",
            "potet",
            "snowmelt",
            "snow_evap",
            "pkwater_equiv",
            "pptmix_nopack",
            "snowcov_area",
            "through_rain",
            "hru_intcpevap",
            "intcp_changeover",
            "ag_soil_moist_prev",
            "ag_soil_rechr_prev",
        )

    @staticmethod
    def get_init_values() -> dict:
        return {
            "contrib_fraction": zero,
            "infil": zero,
            "infil_hru": zero,
            "sroff": zero,
            "sroff_vol": zero,
            "hru_sroffp": zero,
            "hru_sroffi": zero,
            "imperv_stor": zero,
            "imperv_evap": zero,
            "hru_impervevap": zero,
            "hru_impervstor": zero,
            "hru_impervstor_old": zero,
            "hru_impervstor_change": zero,
            "dprst_vol_frac": zero,
            "dprst_vol_clos": zero,
            "dprst_vol_open": zero,
            "dprst_vol_clos_frac": zero,
            "dprst_vol_open_frac": zero,
            "dprst_area_clos": zero,
            "dprst_area_open": zero,
            "dprst_area_clos_max": zero,
            "dprst_area_open_max": zero,
            "dprst_sroff_hru": zero,
            "dprst_seep_hru": zero,
            "dprst_evap_hru": zero,
            "dprst_insroff_hru": zero,
            "dprst_stor_hru": zero,
            "dprst_stor_hru_old": zero,
            "dprst_stor_hru_change": zero,
            "dprst_vol_thres_open": zero,
            "infil_ag": zero,
            "ag_contrib_fraction": zero,
        }

    @staticmethod
    def get_restart_variables() -> list:
        raise NotImplementedError(
            "Restart capability not implemented for PRMSRunoffAg"
        )

    @staticmethod
    def get_mass_budget_terms():
        return {
            "inputs": [
                "through_rain",
                "snowmelt",
                "intcp_changeover",
            ],
            "outputs": [
                "hru_sroffi",
                "hru_sroffp",
                "dprst_sroff_hru",
                "infil_hru",
                "hru_impervevap",
                "dprst_seep_hru",
                "dprst_evap_hru",
                "infil_ag",
            ],
            "storage_changes": [
                "hru_impervstor_change",
                "dprst_stor_hru_change",
            ],
        }

    def basin_init(self):
        super().basin_init()
        # _sroff_ag  is private, apparently not needed by other processes
        self._sroff_ag = np.zeros(self.nhru, dtype=float)
        self._ag_area = self.ag_frac * self.hru_area
        self._ag_soil_rechr_max = (
            self.ag_soil_moist_max * self.ag_soil_rechr_max_frac
        )

        return

    def _advance_variables(self) -> None:
        """Advance variables from previous timestep."""
        self.hru_impervstor_old[:] = self.hru_impervstor
        self.dprst_stor_hru_old[:] = self.dprst_stor_hru
        return None

    def _calculate(self, time_length, vectorized=False):
        """Perform the core calculations with agricultural infiltration.

        Fortran reference: srunoff() main loop lines ~730-982
        """
        # Call parent's calculation routine but with ag-specific compute_infil
        (
            self.infil[:],
            self.contrib_fraction[:],
            self.hru_sroffp[:],
            self.hru_sroffi[:],
            self.imperv_evap[:],
            self.hru_impervevap[:],
            self.imperv_stor[:],
            self.dprst_vol_open[:],
            self.dprst_area_open[:],
            self.dprst_vol_clos[:],
            self.dprst_area_clos[:],
            self.dprst_vol_open_frac[:],
            self.dprst_vol_clos_frac[:],
            self.dprst_vol_frac[:],
            self.dprst_insroff_hru[:],
            self.dprst_evap_hru[:],
            self.dprst_seep_hru[:],
            self.dprst_sroff_hru[:],
            self.sroff[:],
            self.infil_ag[:],
            self.ag_contrib_fraction[:],
        ) = self._calculate_runoff_ag(
            infil=self.infil,
            infil_ag=self.infil_ag,
            ag_contrib_fraction=self.ag_contrib_fraction,
            nhru=self.nhru,
            hru_area=self.hru_area,
            hru_perv=self.hru_perv,
            hru_frac_perv=self.hru_frac_perv,
            ag_frac=self.ag_frac,
            ag_area=self._ag_area,
            hru_sroffp=self.hru_sroffp,
            contrib_fraction=self.contrib_fraction,
            hru_percent_imperv=self.hru_percent_imperv,
            hru_sroffi=self.hru_sroffi,
            imperv_evap=self.imperv_evap,
            hru_imperv=self.hru_imperv,
            hru_impervevap=self.hru_impervevap,
            potet=self.potet,
            snow_evap=self.snow_evap,
            hru_intcpevap=self.hru_intcpevap,
            soil_lower_prev=self.soil_lower_prev,
            soil_rechr_prev=self.soil_rechr_prev,
            ag_soil_moist_prev=self.ag_soil_moist_prev,
            ag_soil_rechr_prev=self.ag_soil_rechr_prev,
            soil_moist_max=self.soil_moist_max,
            # soil_rechr_max=self.soil_rechr_max,
            ag_soil_moist_max=self.ag_soil_moist_max,
            ag_soil_rechr_max=self._ag_soil_rechr_max,
            carea_max=self.carea_max,
            smidx_coef=self.smidx_coef,
            smidx_exp=self.smidx_exp,
            pptmix_nopack=self.pptmix_nopack,
            net_rain=self.net_rain,
            net_ppt=self.net_ppt,
            imperv_stor=self.imperv_stor,
            imperv_stor_max=self.imperv_stor_max,
            snowmelt=self.snowmelt,
            snowinfil_max=self.snowinfil_max,
            net_snow=self.net_snow,
            pkwater_equiv=self.pkwater_equiv,
            hru_type=self.hru_type,
            intcp_changeover=self.intcp_changeover,
            dprst_in=self.dprst_in,
            dprst_seep_hru=self.dprst_seep_hru,
            dprst_area_max=self.dprst_area_max,
            dprst_vol_open=self.dprst_vol_open,
            dprst_vol_clos=self.dprst_vol_clos,
            dprst_sroff_hru=self.dprst_sroff_hru,
            dprst_evap_hru=self.dprst_evap_hru,
            dprst_insroff_hru=self.dprst_insroff_hru,
            dprst_vol_open_frac=self.dprst_vol_open_frac,
            dprst_vol_clos_frac=self.dprst_vol_clos_frac,
            dprst_vol_frac=self.dprst_vol_frac,
            dprst_stor_hru=self.dprst_stor_hru,
            dprst_area_clos_max=self.dprst_area_clos_max,
            dprst_area_clos=self.dprst_area_clos,
            dprst_vol_open_max=self.dprst_vol_open_max,
            dprst_area_open_max=self.dprst_area_open_max,
            dprst_area_open=self.dprst_area_open,
            sro_to_dprst_perv=self.sro_to_dprst_perv,
            sro_to_dprst_imperv=self.sro_to_dprst_imperv,
            dprst_frac_open=self.dprst_frac_open,
            dprst_frac_clos=self.dprst_frac_clos,
            va_open_exp=self.va_open_exp,
            dprst_vol_clos_max=self.dprst_vol_clos_max,
            va_clos_exp=self.va_clos_exp,
            snowcov_area=self.snowcov_area,
            dprst_et_coef=self.dprst_et_coef,
            dprst_seep_rate_open=self.dprst_seep_rate_open,
            dprst_vol_thres_open=self.dprst_vol_thres_open,
            dprst_flow_coef=self.dprst_flow_coef,
            dprst_seep_rate_clos=self.dprst_seep_rate_clos,
            sroff=self.sroff,
            hru_impervstor=self.hru_impervstor,
            check_capacity=self.check_capacity,
            check_capacity_ag=self.check_capacity_ag,
            perv_comp=self.perv_comp,
            ag_comp=self.ag_comp,
            compute_infil_ag_glcr=self.compute_infil_ag_glcr,
            dprst_comp=self.dprst_comp,
            imperv_et=self.imperv_et,
            through_rain=self.through_rain,
            dprst_flag=self._dprst_flag,
        )

        self.infil_hru[:] = self.infil * self.hru_frac_perv
        # infil_ag is already on HRU basis (inches over whole HRU)

        self.hru_impervstor_change[:] = (
            self.hru_impervstor - self.hru_impervstor_old
        )
        self.dprst_stor_hru_change[:] = (
            self.dprst_stor_hru - self.dprst_stor_hru_old
        )

        self.sroff_vol[:] = self.sroff * self.hru_in_to_cf

        return

    @staticmethod
    def _calculate_runoff_ag(
        infil,
        nhru,
        hru_area,
        hru_perv,
        hru_frac_perv,
        hru_sroffp,
        contrib_fraction,
        hru_percent_imperv,
        hru_sroffi,
        imperv_evap,
        hru_imperv,
        hru_impervevap,
        potet,
        snow_evap,
        hru_intcpevap,
        soil_lower_prev,
        soil_rechr_prev,
        soil_moist_max,
        carea_max,
        smidx_coef,
        smidx_exp,
        pptmix_nopack,
        net_rain,
        net_ppt,
        imperv_stor,
        imperv_stor_max,
        snowmelt,
        snowinfil_max,
        net_snow,
        pkwater_equiv,
        hru_type,
        intcp_changeover,
        dprst_in,
        dprst_seep_hru,
        dprst_area_max,
        dprst_vol_open,
        dprst_vol_clos,
        dprst_sroff_hru,
        dprst_evap_hru,
        dprst_insroff_hru,
        dprst_vol_open_frac,
        dprst_vol_clos_frac,
        dprst_vol_frac,
        dprst_stor_hru,
        dprst_area_clos_max,
        dprst_area_clos,
        dprst_vol_open_max,
        dprst_area_open_max,
        dprst_area_open,
        sro_to_dprst_perv,
        sro_to_dprst_imperv,
        dprst_frac_open,
        dprst_frac_clos,
        va_open_exp,
        dprst_vol_clos_max,
        va_clos_exp,
        snowcov_area,
        dprst_et_coef,
        dprst_seep_rate_open,
        dprst_vol_thres_open,
        dprst_flow_coef,
        dprst_seep_rate_clos,
        sroff,
        hru_impervstor,
        # functions at end
        check_capacity_ag,  # vs check_capacity
        perv_comp,
        compute_infil_ag_glcr,  # vs compute_infil
        dprst_comp,
        ag_comp,  # new
        imperv_et,
        through_rain,
        dprst_flag,
        infil_ag,  # <- all new ag vars start here
        ag_contrib_fraction,
        ag_frac,
        ag_area,
        ag_soil_moist_prev,
        ag_soil_rechr_prev,
        soil_rechr_max,
        ag_soil_moist_max,
        ag_soil_rechr_max,
    ):
        """Calculate runoff with agricultural infiltration.

        This is a modified version of PRMSRunoff._calculate_runoff that
        calls compute_infil_ag_glcr instead of compute_infil.

        Fortran reference: srunoff() lines ~730-982
        """
        # Zero out arrays
        infil[:] = zero
        infil_ag[:] = zero
        hru_sroffp[:] = zero
        hru_sroffi[:] = zero
        contrib_fraction[:] = zero
        ag_contrib_fraction[:] = zero

        # HRU loop
        for ihru in range(nhru):
            # Fortran: perv_on = ACTIVE
            perv_on = hru_perv[ihru] > 0.0
            # Fortran: ag_on = ACTIVE
            ag_on = ag_area[ihru] > 0.0

            avail_et = potet[ihru] - snow_evap[ihru] - hru_intcpevap[ihru]
            if avail_et < 0.0:
                avail_et = 0.0

            # Initialize runoff components
            sri = 0.0
            srp = 0.0
            sroff_ag = 0.0

            # Compute infiltration and surface runoff for pervious and ag areas
            # Fortran: CALL compute_infil_ag_glcr
            (
                sri,
                srp,
                sroff_ag,
                imperv_stor[ihru],
                infil[ihru],
                infil_ag[ihru],
                contrib_fraction[ihru],
                ag_contrib_fraction[ihru],
            ) = compute_infil_ag_glcr(
                soil_lower_prev=soil_lower_prev[ihru],
                soil_rechr_prev=soil_rechr_prev[ihru],
                ag_soil_moist_prev=ag_soil_moist_prev[ihru],
                ag_soil_rechr_prev=ag_soil_rechr_prev[ihru],
                soil_moist_max=soil_moist_max[ihru],
                ag_soil_moist_max=ag_soil_moist_max[ihru],
                ag_soil_rechr_max=ag_soil_rechr_max[ihru],
                carea_max=carea_max[ihru],
                smidx_coef=smidx_coef[ihru],
                smidx_exp=smidx_exp[ihru],
                pptmix_nopack=pptmix_nopack[ihru],
                net_rain=net_rain[ihru],
                net_ppt=net_ppt[ihru],
                imperv_stor=imperv_stor[ihru],
                imperv_stor_max=imperv_stor_max[ihru],
                snowmelt=snowmelt[ihru],
                snowinfil_max=snowinfil_max[ihru],
                net_snow=net_snow[ihru],
                pkwater_equiv=pkwater_equiv[ihru],
                infil=infil[ihru],
                infil_ag=infil_ag[ihru],
                hru_type=hru_type[ihru],
                intcp_changeover=intcp_changeover[ihru],
                hruarea_imperv=hru_imperv[ihru],
                sri=sri,
                srp=srp,
                sroff_ag=sroff_ag,
                contrib_fraction=contrib_fraction[ihru],
                ag_contrib_fraction=ag_contrib_fraction[ihru],
                check_capacity=check_capacity,
                check_capacity_ag=check_capacity_ag,
                perv_comp=perv_comp,
                ag_comp=ag_comp,
                through_rain=through_rain[ihru],
                perv_on=perv_on,
                ag_on=ag_on,
            )

            hru_sroffi[ihru] = sri
            hru_sroffp[ihru] = srp + sroff_ag

            # Compute evaporation from impervious area
            if hru_imperv[ihru] > 0.0:
                imperv_stor[ihru], imperv_evap[ihru] = imperv_et(
                    imperv_stor=imperv_stor[ihru],
                    potet=potet[ihru],
                    imperv_evap=imperv_evap[ihru],
                    sca=snowcov_area[ihru],
                    avail_et=avail_et,
                    imperv_frac=hru_percent_imperv[ihru],
                )
                hru_impervevap[ihru] = (
                    imperv_evap[ihru] * hru_percent_imperv[ihru]
                )

            hru_impervstor[ihru] = imperv_stor[ihru] * hru_imperv[ihru]

        # Depression storage computations
        if dprst_flag:
            for ihru in range(nhru):
                # Only call dprst_comp if there's a depression storage area
                if dprst_area_max[ihru] > 0.0:
                    (
                        dprst_in[ihru],
                        dprst_vol_open[ihru],
                        dprst_area_open[ihru],
                        avail_et,
                        dprst_vol_clos[ihru],
                        dprst_sroff_hru[ihru],
                        srp,
                        sri,
                        dprst_evap_hru[ihru],
                        dprst_seep_hru[ihru],
                        dprst_insroff_hru[ihru],
                        dprst_vol_open_frac[ihru],
                        dprst_vol_clos_frac[ihru],
                        dprst_vol_frac[ihru],
                        dprst_stor_hru[ihru],
                    ) = dprst_comp(
                        dprst_vol_clos=dprst_vol_clos[ihru],
                        dprst_area_clos_max=dprst_area_clos_max[ihru],
                        dprst_area_clos=dprst_area_clos[ihru],
                        dprst_vol_open_max=dprst_vol_open_max[ihru],
                        dprst_vol_open=dprst_vol_open[ihru],
                        dprst_area_open_max=dprst_area_open_max[ihru],
                        dprst_sroff_hru=dprst_sroff_hru[ihru],
                        sro_to_dprst_perv=sro_to_dprst_perv[ihru],
                        sro_to_dprst_imperv=sro_to_dprst_imperv[ihru],
                        dprst_evap_hru=dprst_evap_hru[ihru],
                        pptmix_nopack=pptmix_nopack[ihru],
                        snowmelt=snowmelt[ihru],
                        pkwater_equiv=pkwater_equiv[ihru],
                        net_snow=net_snow[ihru],
                        hru_area=hru_area[ihru],
                        dprst_insroff_hru=dprst_insroff_hru[ihru],
                        dprst_frac_open=dprst_frac_open[ihru],
                        dprst_frac_clos=dprst_frac_clos[ihru],
                        va_open_exp=va_open_exp[ihru],
                        dprst_vol_clos_max=dprst_vol_clos_max[ihru],
                        dprst_vol_clos_frac=dprst_vol_clos_frac[ihru],
                        va_clos_exp=va_clos_exp[ihru],
                        potet=potet[ihru],
                        snowcov_area=snowcov_area[ihru],
                        dprst_et_coef=dprst_et_coef[ihru],
                        dprst_seep_rate_open=dprst_seep_rate_open[ihru],
                        dprst_vol_thres_open=dprst_vol_thres_open[ihru],
                        dprst_flow_coef=dprst_flow_coef[ihru],
                        dprst_seep_rate_clos=dprst_seep_rate_clos[ihru],
                        avail_et=avail_et,
                        net_rain=net_rain[ihru],
                        dprst_in=dprst_in[ihru],
                        srp=srp,
                        sri=sri,
                        imperv_frac=hru_percent_imperv[ihru],
                        perv_frac=hru_frac_perv[ihru],
                    )

        # Combine surface runoff components
        for ihru in range(nhru):
            sroff[ihru] = hru_sroffp[ihru] + hru_sroffi[ihru]
            if dprst_flag:
                sroff[ihru] = sroff[ihru] + dprst_sroff_hru[ihru]

        return (
            infil,
            contrib_fraction,
            hru_sroffp,
            hru_sroffi,
            imperv_evap,
            hru_impervevap,
            imperv_stor,
            dprst_vol_open,
            dprst_area_open,
            dprst_vol_clos,
            dprst_area_clos,
            dprst_vol_open_frac,
            dprst_vol_clos_frac,
            dprst_vol_frac,
            dprst_insroff_hru,
            dprst_evap_hru,
            dprst_seep_hru,
            dprst_sroff_hru,
            sroff,
            infil_ag,
            ag_contrib_fraction,
        )

    @staticmethod
    def compute_infil_ag_glcr(
        contrib_fraction,
        soil_moist_prev,
        soil_moist_max,
        carea_max,
        smidx_coef,
        smidx_exp,
        pptmix_nopack,
        net_rain,
        net_ppt,
        imperv_stor,
        imperv_stor_max,
        snowmelt,
        snowinfil_max,
        net_snow,
        pkwater_equiv,
        infil,
        hru_type,
        intcp_changeover,
        hruarea_imperv,
        sri,
        srp,
        check_capacity,
        perv_comp,
        through_rain,
        # Agricultural-specific parameters
        soil_rechr_prev,
        ag_soil_moist_prev,
        ag_soil_rechr_prev,
        ag_soil_moist_max,
        ag_soil_rechr_max,
        infil_ag,
        sroff_ag,
        ag_contrib_fraction,
        check_capacity_ag,
        ag_comp,
        perv_on,
        ag_on,
    ):
        """Compute infiltration for both pervious and agricultural areas.

        Fortran reference: compute_infil_ag_glcr() subroutine lines ~1120-1241

        This is a modified version of compute_infil that handles both
        pervious and agricultural areas in parallel.
        """
        # TODO: missing net_apply logic
        # fmt: off
        # ! irrigation application for pervious and agriculture areas (just like infiltration)
        #       IF ( Net_apply>0.0 ) THEN
        #         avail_water = Net_apply * glacier_free
        #         IF ( Perv_on==ACTIVE ) Infil = avail_water
        #         IF ( Ag_on==ACTIVE ) Infil_ag = avail_water
        #         IF ( hru_flag==1 ) THEN
        #           IF ( Perv_on==ACTIVE ) CALL perv_comp(avail_water, avail_water, Infil, Sra)
        #           IF ( Ag_on==ACTIVE ) CALL ag_comp(avail_water, avail_water, Infil_ag, Sroff_ag)
        # !          apply_sroff = Sra + Sroff_ag ! may want apply_sroff be a declared variable
        #         ENDIF
        #       ENDIF
        # fmt: on

        isglacier = False
        hru_flag = 0
        if hru_type == LAND or isglacier:
            hru_flag = 1

        avail_water = 0.0

        # Compute runoff from canopy changeover water
        # Fortran: lines ~1151-1156
        if intcp_changeover > 0.0:
            avail_water = avail_water + intcp_changeover
            if perv_on:
                infil = infil + intcp_changeover
            if ag_on:
                infil_ag = infil_ag + intcp_changeover
            if hru_flag == 1:
                if perv_on:
                    infil, srp, contrib_fraction = perv_comp(
                        soil_moist_prev=soil_lower_prev,
                        carea_max=carea_max,
                        smidx_coef=smidx_coef,
                        smidx_exp=smidx_exp,
                        pptp=intcp_changeover,
                        ptc=intcp_changeover,
                        infil=infil,
                        srp=srp,
                    )
                if ag_on:
                    infil_ag, sroff_ag, ag_contrib_fraction = ag_comp(
                        ag_soil_moist_prev=ag_soil_moist_prev,
                        ag_soil_rechr_prev=ag_soil_rechr_prev,
                        ag_soil_rechr_max=ag_soil_rechr_max,
                        carea_max=carea_max,
                        smidx_coef=smidx_coef,
                        smidx_exp=smidx_exp,
                        pptp=intcp_changeover,
                        ptc=intcp_changeover,
                        infil_ag=infil_ag,
                        sroff_ag=sroff_ag,
                    )

        # If rain/snow event with no antecedent snowpack
        # Fortran: lines ~1158-1167
        cond2 = pptmix_nopack != 0
        if cond2:
            avail_water = avail_water + through_rain
            if perv_on:
                infil = infil + through_rain
            if ag_on:
                infil_ag = infil_ag + through_rain
            if hru_flag == 1:
                if perv_on:
                    infil, srp, contrib_fraction = perv_comp(
                        soil_moist_prev=soil_lower_prev,
                        carea_max=carea_max,
                        smidx_coef=smidx_coef,
                        smidx_exp=smidx_exp,
                        pptp=through_rain,
                        ptc=through_rain,
                        infil=infil,
                        srp=srp,
                    )
                if ag_on:
                    infil_ag, sroff_ag, ag_contrib_fraction = ag_comp(
                        ag_soil_moist_prev=ag_soil_moist_prev,
                        carea_max=carea_max,
                        smidx_coef=smidx_coef,
                        smidx_exp=smidx_exp,
                        pptp=through_rain,
                        ptc=through_rain,
                        infil_ag=infil_ag,
                        sroff_ag=sroff_ag,
                    )

        # Handle snowmelt and precipitation
        # Fortran: lines ~1173-1207
        cond4 = pkwater_equiv < dnearzero
        cond6 = net_snow < nearzero

        if snowmelt > 0.0:
            avail_water = avail_water + snowmelt
            if perv_on:
                infil = infil + snowmelt
            if ag_on:
                infil_ag = infil_ag + snowmelt
            if hru_flag == 1:
                if (pkwater_equiv > 0.0) or (net_rain < nearzero):
                    # Pervious area computations
                    if perv_on:
                        infil, srp = check_capacity(
                            soil_moist_prev=soil_lower_prev,
                            soil_moist_max=soil_moist_max,
                            snowinfil_max=snowinfil_max,
                            infil=infil,
                            srp=srp,
                        )
                    # Agriculture area computations
                    if ag_on:
                        infil_ag, sroff_ag = check_capacity_ag(
                            ag_soil_moist_prev=ag_soil_moist_prev,
                            ag_soil_moist_max=ag_soil_moist_max,
                            snowinfil_max=snowinfil_max,
                            infil_ag=infil_ag,
                            sroff_ag=sroff_ag,
                        )
                else:
                    # Snowmelt occurred and depleted the snowpack
                    if perv_on:
                        infil, srp, contrib_fraction = perv_comp(
                            soil_moist_prev=soil_lower_prev,
                            carea_max=carea_max,
                            smidx_coef=smidx_coef,
                            smidx_exp=smidx_exp,
                            pptp=snowmelt,
                            ptc=net_ppt,
                            infil=infil,
                            srp=srp,
                        )
                    if ag_on:
                        infil_ag, sroff_ag, ag_contrib_fraction = ag_comp(
                            ag_soil_moist_prev=ag_soil_moist_prev,
                            carea_max=carea_max,
                            smidx_coef=smidx_coef,
                            smidx_exp=smidx_exp,
                            pptp=snowmelt,
                            ptc=net_ppt,
                            infil_ag=infil_ag,
                            sroff_ag=sroff_ag,
                        )

        elif cond4:
            # No snowmelt and no snowpack
            # Fortran: lines ~1217-1227
            if cond6 and through_rain > 0.0:
                avail_water = avail_water + through_rain
                if perv_on:
                    infil = infil + through_rain
                if ag_on:
                    infil_ag = infil_ag + through_rain

                if hru_flag == 1:
                    if perv_on:
                        infil, srp, contrib_fraction = perv_comp(
                            soil_moist_prev=soil_lower_prev,
                            carea_max=carea_max,
                            smidx_coef=smidx_coef,
                            smidx_exp=smidx_exp,
                            pptp=through_rain,
                            ptc=through_rain,
                            infil=infil,
                            srp=srp,
                        )
                    if ag_on:
                        infil_ag, sroff_ag, ag_contrib_fraction = ag_comp(
                            ag_soil_moist_prev=ag_soil_moist_prev,
                            carea_max=carea_max,
                            smidx_coef=smidx_coef,
                            smidx_exp=smidx_exp,
                            pptp=through_rain,
                            ptc=through_rain,
                            infil_ag=infil_ag,
                            sroff_ag=sroff_ag,
                        )

        # Snowpack exists, check capacity
        # Fortran: lines ~1233-1238
        elif infil > 0.0 or infil_ag > 0.0:
            if hru_flag == 1:
                if infil > 0.0 and perv_on:
                    infil, srp = check_capacity(
                        soil_moist_prev=soil_lower_prev,
                        soil_moist_max=soil_moist_max,
                        snowinfil_max=snowinfil_max,
                        infil=infil,
                        srp=srp,
                    )
                if infil_ag > 0.0 and ag_on:
                    infil_ag, sroff_ag = check_capacity_ag(
                        ag_soil_moist_prev=ag_soil_moist_prev,
                        ag_soil_moist_max=ag_soil_moist_max,
                        snowinfil_max=snowinfil_max,
                        infil_ag=infil_ag,
                        sroff_ag=sroff_ag,
                    )

        # Handle impervious area storage
        if hruarea_imperv > 0.0:
            imperv_stor = imperv_stor + avail_water
            if hru_flag == 1:
                if imperv_stor > imperv_stor_max:
                    sri = imperv_stor - imperv_stor_max
                    imperv_stor = imperv_stor_max

        return (
            sri,
            srp,
            sroff_ag,
            imperv_stor,
            infil,
            infil_ag,
            contrib_fraction,
            ag_contrib_fraction,
        )

    @staticmethod
    def ag_comp(
        ag_soil_moist_prev,
        carea_max,
        smidx_coef,
        smidx_exp,
        pptp,
        ptc,
        infil_ag,
        sroff_ag,
        ag_soil_rechr_prev=None,
        ag_soil_rechr_max=None,
    ):
        """Agricultural area contributing area computations.

        Fortran reference: ag_comp() subroutine lines ~1284-1321

        This is parallel to perv_comp() but for agricultural areas.
        Uses antecedent ag_soil_moist to compute contributing area fraction.
        """
        smidx_module = True
        if smidx_module:
            # Use antecedent ag_soil_moist
            # Fortran: lines ~1299-1307
            smidx = ag_soil_moist_prev + 0.5 * ptc
            if smidx > 25.0:
                ca_fraction = carea_max
            else:
                ca_fraction = smidx_coef * 10.0 ** (smidx_exp * smidx)
        else:
            # Use antecedent ag_soil_rechr
            # Fortran: lines ~1309-1310
            # Note: This branch uses carea_min and carea_dif which aren't
            # currently passed. For now, we only support smidx_module=True
            # which is the standard case.
            raise NotImplementedError(
                "ag_comp only supports smidx_module mode currently"
            )

        if ca_fraction > carea_max:
            ca_fraction = carea_max
        elif not (ca_fraction > 0.0):
            ca_fraction = 0.0

        srpp = ca_fraction * pptp
        if srpp < 0.0:
            srpp = 0.0

        infil_ag = infil_ag - srpp
        sroff_ag = sroff_ag + srpp

        return infil_ag, sroff_ag, ca_fraction

    @staticmethod
    def check_capacity_ag(
        ag_soil_moist_prev,
        ag_soil_moist_max,
        snowinfil_max,
        infil_ag,
        sroff_ag,
    ):
        """Check agricultural infiltration capacity.

        Fortran reference: check_capacity_ag() subroutine lines ~1409-1430

        Fill ag soil to ag_soil_moist_max, if more than capacity restrict
        infiltration by snowinfil_max, with excess added to runoff.
        """
        capacity = ag_soil_moist_max - ag_soil_moist_prev
        excess = infil_ag - capacity
        if excess > snowinfil_max:
            sroff_ag = sroff_ag + excess - snowinfil_max
            infil_ag = snowinfil_max + capacity

        return infil_ag, sroff_ag
