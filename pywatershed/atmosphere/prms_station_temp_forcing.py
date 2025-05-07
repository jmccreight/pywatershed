import pathlib as pl
from typing import Union

from pywatershed.base.control import Control
from pywatershed.base.hru_mixin import HruMixin
from pywatershed.base.parameters import Parameters
from pywatershed.base.process import Process
from pywatershed.constants import c_to_f, nan


class PRMSStationTempForcing(Process, HruMixin):
    """PRMS station temperature data interpolation for forcing inputs.

    Implementation based on PRMS 5.2.1 with theoretical documentation given in
    the PRMS-IV documentation:

    `Markstrom, S. L., Regan, R. S., Hay, L. E., Viger, R. J., Webb, R. M.,
    Payn, R. A., & LaFontaine, J. H. (2015). PRMS-IV, the
    precipitation-runoff modeling system, version 4. US Geological Survey
    Techniques and Methods, 6, B7.
    <https://pubs.usgs.gov/tm/6b7/pdf/tm6-b7.pdf>`__

    Units Note: Input units are governed by the parameter "temp_units". Output units
    are in degrees Farenheit.

    Args:
        control: a Control object
        discretization: a discretization of class Parameters
        parameters: a parameter object of class Parameters

        tmax_stns: daily maximum temperature at ntemp stations.
        tmin_stns: daily minimum temperature at ntemp stations.
    """

    def __init__(
        self,
        control: Control,
        discretization: Parameters,
        parameters: Parameters,
        tmax_sta: Union[str, pl.Path],
        tmin_sta: Union[str, pl.Path],
        verbose: bool = False,
        check_min_max: bool = False,
    ):
        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
        )
        self.name = "PRMSStationTempForcing"
        self._set_active_hrus()
        self._mask_inactive_hrus()

        self._set_inputs(locals())
        self._set_options(locals())

        return

    @staticmethod
    def get_dimensions() -> tuple:
        return (
            "nhru",
            "ntemp",
        )

    @staticmethod
    def get_parameters() -> tuple:
        return (
            "hru_elev",
            "hru_tsta",
            "hru_type",
            "temp_units",
            "tmax_lapse",
            "tmin_lapse",
            "tmax_adj",
            "tmin_adj",
            "tsta_elev",
        )

    @staticmethod
    def get_inputs() -> tuple:
        return (
            "tmax_sta",
            "tmin_sta",
        )

    @staticmethod
    def get_init_values() -> dict:
        return {
            "tmax": nan,
            "tmin": nan,
        }

    def _set_initial_conditions(self):
        # Have renamed tmin/tmax_aspect_adjust -> tmin/tmax_adj
        self._previous_month = -1
        self._elfac = self.tmax * nan
        self._tcrn = self.tmax * nan
        self._tcrx = self.tmax * nan

        # Approximately temp_1sta_laps.f90:L238
        # Using previous_month and current_month, tcrx and tcrn calculations
        # are deferred and the init colapses to this:
        self._elfac = (
            self.hru_elev - self.tsta_elev[self.hru_tsta - 1]
        ) / 1000.0
        #
        # for ii in range(self._nactive_hrus):
        #     jj = self._wh_active_hrus[ii]
        #     kk = self.hru_tsta[jj] - 1  # this is always 1/0 in this 1sta case
        #     # JLM:  Why is this not just a special case of multi station? or is it a global variable
        #     # Nuse_tsta[kk] = 1
        #     # Hru_elev_ts is the current elevation, either hru_elev or for restart Hru_elev_ts
        #     # JLM: does this normalization depend on units??
        #     self._elfac[jj] = (self.hru_elev[jj] - self.tsta_elev[kk]) / 1000.0
        #     self._tcrx[jj] = (
        #         self.tmax_lapse[jj, start_month - 1] * elfac[jj]
        #         - self.tmax_adj[jj, start_month]
        #     )
        #     self._tcrn[jj] = (
        #         self.tmin_lapse[jj, start_month - 1] * elfac[jj]
        #         - self.tmin_adj[jj, start_month]
        #     )

        return

    def _init_calc_method(self):
        # TODO
        pass

    def _advance_variables(self):
        pass

    def _calculate(self, simulation_time):
        (
            self._previous_month,
            self.tmax[:],
            self.tmin[:],
            self._tcrn[:],
            self._tcrx[:],
        ) = self._calculate_station(
            nactive_hrus=self._nactive_hrus,
            wh_active_hrus=self._wh_active_hrus,
            current_month=self.control.current_month,
            elfac=self._elfac,
            hru_elev=self.hru_elev,
            hru_tsta=self.hru_tsta,
            tmax_lapse=self.tmax_lapse,
            tmin_lapse=self.tmin_lapse,
            tmax_adj=self.tmax_adj,
            tmin_adj=self.tmin_adj,
            tsta_elev=self.tsta_elev,
            temp_units=self.temp_units,
            check_min_max=self._check_min_max,
            c_to_f=c_to_f,
            previous_month=self._previous_month,
            tmax_sta=self.tmax_sta,
            tmin_sta=self.tmin_sta,
            tcrn=self._tcrn,
            tcrx=self._tcrx,
        )

    @staticmethod
    def _calculate_station(
        nactive_hrus,
        wh_active_hrus,
        current_month,
        elfac,
        hru_elev,
        hru_tsta,
        tmax_lapse,
        tmin_lapse,
        tmax_adj,
        tmin_adj,
        tsta_elev,
        temp_units,
        check_min_max,
        c_to_f,
        previous_month,  # inout variables
        tmax_sta,
        tmin_sta,
        tcrn,
        tcrx,
    ):
        """Calculate min & max HRU temperatures in degrees F from temp_units inputs."""

        # IMPLEMENT CHECKS

        tmax = hru_elev * nan
        tmin = hru_elev * nan

        # vectorize the PRMS calculations
        jj = wh_active_hrus
        kk = hru_tsta[jj] -1

        if current_month != previous_month:
            cm = current_month - 1
            tcrx[wh_active_hrus] = tmax_lapse[cm, jj] * elfac[jj] - tmax_adj[cm, jj]
            tcrn[wh_active_hrus] = tmin_lapse[cm, jj] * elfac[jj] - tmin_adj[cm, jj]

        tmax[wh_active_hrus] = tmax_sta[kk] - tcrx[wh_active_hrus]
        tmin[wh_active_hrus] = tmin_sta[kk] - tcrn[wh_active_hrus]
        previous_month = current_month

        if temp_units == 1:
            # input in Celsius, output in Farenheit
            tmax = c_to_f(tmax)
            tmin = c_to_f(tmin)

        return (previous_month, tmax, tmin, tcrn, tcrx)
