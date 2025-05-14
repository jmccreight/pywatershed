import pathlib as pl
from typing import Union

import numpy as np

from pywatershed.base.control import Control
from pywatershed.base.hru_mixin import HruMixin
from pywatershed.base.parameters import Parameters
from pywatershed.base.process import Process
from pywatershed.constants import (
    c_to_f,
    mm2in,
    nan,
    nearzero,
    zero,
    PrecipUnits,
    TempUnits,
)


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

        self._elfac = (
            self.hru_elev - self.tsta_elev[self.hru_tsta - 1]
        ) / 1000.0

        return

    def _init_calc_method(self):
        # probably moot being vectorized
        pass

    def _advance_variables(self):
        # no memory
        pass

    def _calculate(self, simulation_time):
        (
            self._previous_month,
            self.tmax[:],
            self.tmin[:],
            self._tcrn[:],
            self._tcrx[:],
        ) = self._calculate_station(
            wh_active_hrus=self._wh_active_hrus,
            current_month=self.control.current_month,
            elfac=self._elfac,
            hru_tsta=self.hru_tsta,
            tmax_lapse=self.tmax_lapse,
            tmin_lapse=self.tmin_lapse,
            tmax_adj=self.tmax_adj,
            tmin_adj=self.tmin_adj,
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
        wh_active_hrus,
        current_month,
        elfac,
        hru_tsta,
        tmax_lapse,
        tmin_lapse,
        tmax_adj,
        tmin_adj,
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

        # no memory here
        tmax = tcrx * nan
        tmin = tcrn * nan

        # vectorize the PRMS calculations
        jj = wh_active_hrus
        kk = hru_tsta[jj] - 1

        if current_month != previous_month:
            cm = current_month - 1
            tcrx[wh_active_hrus] = (
                tmax_lapse[cm, jj] * elfac[jj] - tmax_adj[cm, jj]
            )
            tcrn[wh_active_hrus] = (
                tmin_lapse[cm, jj] * elfac[jj] - tmin_adj[cm, jj]
            )

        tmax[wh_active_hrus] = tmax_sta[kk] - tcrx[wh_active_hrus]
        tmin[wh_active_hrus] = tmin_sta[kk] - tcrn[wh_active_hrus]
        previous_month = current_month

        if temp_units == 1:
            # input in Celsius, output in Farenheit
            tmax = c_to_f(tmax)
            tmin = c_to_f(tmin)

        return (previous_month, tmax, tmin, tcrn, tcrx)


class PRMSStationPrecipForcing(Process, HruMixin):
    """PRMS station precipitation data interpolation for forcing inputs.

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

        precip_sta: daily precipitation at nrain stations.
    """

    def __init__(
        self,
        control: Control,
        discretization: Parameters,
        parameters: Parameters,
        precip_sta: Union[str, pl.Path],
        tmaxf: Union[str, pl.Path],
        tminf: Union[str, pl.Path],
        verbose: bool = False,
        check_min_max: bool = False,
    ):
        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
        )
        self.name = "PRMSStationPrecipForcing"
        self._set_active_hrus()
        self._mask_inactive_hrus()

        self._set_inputs(locals())
        self._set_options(locals())

        return

    @staticmethod
    def get_dimensions() -> tuple:
        return (
            "nhru",
            "nrain",
        )

    @staticmethod
    def get_parameters() -> tuple:
        return (
            "adjmix_rain",
            "hru_psta",
            "hru_type",
            "precip_units",
            "rain_adj",
            "snow_adj",
            "temp_units",
            "tmax_allrain_offset",
            "tmax_allsnow",
        )

    @staticmethod
    def get_inputs() -> tuple:
        return (
            "precip_sta",
            "tmaxf",
            "tminf",
        )

    @staticmethod
    def get_init_values() -> dict:
        return {
            "hru_ppt": nan,
            "hru_rain": nan,
            "hru_snow": nan,
            "prmx": nan,
            "newsnow": False,
            "pptmix": False,
        }

    def _set_initial_conditions(self):
        self._tmax_allrain_f = self.tmax_allsnow + self.tmax_allrain_offset
        self._tmax_allsnow_f = self.tmax_allsnow
        if self.temp_units == TempUnits.CELSIUS.value:
            self._tmax_allrain_f = c_to_f(self._tmax_allrain_f)
            self._tmax_allsnow_f = c_to_f(self.tmax_allsnow)
        # <
        return

    def _init_calc_method(self):
        pass

    def _advance_variables(self):
        pass

    def _calculate(self, simulation_time):
        (
            self.hru_ppt[:],
            self.hru_rain[:],
            self.hru_snow[:],
            self.hru_snow[:],
            self.prmx[:],
            self.newsnow[:],
        ) = self._calculate_station(
            wh_active_hrus=self._wh_active_hrus,
            current_month=self.control.current_month,
            precip_sta=self.precip_sta,
            tmaxf=self.tmaxf,
            tminf=self.tminf,
            hru_psta=self.hru_psta,
            tmax_allrain_f=self._tmax_allrain_f,
            tmax_allsnow_f=self._tmax_allsnow_f,
            adjmix_rain=self.adjmix_rain,
            rain_adj=self.rain_adj,
            snow_adj=self.snow_adj,
            precip_units=self.precip_units,
            PrecipUnitsEnum=PrecipUnits,
            zero=zero,
            nearzero=nearzero,
            mm2in=mm2in,
        )

    @staticmethod
    def _calculate_station(
        wh_active_hrus,
        current_month,
        precip_sta,
        tmaxf,
        tminf,
        hru_psta,
        tmax_allrain_f,
        tmax_allsnow_f,
        adjmix_rain,
        rain_adj,
        snow_adj,
        precip_units,
        PrecipUnitsEnum,
        zero,
        nearzero,
        mm2in,
    ):
        """Calculate rain and snow on HRUs in inches from rain_units inputs."""

        cm = current_month - 1

        wh_precip_sta_neg = np.where(precip_sta < zero)[0]
        if len(wh_precip_sta_neg):
            msg = "Negative precip encountered"
            # maybe should just be a warning?
            raise ValueError(msg)

        if precip_units == PrecipUnitsEnum.mm.value:
            precip_sta *= mm2in

        hru_ppt = tmaxf * zero
        hru_rain = tmaxf * zero
        hru_snow = tmaxf * zero
        prmx = tmaxf * zero
        pptmix = tmaxf * 0
        newsnow = tmaxf * 0

        for ii in wh_active_hrus:
            if tmaxf[ii] <= tmax_allsnow_f[cm, ii]:
                hru_ppt[ii] = precip_sta[hru_psta[ii] - 1] * snow_adj[cm, ii]
                hru_snow[ii] = hru_ppt[ii]
                newsnow[ii] = 1

            elif (
                tminf[ii] > tmax_allsnow_f[cm, ii]
                or tmaxf[ii] >= tmax_allrain_f[cm, ii]
            ):
                # If minimum temperature is above base temperature for snow or
                # maximum temperature is above all_rain temperature then
                # precipitation is all rain
                hru_ppt[ii] = precip_sta[hru_psta[ii] - 1] * rain_adj[cm, ii]
                hru_rain[ii] = hru_ppt[ii]
                prmx[ii] = 1.0

            else:
                # Otherwise precipitation is a mixture of rain and snow
                tdiff = tmaxf[ii] - tminf[ii]
                if tdiff < zero:
                    pass
                # PRINT *, 'ERROR, tmax < tmin (degrees Fahrenheit), tmax:', Tmaxf, ' tmin:', TminF
                # CALL print_date(1)

                if abs(tdiff) < nearzero:
                    tdiff = 0.0001

                # <
                prmx[ii] = (
                    (tmaxf[ii] - tmax_allsnow_f[cm, ii]) / tdiff
                ) * adjmix_rain[cm, ii]
                if prmx[ii] < zero:
                    prmx[ii] = zero

                if prmx[ii] < 1.0:
                    # Unless mixture adjustment raises the proportion of rain to
                    # greater than or equal to 1.0 in which case it all rain
                    # If not, it is a rain/snow mixture
                    pptmix[ii] = 1
                    hru_ppt[ii] = (
                        precip_sta[hru_psta[ii] - 1] * snow_adj[cm, ii]
                    )
                    hru_rain[ii] = prmx[ii] * hru_ppt[ii]
                    hru_snow[ii] = hru_ppt[ii] - hru_rain[ii]
                    newsnow[ii] = 1
                else:
                    hru_ppt[ii] = (
                        precip_sta[hru_psta[ii] - 1] * rain_adj[cm, ii]
                    )
                    hru_rain[ii] = hru_ppt[ii]
                    prmx[ii] = 1.0

        # <<
        return (hru_ppt, hru_rain, hru_snow, prmx, pptmix, newsnow)
