"""PRMS stream shade computation strategies.

These classes provide different methods for computing shade on stream segments.
They are designed to be composed into PRMSStreamTemp, not used as standalone
processes.
"""

import math

import numpy as np

from ..parameters import Parameters

# Constants from PRMS
NEARZERO = 1e-6
PI = np.pi
HALF_PI = PI / 2.0
RADTOHOUR = 24.0 / (2.0 * PI)

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


class PRMSStreamShade:
    """Base class for stream shade computation strategies.

    This is an abstract base class that defines the interface for shade
    computation. Subclasses implement different strategies (dynamic vs constant).
    """

    def __init__(
        self, parameters: Parameters, nsegment: int, parent_process=None
    ):
        """Initialize shade computer.

        Args:
            parameters: Parameters object containing shade parameters
            nsegment: Number of stream segments
            parent_process: Optional parent PRMSStreamTemp process for accessing shade methods
        """
        self.nsegment = nsegment
        self.parent_process = parent_process
        self._load_parameters(parameters)

    def _load_parameters(self, parameters: Parameters) -> None:
        """Load parameters from Parameters object.

        Subclasses should override this to load their specific parameters.

        Args:
            parameters: Parameters object
        """
        raise NotImplementedError("Subclasses must implement _load_parameters")

    @staticmethod
    def get_parameters() -> tuple:
        """Get required parameters for this shade computation strategy.

        Returns:
            Tuple of parameter names required by this strategy
        """
        raise NotImplementedError("Subclasses must implement get_parameters")

    def compute(
        self,
        seg_idx: int,
        declination: float,
        summer_flag: int,
        seg_flow_width: float,
    ) -> tuple[float, float]:
        """Compute shade for a segment.

        Args:
            seg_idx: Segment index (0-based)
            declination: Solar declination (radians)
            summer_flag: 1 for summer, 0 for winter
            seg_flow_width: Flow width for the segment (meters)

        Returns:
            Tuple of (shade, svi) where:
                shade: Shade fraction (0-1)
                svi: Vegetation shade index
        """
        raise NotImplementedError("Subclasses must implement compute")


class PRMSStreamShadeConstant(PRMSStreamShade):
    """Constant shade parameters by season.

    Uses pre-specified constant shade values for summer and winter seasons.
    This is used when stream_temp_shade_flag = 1.

    Requires 2 parameters: summer shade and winter shade fractions.
    """

    def _load_parameters(self, parameters: Parameters) -> None:
        """Load constant shade parameters.

        Args:
            parameters: Parameters object
        """
        self.segshade_sum = parameters.get_param_values("segshade_sum")
        self.segshade_win = parameters.get_param_values("segshade_win")
        self.seg_lat = parameters.get_param_values("seg_lat")
        self._seg_lat_rad = np.deg2rad(self.seg_lat)

    @staticmethod
    def get_parameters() -> tuple:
        """Get required constant shade parameters."""
        return (
            "segshade_sum",  # Total shade fraction for summer vegetation
            "segshade_win",  # Total shade fraction for winter vegetation
            "seg_lat",
        )

    def compute(
        self,
        seg_idx: int,
        declination: float,
        summer_flag: int,
        seg_flow_width: float,
    ) -> tuple[float, float]:
        """Return constant shade value based on season.

        Args:
            seg_idx: Segment index (0-based)
            declination: Solar declination (not used for constant shade)
            summer_flag: 1 for summer, 0 for winter
            seg_flow_width: Flow width (not used for constant shade)

        Returns:
            Tuple of (shade, svi) where:
                shade: Constant shade fraction for the season (0-1)
                svi: Always 0.0 for constant shade
        """
        if summer_flag == 1:
            shade = self.segshade_sum[seg_idx]
        else:
            shade = self.segshade_win[seg_idx]

        # svi (vegetation shade index) is not used for constant shade
        return shade, 0.0


class PRMSStreamShadeDynamic(PRMSStreamShade):
    """Dynamic shade computation from topography and vegetation.

    Computes shade dynamically based on topographic and vegetation parameters
    using solar geometry calculations. This is the default PRMS behavior when
    stream_temp_shade_flag = 0.

    Requires 13 parameters describing topography and vegetation characteristics
    for each stream segment.
    """

    def _load_parameters(self, parameters: Parameters) -> None:
        """Load topography and vegetation parameters.

        Args:
            parameters: Parameters object
        """
        self.azrh = parameters.get_param_values("azrh")
        self.alte = parameters.get_param_values("alte")
        self.altw = parameters.get_param_values("altw")
        self.vce = parameters.get_param_values("vce")
        self.vdemx = parameters.get_param_values("vdemx")
        self.vdemn = parameters.get_param_values("vdemn")
        self.vhe = parameters.get_param_values("vhe")
        self.voe = parameters.get_param_values("voe")
        self.vcw = parameters.get_param_values("vcw")
        self.vdwmx = parameters.get_param_values("vdwmx")
        self.vdwmn = parameters.get_param_values("vdwmn")
        self.vhw = parameters.get_param_values("vhw")
        self.vow = parameters.get_param_values("vow")
        self.maxiter_sntemp = parameters.get_param_values("maxiter_sntemp")
        # Dynamic shade: zero latitude to matches Fortran
        self._seg_lat_rad = np.zeros(self.nsegment, dtype=np.float64)

    @staticmethod
    def get_parameters() -> tuple:
        """Get required topography and vegetation parameters."""
        return (
            "azrh",  # Azimuth angle
            "alte",  # East bank topographic altitude
            "altw",  # West bank topographic altitude
            "vce",  # East bank vegetation crown width
            "vdemx",  # Maximum east bank vegetation density
            "vdemn",  # Minimum east bank vegetation density
            "vhe",  # East bank vegetation height
            "voe",  # East bank vegetation offset
            "vcw",  # West bank vegetation crown width
            "vdwmx",  # Maximum west bank vegetation density
            "vdwmn",  # Minimum west bank vegetation density
            "vhw",  # West bank vegetation height
            "vow",  # West bank vegetation offset
            "maxiter_sntemp",
        )

    def compute(
        self,
        seg_idx: int,
        declination: float,
        summer_flag: int,
        seg_flow_width: float,
    ) -> tuple[float, float]:
        """Compute shade using shday algorithm.

        Args:
            seg_idx: Segment index (0-based)
            declination: Solar declination (radians)
            summer_flag: 1 for summer, 0 for winter
            seg_flow_width: Flow width for the segment (meters)

        Returns:
            Tuple of (shade, svi) where:
                shade: Shade fraction (0-1)
                svi: Vegetation shade index
        """
        shade, svi = _shday(
            self._seg_lat_rad[seg_idx],
            declination,
            seg_flow_width,
            self.azrh[seg_idx],
            self.alte[seg_idx],
            self.altw[seg_idx],
            self.vce[seg_idx],
            self.voe[seg_idx],
            self.vhe[seg_idx],
            self.vdemx[seg_idx],
            self.vdemn[seg_idx],
            summer_flag,
            self.vcw[seg_idx],
            self.vow[seg_idx],
            self.vhw[seg_idx],
            self.vdwmx[seg_idx],
            self.vdwmn[seg_idx],
            self.maxiter_sntemp,
        )

        return shade, svi


def _shday(
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
    maxiter_sntemp,
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
    # alsmx = max_solar_altitude

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
        alrs = _solalt(
            coso, sino, sin_d, azrh, 0.0, max_solar_altitude, maxiter_sntemp
        )

    sin_alrs = np.sin(alrs)

    # Calculate level-plain sunrise/set hour angle
    tano = sino / coso
    tan_d = sin_d / cos_d
    tanod = tano * tan_d
    horizontal_hour_angle = np.arccos(-tanod)
    sinhro = np.sin(horizontal_hour_angle)

    # Calculate total potential shade on level-plain
    total_shade = 2.0 * ((horizontal_hour_angle * sinod) + (sinhro * cosod))
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
        svi = _rprnvg(
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
            als = _solalt(coso, sino, sin_d, azs, almn, almx, maxiter_sntemp)
            azs, als, hrs = _snr_sst(
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
                maxiter_sntemp,
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
            als = _solalt(coso, sino, sin_d, azs, almn, almx, maxiter_sntemp)
            azs, als, hrs = _snr_sst(
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
                maxiter_sntemp,
            )
            hrss = hrs

        if hrrs < hrsr:
            hrrh = hrsr
        elif hrrs > hrss:
            hrrh = hrss
        else:
            hrrh = hrrs

        # seg_daylight = (hrss - hrsr) * RADTOHOUR
        sti = 1.0 - (
            (
                ((hrss - hrsr) * sinod)
                + ((math.sin(hrss) - math.sin(hrsr)) * cosod)
            )
            / (totsh)
        )
        svi = (
            _rprnvg(
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


def _solalt(coso, sino, sin_d, az, almn, almx, maxiter_sntemp):
    """Determine solar altitude from trigonometric parameters.

    This is the solalt function from PRMS.
    """
    maxiter_sntemp = int(maxiter_sntemp[0])

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
    maxiter_sntemp,
):
    """Determine local solar sunrise/set azimuth, altitude, and hour angle.

    This is the snr_sst function from PRMS.
    """
    maxiter_sntemp = int(maxiter_sntemp[0])

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
