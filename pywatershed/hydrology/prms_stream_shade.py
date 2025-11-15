"""PRMS stream shade computation strategies.

These classes provide different methods for computing shade on stream segments.
They are designed to be composed into PRMSStreamTemp, not used as standalone
processes.
"""

import numpy as np

from ..parameters import Parameters

# Constants from PRMS
NEARZERO = 1e-6
PI = np.pi
HALF_PI = PI / 2.0


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
        # This will call the shday method that computes shade from
        # topography and vegetation parameters
        # For now, return placeholder - will be implemented when refactoring
        # PRMSStreamTemp to use this interface
        shade, svi = self._shday(
            seg_idx,
            declination,
            seg_flow_width,
            summer_flag,
        )
        return shade, svi

    def _shday(
        self,
        seg_idx: int,
        declination: float,
        seg_width: float,
        summer_flag: int,
    ) -> tuple[float, float]:
        """Compute daily shade from topography and vegetation.

        This delegates to the parent PRMSStreamTemp process which contains
        the full _shday implementation and helper methods.

        Args:
            seg_idx: Segment index
            declination: Solar declination (radians)
            seg_width: Stream width (meters)
            summer_flag: 1 for summer, 0 for winter

        Returns:
            Tuple of (shade, svi)
        """
        if self.parent_process is None:
            # Fallback if no parent process provided
            return 0.0, 0.0

        # Delegate to parent process _shday method
        return self.parent_process._shday(
            self.parent_process.seg_lat_rad[seg_idx],
            declination,
            seg_width,
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
        )


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

    @staticmethod
    def get_parameters() -> tuple:
        """Get required constant shade parameters."""
        return (
            "segshade_sum",  # Total shade fraction for summer vegetation
            "segshade_win",  # Total shade fraction for winter vegetation
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
