import numpy as np

from ..base.adapter import adaptable
from ..base.control import Control
from ..base.process import Process
from ..parameters import Parameters

# Constants from PRMS
NEARZERO = 1e-6
CFS_TO_CMS = 0.028316847


class PRMSHydraulicGeometryFull(Process):
    """PRMS hydraulic geometry.

    Computes flow-dependent hydraulic geometry (width, depth, area, velocity,
    residence time) for stream segments using power-law relationships. This
    implementation is based on the strmflow_character module from PRMS 5.2.1.

    Hydraulic geometry relationships:
    - width = width_alpha * flow^width_m
    - depth = depth_alpha * flow^depth_m
    - area = width * depth
    - velocity = flow / area
    - residence_time = (area * length) / flow

    Args:
        control: a Control object
        discretization: a discretization of class Parameters
        parameters: a parameter object of class Parameters
        seg_outflow: Streamflow leaving each segment (cfs)
        verbose: Print extra information or not?
    """

    def __init__(
        self,
        control: Control,
        discretization: Parameters,
        parameters: Parameters,
        seg_outflow: adaptable,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
        )
        self.name = "PRMSHydraulicGeometryFull"

        self._set_inputs(locals())
        self._set_options(locals())

        return

    @staticmethod
    def get_dimensions() -> tuple:
        return ("nsegment",)

    @staticmethod
    def get_parameters() -> tuple:
        return (
            "width_alpha",
            "width_m",
            "depth_alpha",
            "depth_m",
            "seg_length",
        )

    @staticmethod
    def get_inputs() -> tuple:
        return ("seg_outflow",)

    @staticmethod
    def get_init_values() -> dict:
        return {
            "seg_flow_width": 0.0,
            "seg_flow_depth": 0.0,
            "seg_flow_area": 0.0,
            "seg_flow_velocity": 0.0,
            "seg_res_time": 0.0,
        }

    @staticmethod
    def get_variables() -> tuple:
        return (
            "seg_flow_width",
            "seg_flow_depth",
            "seg_flow_area",
            "seg_flow_velocity",
            "seg_res_time",
        )

    def _set_initial_conditions(self) -> None:
        return

    def _advance_variables(self) -> None:
        """Advance variables in time (not used for this process)"""
        return

    def _calculate(self, time_length) -> None:
        """Calculate hydraulic geometry for current timestep.

        Args:
            time_length: length of time step
        """
        self._compute_hydraulic_geometry()
        return

    def _compute_hydraulic_geometry(self) -> None:
        """Compute hydraulic geometry from flow using power-law relationships.

        Implements the strmflow_character.f90 module calculations:
        - seg_flow_width = width_alpha * flow^width_m
        - seg_flow_depth = depth_alpha * flow^depth_m
        - seg_flow_area = seg_flow_width * seg_flow_depth
        - seg_flow_velocity = flow / seg_flow_area
        - seg_res_time = (seg_flow_area * seg_length) / flow

        VECTORIZED: Uses NumPy array operations for all segments at once.
        """
        # Convert flow from cfs to cms for all segments
        flow_cms = self.seg_outflow * CFS_TO_CMS

        # Create mask for segments with flow
        has_flow = flow_cms > 0.0

        # Initialize all to zero
        self.seg_flow_width[:] = 0.0
        self.seg_flow_depth[:] = 0.0
        self.seg_flow_area[:] = 0.0
        self.seg_flow_velocity[:] = 0.0
        self.seg_res_time[:] = 0.0

        # Compute width and depth from power-law relationships (vectorized)
        # Only for segments with flow
        self.seg_flow_width[has_flow] = self.width_alpha[has_flow] * (
            flow_cms[has_flow] ** self.width_m[has_flow]
        )
        self.seg_flow_depth[has_flow] = self.depth_alpha[has_flow] * (
            flow_cms[has_flow] ** self.depth_m[has_flow]
        )

        # Compute cross-sectional area (vectorized)
        self.seg_flow_area[has_flow] = (
            self.seg_flow_width[has_flow] * self.seg_flow_depth[has_flow]
        )

        # Compute velocity where area > NEARZERO (vectorized)
        has_area = has_flow & (self.seg_flow_area > NEARZERO)
        self.seg_flow_velocity[has_area] = (
            flow_cms[has_area] / self.seg_flow_area[has_area]
        )

        # Compute residence time (vectorized)
        # residence_time (seconds) = (area (m²) * length (m)) / flow (m³/s)
        self.seg_res_time[has_flow] = (
            self.seg_flow_area[has_flow] * self.seg_length[has_flow]
        ) / flow_cms[has_flow]

        return


# JLM: I dont love the design conceptually, but it is efficient.
class PRMSHydraulicGeometryWidthOnly(PRMSHydraulicGeometryFull):
    """PRMS hydraulic geometry with default depth parameters.

    This subclass uses PRMS default values for depth_alpha and depth_m when
    those parameters are not provided in the parameter file. This matches the
    PRMS 5.2.1 behavior where missing parameters fall back to defaults.

    Default values from strmflow_character.f90:
    - depth_alpha = 0.27 (range: 0.12 - 0.63 meters)
    - depth_m = 0.39 (range: 0.38 - 0.40)

    Width parameters (width_alpha, width_m) are still required.

    Args:
        control: a Control object
        discretization: a discretization of class Parameters
        parameters: a parameter object of class Parameters
        seg_outflow: Streamflow leaving each segment (cfs)
        verbose: Print extra information or not?
    """

    def __init__(
        self,
        control: Control,
        discretization: Parameters,
        parameters: Parameters,
        seg_outflow: adaptable,
        verbose: bool = False,
    ) -> None:
        # Call parent init
        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
            seg_outflow=seg_outflow,
            verbose=verbose,
        )
        self.name = "PRMSHydraulicGeometryWidthOnly"
        self.depth_alpha = np.full(self.nsegment, 0.27, dtype=np.float64)
        self.depth_m = np.full(self.nsegment, 0.39, dtype=np.float64)

        return

    @staticmethod
    def get_parameters() -> tuple:
        """Get required parameters (only width, depth uses defaults)."""
        return (
            "width_alpha",
            "width_m",
            "seg_length",
        )
