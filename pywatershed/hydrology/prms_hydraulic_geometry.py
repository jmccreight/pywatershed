import numpy as np

from ..base.adapter import adaptable
from ..base.control import Control
from ..base.process import Process
from ..parameters import Parameters

# Constants from PRMS
NEARZERO = 1e-6
CFS_TO_CMS = 0.028316847


class PRMSHydraulicGeometry(Process):
    """PRMS hydraulic geometry with custom power-law parameters.

    Computes flow-dependent hydraulic geometry (width, depth, area, velocity)
    for stream segments using power-law relationships. This implementation is
    based on the strmflow_character module from PRMS 5.2.1.

    Hydraulic geometry relationships:
    - width = width_alpha * flow^width_m
    - depth = depth_alpha * flow^depth_m
    - area = width * depth
    - velocity = flow / area

    Args:
        control: a Control object
        discretization: a discretization of class Parameters
        parameters: a parameter object of class Parameters
        seg_outflow: Streamflow leaving each segment (cfs)
        budget_type: one of ["defer", None, "warn", "error"]
        verbose: Print extra information or not?
    """

    def __init__(
        self,
        control: Control,
        discretization: Parameters,
        parameters: Parameters,
        seg_outflow: adaptable,
        budget_type: str = "defer",
        verbose: bool = False,
    ) -> None:
        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
        )
        self.name = "PRMSHydraulicGeometry"

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
        }

    @staticmethod
    def get_variables() -> tuple:
        return (
            "seg_flow_width",
            "seg_flow_depth",
            "seg_flow_area",
            "seg_flow_velocity",
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
        """
        for i in range(self.nsegment):
            flow_cfs = self.seg_outflow[i]

            if flow_cfs > NEARZERO:
                # Convert flow from cfs to cms
                flow_cms = flow_cfs * CFS_TO_CMS

                # Compute width and depth from power-law relationships
                self.seg_flow_width[i] = self.width_alpha[i] * (
                    flow_cms ** self.width_m[i]
                )
                self.seg_flow_depth[i] = self.depth_alpha[i] * (
                    flow_cms ** self.depth_m[i]
                )

                # Compute cross-sectional area
                self.seg_flow_area[i] = (
                    self.seg_flow_width[i] * self.seg_flow_depth[i]
                )

                # Compute velocity
                if self.seg_flow_area[i] > NEARZERO:
                    # Convert flow to m³/s for velocity calculation
                    self.seg_flow_velocity[i] = (
                        flow_cms / self.seg_flow_area[i]
                    )
                else:
                    self.seg_flow_velocity[i] = 0.0
            else:
                # No flow - set all to zero
                self.seg_flow_width[i] = 0.0
                self.seg_flow_depth[i] = 0.0
                self.seg_flow_area[i] = 0.0
                self.seg_flow_velocity[i] = 0.0

        return


class PRMSHydraulicGeometryDefault(PRMSHydraulicGeometry):
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
        budget_type: one of ["defer", None, "warn", "error"]
        verbose: Print extra information or not?
    """

    def __init__(
        self,
        control: Control,
        discretization: Parameters,
        parameters: Parameters,
        seg_outflow: adaptable,
        budget_type: str = "defer",
        verbose: bool = False,
    ) -> None:
        # Call parent init
        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
            seg_outflow=seg_outflow,
            budget_type=budget_type,
            verbose=verbose,
        )
        self.name = "PRMSHydraulicGeometryDefault"

        # Override depth parameters with defaults if not already set
        # This happens after parent __init__, which would have set them if
        # present. Check if they exist and have non-default values,
        # otherwise use PRMS defaults
        if not hasattr(self, "depth_alpha") or self.depth_alpha is None:
            self.depth_alpha = np.full(self.nsegment, 0.27, dtype=np.float64)
        if not hasattr(self, "depth_m") or self.depth_m is None:
            self.depth_m = np.full(self.nsegment, 0.39, dtype=np.float64)

        return

    @staticmethod
    def get_parameters() -> tuple:
        """Get required parameters (only width, depth uses defaults)."""
        return (
            "width_alpha",
            "width_m",
            # depth_alpha and depth_m are optional, defaults used
        )
