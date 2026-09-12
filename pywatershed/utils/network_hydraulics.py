"""Network hydraulics helpers and a model-agnostic NetCDF export.

The export carries reach topology, planform geometry, and per-reach,
per-time-step hydraulics (flow, velocity, depth, width, shear velocity)
in SI units for consumers such as 1D network particle trackers. See
``docs/superpowers/specs/2026-09-11-network-hydraulics-export-design.md``.
"""

import numpy as np

from ..base.parameters import Parameters

G = 9.80665
"""Gravitational acceleration (m/s^2)."""

SLOPE_FLOOR = 1.0e-7
"""Minimum slope (m/m), the floor used by PRMS stream temperature."""


def shear_velocity(depth: np.ndarray, slope: np.ndarray) -> np.ndarray:
    """Shear velocity sqrt(g * depth * slope) with the slope floor.

    Args:
        depth: flow depth (m), any shape.
        slope: channel slope (m/m), broadcastable to ``depth``.

    Returns:
        Shear velocity (m/s), same shape as the broadcast of the inputs.
    """
    slope_floored = np.maximum(np.asarray(slope, dtype=float), SLOPE_FLOOR)
    return np.sqrt(G * np.asarray(depth, dtype=float) * slope_floored)


def calculate_seg_mid_elevations(
    parameters: Parameters,
) -> tuple[np.ndarray, dict[int, float]]:
    """Elevation at the midpoint of each segment, walked up from outlets.

    Each outlet's downstream end takes the lowest elevation of the HRUs
    that drain to it; every segment's upstream end is its downstream
    end plus ``seg_slope * seg_length``; the midpoint is the mean of the
    two. Requires ``tosegment``, ``seg_slope``, ``seg_length``,
    ``hru_segment`` and ``hru_elev``.

    Args:
        parameters: a Parameters object with the parameters above.

    Returns:
        ``(seg_mid_elevation, outlet_mid_elevation)`` where the first is
        an array over segments (m) and the second maps each outlet's
        zero-based segment index to its midpoint elevation (m).
    """
    params = parameters.parameters
    seg_dy = params["seg_slope"] * params["seg_length"]
    nseg = len(seg_dy)
    seg_y = np.full(nseg, np.nan)  # elevation at the upstream end
    tosegment0 = params["tosegment"] - 1
    is_outflow = -1
    hru_seg = params["hru_segment"] - 1
    hru_elev = params["hru_elev"]
    outlet_mid = {}

    for ss in range(nseg):
        if not np.isnan(seg_y[ss]):
            continue
        # walk downstream until a solved segment or an outlet
        chain = []
        ind = ss
        while ind != is_outflow and np.isnan(seg_y[ind]):
            chain.append(ind)
            ind = tosegment0[ind]
        # solve from the most downstream unsolved segment upward
        for seg in reversed(chain):
            down = tosegment0[seg]
            if down == is_outflow:
                outlet_hrus = np.where(hru_seg == seg)
                outlet_elev = hru_elev[outlet_hrus].min()
                seg_y[seg] = seg_dy[seg] + outlet_elev
                outlet_mid[int(seg)] = float(seg_y[seg] - seg_dy[seg] / 2)
            else:
                seg_y[seg] = seg_dy[seg] + seg_y[down]

    return seg_y - seg_dy / 2, outlet_mid
