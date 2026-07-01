"""Pure board geometry: solve board pose from markers and generate square centers.

This is the hardware-free core of ``ChessRoboPlayer.calculate_board_grids``.
Keeping it free of ``rospy`` / OpenCV / the ``utili`` package lets it run under
the unit-test interpreter and pins the square-coordinate math before the
Phase C.3 hand-eye/transform refactor changes how marker world positions are
produced.

Frame and indexing conventions match the legacy ``board_fit`` module (now in
``archive/scripts/utili/``) that the runtime used previously; this module
supersedes it.
``A1`` is the board origin, the file axis runs A->H and the rank axis 1->8, and
``th`` is the board yaw about base Z (radians).
"""

from math import cos, sin

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation

FILES = "ABCDEFGH"

# Lower/upper bounds for the board-pose solve: x in [0, 2] m, y/th in [-1, 1].
DEFAULT_POSE_BOUNDS = ([0, -1, -1], [2, 1, 1])


def corner_xy(size, x0, y0, th):
    """Return the four board corners (A1, H1, H8, A8) as (x, y) tuples.

    ``size`` is the full edge length of the playable area (8 * square size).
    """
    a1 = (x0, y0)
    h1 = (x0 + size * cos(th), y0 + size * sin(th))
    h8 = (x0 + size * cos(th) - size * sin(th), y0 + size * sin(th) + size * cos(th))
    a8 = (x0 - size * sin(th), y0 + size * cos(th))
    return a1, h1, h8, a8


def board_corners(size, x, y, th, height):
    """Return the 4x3 corner array (A1, H1, H8, A8) at a fixed ``height``."""
    corners = np.zeros((4, 3))
    for i, (cx, cy) in enumerate(corner_xy(size, x, y, th)):
        corners[i, 0] = cx
        corners[i, 1] = cy
    corners[:, 2] = height
    return corners


def _marker_distance(var, x, y, th):
    """Distance of each marker to its nearest board corner (curve_fit residual)."""
    markers = var["markers"]
    corners = corner_xy(var["size"], x, y, th)
    residuals = []
    for m in range(len(markers)):
        residuals.append(
            min(
                ((cx - markers[m, 0]) ** 2 + (cy - markers[m, 1]) ** 2) ** 0.5
                for cx, cy in corners
            )
        )
    return residuals


def board_height(markers):
    """Board plane height: mean marker Z, floored at 0."""
    markers = np.asarray(markers, dtype=float)
    return float(np.max([0.0, np.mean(markers[:, 2])]))


def solve_board_pose(markers, size, bounds=DEFAULT_POSE_BOUNDS):
    """Fit ``(x, y, th)`` minimizing each marker's distance to a board corner.

    The fit is symmetric in the markers (it never reads marker IDs), so corner
    roles are determined by geometry, not by which ID lands where.
    """
    markers = np.asarray(markers, dtype=float)
    var = {"markers": markers, "size": size}
    [x, y, th], _ = curve_fit(
        _marker_distance, var, [0] * len(markers), bounds=bounds
    )
    return float(x), float(y), float(th)


def square_centers(size, x, y, th, height, square_size, x_offset=0.0, y_offset=0.0):
    """Map every square ("A1".."H8") to its base-frame XYZ center.

    ``x_offset`` / ``y_offset`` are fixed configured board offsets (settings
    ``X_OFFSET`` / ``Y_OFFSET``); they shift the whole board, not per square.
    """
    corners = board_corners(size, x + x_offset, y + y_offset, th, height)
    origin = np.array([corners[3, 0], corners[3, 1], 0.0])  # A8 corner, z=0
    rotation = Rotation.from_euler("XYZ", [0, 0, th])
    grid = {}
    for file_index, letter in enumerate(FILES):
        for number in range(1, 9):
            local = np.array(
                [
                    square_size * (-0.5 + number),
                    -square_size * (0.5 + file_index),
                    height,
                ]
            )
            grid[f"{letter}{number}"] = origin + rotation.apply(local)
    return grid
