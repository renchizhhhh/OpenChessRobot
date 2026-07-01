"""Project the localized 3D board markers into a camera image.

This lets recognition reuse the geometry already solved at the high localization
pose instead of re-detecting ArUco markers at the shallow low pose, where they
decode unreliably. The transform mirrors the runtime's forward marker->world
math in ``ChessRoboPlayer.update_world_positions``:

    world = (R_arm * R_zed).apply(p_cam) + (R_arm * R_zed).apply(T) + P

inverted here into the world->camera ``rvec``/``tvec`` that ``cv2.projectPoints``
expects. No ROS dependency, so the math is unit tested off-robot.
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def camera_to_base(arm_quat_xyzw, r_flange2zed_zxy_deg, t_flange2zed, flange_xyz):
    """Return (R_cam2base, t_cam2base) for the camera pose in the base frame.

    ``arm_quat_xyzw`` is the flange orientation (x, y, z, w); ``flange_xyz`` its
    position. ``r_flange2zed_zxy_deg`` / ``t_flange2zed`` are the hand-eye
    rotation (ZXY Euler degrees) and translation for the active lens.
    """
    R_arm = Rotation.from_quat(list(arm_quat_xyzw))
    R_zed = Rotation.from_euler("ZXY", list(r_flange2zed_zxy_deg), degrees=True)
    R_cam2base = R_arm * R_zed
    t_cam2base = R_cam2base.apply(np.asarray(t_flange2zed, dtype=float)) + np.asarray(
        flange_xyz, dtype=float
    )
    return R_cam2base, t_cam2base


def world_to_camera_rtvec(arm_quat_xyzw, r_flange2zed_zxy_deg, t_flange2zed, flange_xyz):
    """Return (rvec, tvec) mapping base-frame points into the camera frame."""
    R_cam2base, t_cam2base = camera_to_base(
        arm_quat_xyzw, r_flange2zed_zxy_deg, t_flange2zed, flange_xyz
    )
    R_base2cam = R_cam2base.inv()
    rvec = R_base2cam.as_rotvec()
    tvec = -R_base2cam.apply(t_cam2base)
    return rvec, tvec


def project_markers(
    markers_world,
    arm_quat_xyzw,
    r_flange2zed_zxy_deg,
    t_flange2zed,
    flange_xyz,
    camera_matrix,
    dist_coeff,
):
    """Project (N,3) base-frame marker positions to (N,2) image pixels."""
    markers_world = np.asarray(markers_world, dtype=float).reshape(-1, 3)
    rvec, tvec = world_to_camera_rtvec(
        arm_quat_xyzw, r_flange2zed_zxy_deg, t_flange2zed, flange_xyz
    )
    dist = None if dist_coeff is None else np.asarray(dist_coeff, dtype=float)
    pts, _ = cv2.projectPoints(
        markers_world, rvec, tvec, np.asarray(camera_matrix, dtype=float), dist
    )
    return pts.reshape(-1, 2)
