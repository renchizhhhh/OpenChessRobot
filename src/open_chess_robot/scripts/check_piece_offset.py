#!/usr/bin/env python3
"""Measure pick alignment from the look pose (isolated, no motion).

Run this with the robot ALREADY parked at a ``look <SQ>`` pose (e.g. ``look A8``)
so the active left lens is centred straight above the square's centre. The
script opens its own left ZED stream (the sender allows multiple receivers, so it
coexists with the running robot node) and never commands motion. Re-run after
shifting / replacing a piece to iterate; an annotated image is saved each time.

The detection/geometry core lives in
``ocr_runtime.piece_offset`` and is shared with the robot node's
``ChessRoboPlayer.measure_pick_offset``; this script is the standalone bench
for it.

Two modes (``_mode`` rosparam):

  calibrate_pick (default) - PURE CV, no kinematics for the detection. Detect the
    centre square of the grid, mask to it, fit the piece's round BASE circle
    inside, and report the offset between the square centre and the base-circle
    centre. The offset is reported on the board plane (base frame) via
    differential back-projection - the same number the grasp is nudged by.

  check_pick - localization check. Same CV square centre, but compared against
    the kinematic descend point (the red cross: FK + hand-eye projection of where
    the gripper would actually go down). The offset is the localization error.
    No piece is needed; if one is present it is ignored.

    rosrun open_chess_robot check_piece_offset.py                      # calibrate
    rosrun open_chess_robot check_piece_offset.py _mode:=check_pick    # check
"""

from ocr_runtime.script_imports import prefer_source_scripts
prefer_source_scripts(__file__)

import cv2
import numpy as np
import rospy
import tf2_ros

from ocr_runtime.paths import user_data_path
from ocr_runtime.settings import (
    CAM_IP,
    SQUARE_SIZE,
    HANDEYE_R_FLANGE2ZED,
    HANDEYE_T_FLANGE2ZED_LEFT,
    PICK_RADIUS_MIN_RATIO,
    PICK_RADIUS_MAX_RATIO,
    PICK_CIRCLE_CLAHE_CLIP,
    PICK_CIRCLE_HOUGH_PARAM1,
    PICK_CIRCLE_HOUGH_PARAM2,
)
from ocr_runtime.recognition_projection import (
    camera_to_base,
    project_markers,
)
from ocr_runtime.piece_offset import (
    detect_center_square,
    detect_base_circle,
    backproject,
    differential_offset_base,
    draw_offset_annotation,
)

BASE_FRAME = "panda_link0"
FLANGE_FRAME = "panda_link8"


def _detect_center_square_configured(gray):
    """Apply ROS-tunable validation thresholds to centre-square detection."""
    return detect_center_square(
        gray,
        pitch_tol=rospy.get_param("~grid_pitch_tolerance", 0.08),
        aspect_tol=rospy.get_param("~grid_aspect_tolerance", 0.25),
        center_tol=rospy.get_param("~grid_center_tolerance", 0.45),
        canny_low=rospy.get_param("~grid_canny_low", 35),
        canny_high=rospy.get_param("~grid_canny_high", 100),
    )


def _detect_base_circle_configured(gray, corners, sq_centre, side_px):
    return detect_base_circle(
        gray, corners, sq_centre, side_px,
        r_min_ratio=rospy.get_param("~radius_min_ratio", PICK_RADIUS_MIN_RATIO),
        r_max_ratio=rospy.get_param("~radius_max_ratio", PICK_RADIUS_MAX_RATIO),
        clahe_clip=rospy.get_param("~circle_clahe_clip", PICK_CIRCLE_CLAHE_CLIP),
        hough_param1=rospy.get_param("~circle_hough_param1", PICK_CIRCLE_HOUGH_PARAM1),
        hough_param2=rospy.get_param("~circle_hough_param2", PICK_CIRCLE_HOUGH_PARAM2))


def _flange_pose(tf_buffer):
    tf = tf_buffer.lookup_transform(
        BASE_FRAME, FLANGE_FRAME, rospy.Time(0), rospy.Duration(2.0))
    q = tf.transform.rotation
    t = tf.transform.translation
    return [q.x, q.y, q.z, q.w], np.array([t.x, t.y, t.z])


def _board_plane_z(cam_centre, default_height):
    """Board-plane z in the base frame, from the localized corners if available."""
    if rospy.has_param("/open_chess_robot/board_corners"):
        return float(np.asarray(
            rospy.get_param("/open_chess_robot/board_corners"), float)[:, 2].mean())
    return cam_centre[2] - rospy.get_param("~look_height", default_height)


def _save(name, vis):
    out = user_data_path("calibration", name)
    cv2.imwrite(str(out), vis)
    print(f"  annotated image saved to {out}")


def run_calibrate_pick(img, tf_buffer):
    """Pure-CV piece-vs-square offset, reported on the board plane (base frame)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sq = _detect_center_square_configured(gray)
    if sq is None:
        print("Could not detect the centre square.")
        _save("calibrate_pick.png", img)
        return
    corners, sq_centre, side_px = sq
    mm_per_px = (SQUARE_SIZE * 1000.0) / side_px

    circ_centre, radius = _detect_base_circle_configured(
        gray, corners, sq_centre, side_px)
    if circ_centre is None:
        print("Detected the square but no base circle.")

    print("\n=== calibrate_pick (pure CV) ===")
    print(f"  square centre px : ({sq_centre[0]:.1f}, {sq_centre[1]:.1f}); "
          f"side {side_px:.1f} px -> {mm_per_px:.3f} mm/px")
    if circ_centre is not None:
        off_px = circ_centre - sq_centre
        print(f"  base circle px   : ({circ_centre[0]:.1f}, {circ_centre[1]:.1f})"
              f" r={radius:.1f}")
        print(f"  offset px        : ({off_px[0]:+.1f}, {off_px[1]:+.1f}) px")
        quat, flange_xyz = _flange_pose(tf_buffer)
        R_cam, cam_centre = camera_to_base(
            quat, HANDEYE_R_FLANGE2ZED, HANDEYE_T_FLANGE2ZED_LEFT, flange_xyz)
        board_z = _board_plane_z(cam_centre, 0.30)
        off = differential_offset_base(
            sq_centre, circ_centre, R_cam, cam_centre, K, dist, board_z) * 1000.0
        print(f"  offset base      : x {off[0]:+.1f} mm, y {off[1]:+.1f} mm "
              f"(|d|={np.linalg.norm(off):.1f})  board plane z {board_z:.4f}")

    vis = draw_offset_annotation(img, corners, sq_centre, circ_centre, radius)
    _save("calibrate_pick.png", vis)


def run_check_pick(img, tf_buffer):
    """Localization check: CV square centre vs kinematic descend point."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sq = _detect_center_square_configured(gray)
    if sq is None:
        print("Could not detect the centre square.")
        _save("check_pick.png", img)
        return
    corners, sq_centre, _ = sq

    quat, flange_xyz = _flange_pose(tf_buffer)
    R_cam, cam_centre = camera_to_base(
        quat, HANDEYE_R_FLANGE2ZED, HANDEYE_T_FLANGE2ZED_LEFT, flange_xyz)
    board_z = _board_plane_z(cam_centre, 0.20)

    # The square sits on the board plane; back-project its detected centre there.
    sq_base = backproject(sq_centre, R_cam, cam_centre, K, dist, board_z)
    offset = sq_base[:2] - cam_centre[:2]   # CV square centre - descend point

    print("\n=== check_pick (localization) ===")
    print(f"  descend xy (red) : {np.round(cam_centre[:2], 4)}")
    print(f"  CV square centre : {np.round(sq_base[:2], 4)}  (board z {board_z:.4f})")
    print(f"  BASE offset      : x {offset[0]*1000:+.1f} mm, y {offset[1]*1000:+.1f}"
          f" mm  (|d|={np.linalg.norm(offset)*1000:.1f})")

    vis = img.copy()
    cv2.polylines(vis, [corners.astype(np.int32)], True, (0, 255, 0), 2)
    cv2.drawMarker(vis, tuple(sq_centre.astype(int)), (0, 255, 0),
                   cv2.MARKER_TILTED_CROSS, 24, 2)
    descend_px = project_markers(
        [[cam_centre[0], cam_centre[1], board_z]], quat, HANDEYE_R_FLANGE2ZED,
        HANDEYE_T_FLANGE2ZED_LEFT, flange_xyz, K, dist)[0]
    cv2.drawMarker(vis, (int(descend_px[0]), int(descend_px[1])), (0, 0, 255),
                   cv2.MARKER_CROSS, 24, 2)
    _save("check_pick.png", vis)


def main():
    # Keep the ZED SDK dependency out of the pure CV helpers so recorded images
    # and synthetic grids can be tested on a machine without camera hardware.
    from ocr_runtime.camera_config import Camera

    rospy.init_node("check_piece_offset", anonymous=True)
    mode = rospy.get_param("~mode", "calibrate_pick")

    camera = Camera(ip=CAM_IP, port=30000, name="offset")
    img = camera.get_img("left")
    camera._load_calibration("left")
    global K, dist
    K = np.asarray(camera.camera_matrix, dtype=np.float64)
    dist = np.asarray(camera.dist_coeff, dtype=np.float64).reshape(-1)

    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(1.0)

    if mode == "check_pick":
        run_check_pick(img, tf_buffer)
    elif mode == "calibrate_pick":
        run_calibrate_pick(img, tf_buffer)
    else:
        print(f"Unknown _mode '{mode}'; use calibrate_pick or check_pick.")

    camera.close()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
