#!/usr/bin/env python3
"""Touch-probe board-corner calibration ("cheating" ground truth).

Hand-guide the gripper so its tip touches each of the four outer board corners
(where the playing-area edges meet) and press Enter to record the TCP pose. From
the four ground-truth points this prints the measured board pose, its tilt, and
suggested X/Y/Z localization offsets (the current offset plus the residual
between the touched corners and the camera-localized corners), then saves a YAML.

This NEVER commands motion - it only reads tf when you press Enter, so run it
with the arm in free-drive / hand-guiding mode. Run it AFTER the robot node has
localized the board (so /open_chess_robot/board_corners is on the param server),
in a terminal where stdin works:

    rosrun open_chess_robot collect_board_corners.py

The four corners may be touched in any order; correspondence to the camera
corners is resolved by nearest neighbour.
"""

from ocr_runtime.script_imports import prefer_source_scripts
prefer_source_scripts(__file__)

import numpy as np
import rospy
import tf2_ros
import yaml

from ocr_runtime.paths import user_data_path
from ocr_runtime.settings import X_OFFSET, Y_OFFSET

N_CORNERS = 4


def _order_like(reference, other):
    """Reorder ``other`` (N,3) so each row is nearest its ``reference`` row."""
    reference = np.asarray(reference, float)
    other = np.asarray(other, float)
    used, order = set(), []
    for r in reference:
        dists = [np.inf if j in used else float(np.linalg.norm(other[j] - r))
                 for j in range(len(other))]
        j = int(np.argmin(dists))
        used.add(j)
        order.append(j)
    return other[order]


def _lookup_tcp(tf_buffer, base_frame, tcp_frame):
    tf = tf_buffer.lookup_transform(
        base_frame, tcp_frame, rospy.Time(0), rospy.Duration(2.0))
    t = tf.transform.translation
    return np.array([t.x, t.y, t.z])


def main():
    rospy.init_node("collect_board_corners", anonymous=True)
    base_frame = rospy.get_param("~base_frame", "panda_link0")
    tcp_frame = rospy.get_param("~tcp_frame", "panda_hand_tcp")

    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(1.0)  # let the tf buffer fill

    print(f"\nRecording TCP frame '{tcp_frame}' in '{base_frame}'.")
    print("Hand-guide the gripper tip onto each outer board corner, any order.\n")

    measured = []
    i = 0
    while i < N_CORNERS and not rospy.is_shutdown():
        try:
            input(f"  Corner {i + 1}/{N_CORNERS}: touch it, then press Enter "
                  f"(Ctrl-C to abort)...")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return
        try:
            p = _lookup_tcp(tf_buffer, base_frame, tcp_frame)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException) as exc:
            print(f"    tf lookup failed ({exc}); try again.")
            continue
        measured.append(p)
        print(f"    recorded: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")
        i += 1

    if len(measured) < N_CORNERS:
        return
    pts = np.array(measured)

    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    normal = vh[2]
    tilt = float(np.degrees(np.arccos(
        min(1.0, abs(normal[2]) / np.linalg.norm(normal)))))
    print(f"\nMeasured corners centroid: ({centroid[0]:.4f}, {centroid[1]:.4f}, "
          f"{centroid[2]:.4f}); board tilt from vertical: {tilt:.2f} deg")

    result = {
        "measured_corners": [p.tolist() for p in pts],
        "centroid": centroid.tolist(),
        "tilt_deg": tilt,
    }

    localized = rospy.get_param("/open_chess_robot/board_corners", None)
    if localized is None:
        print("\nNo /open_chess_robot/board_corners on the param server - is the "
              "robot node up and the board localized? Saved measurements only.")
    else:
        loc = _order_like(pts, np.asarray(localized, float))
        residual = pts - loc  # measured (truth) minus camera localization
        dx, dy, dz = residual[:, 0].mean(), residual[:, 1].mean(), residual[:, 2].mean()
        print("\nPer-corner residual (measured - localized), mm:")
        for k, r in enumerate(residual):
            print(f"  corner {k}: ({r[0] * 1000:+.1f}, {r[1] * 1000:+.1f}, "
                  f"{r[2] * 1000:+.1f})")
        print("\nSuggested settings (current offset + mean residual):")
        print(f"  X_OFFSET = {X_OFFSET + dx:+.4f}   (current {X_OFFSET:+.4f}, "
              f"shift {dx * 1000:+.1f} mm)")
        print(f"  Y_OFFSET = {Y_OFFSET + dy:+.4f}   (current {Y_OFFSET:+.4f}, "
              f"shift {dy * 1000:+.1f} mm)")
        print(f"  Z offset = {dz:+.4f}   (mean board-height correction; not yet "
              f"applied by the grid)")
        result["localized_corners"] = loc.tolist()
        result["residual_m"] = residual.tolist()
        result["suggested"] = {
            "X_OFFSET": float(X_OFFSET + dx),
            "Y_OFFSET": float(Y_OFFSET + dy),
            "Z_offset": float(dz),
        }

    path = user_data_path("calibration", "board_corners.yaml")
    with open(path, "w") as handle:
        yaml.safe_dump(result, handle, default_flow_style=False)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
