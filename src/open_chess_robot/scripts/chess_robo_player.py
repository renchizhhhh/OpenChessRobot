#!/usr/bin/env python3

import time
import sys
import numpy as np
import copy
import cv2
from contextlib import contextmanager
from ocr_runtime.logger import setup_logger
from ocr_runtime.camera_config import Camera
from scipy.spatial.transform import Rotation
import rospy
import moveit_commander
import actionlib
from moveit_commander import MoveGroupCommander
from std_msgs.msg import String, Bool
from moveit_msgs.msg import (
    Constraints,
    OrientationConstraint,
    PositionConstraint,
    JointConstraint,
    BoundingVolume,
)
from geometry_msgs.msg import Pose, Vector3
from shape_msgs.msg import SolidPrimitive
from franka_gripper.msg import MoveGoal, MoveAction, GraspGoal, GraspAction, HomingGoal, HomingAction
from franka_msgs.msg import ErrorRecoveryAction, ErrorRecoveryGoal
from ocr_runtime import board_geometry
from ocr_runtime import piece_offset
from ocr_runtime.recognition_projection import camera_to_base
from ocr_runtime.move_result import encode_result
from ocr_runtime.paths import user_data_path
from ocr_runtime.move_sequences import (
    EncodedMove,
    castling_rook_move,
    en_passant_capture_drop,
    en_passant_capture_square,
    normal_capture_drop,
    parse_encoded_move,
    promotion_prompt,
)

from ocr_runtime.settings import (
    CAMERA, CAM_IP, SQUARE_SIZE,
    Z_ABOVE_BOARD, Z_TO_PIECE, Z_DROP,
    X_OFFSET, Y_OFFSET, ACC, VEL,
    CARTESIAN_SUCCESS_FRACTION, EXECUTION_RECOVERY_ATTEMPTS,
    HANDEYE_R_FLANGE2ZED, HANDEYE_T_FLANGE2ZED_LEFT, HANDEYE_T_FLANGE2ZED_RIGHT,
    LOW_CAM_JOINTS, HIGH_CAM_JOINTS, LEFT_CAM_JOINTS, RIGHT_CAM_JOINTS,
    LOOK_AT_HUMAN, LOOK_AWAY, LOOK_AWAY_R, ROTATE_LEFT, ROTATE_RIGHT,
    MARKER_SAMPLES,
    PICK_SETTLE_TIME, PICK_RADIUS_MIN_RATIO, PICK_RADIUS_MAX_RATIO,
    PICK_CIRCLE_CLAHE_CLIP, PICK_CIRCLE_HOUGH_PARAM1, PICK_CIRCLE_HOUGH_PARAM2,
)

GRASP_WIDTH = 0.035
GOALWIDTH = 0
GOALSPEED = 1
GOALFORCE = 10
PROMOTION_CONFIRMATION_PARAM = "/open_chess_robot/operator/promotion_confirmed"
# Opt-in vision-based grasp correction. When true, pick() first parks the lens
# near-nadir over the from-square, measures the piece's actual base offset and
# nudges the descend by it (see measure_pick_offset). Off by default so the
# nominal kinematic grasp is unchanged unless explicitly enabled.
PICK_VISION_REFINE_PARAM = "/open_chess_robot/pick/vision_refine"
# Reject a measured offset larger than this fraction of a square - that large a
# shift means a misdetection, not a human nudging a piece off centre.
PICK_OFFSET_MAX_RATIO = 0.6


class MoveExecutionError(Exception):
    """Raised when a motion plan could not be fully planned or executed."""


class ChessRoboPlayer(object):
    robot_acc = ACC
    robot_vel = VEL

    def __init__(self, cam=True):
        self.execution = True
        rospy.on_shutdown(self.shutdown)
        hardware_ns = "/open_chess_robot/hardware"
        self.robot_name = rospy.get_param(f"{hardware_ns}/robot", "panda")
        self.move_group = rospy.get_param(
            f"{hardware_ns}/move_group", f"{self.robot_name}_arm"
        )
        self.end_effector_link = rospy.get_param(
            f"{hardware_ns}/end_effector_link", f"{self.robot_name}_link8"
        )
        self.commander = MoveGroupCommander(self.move_group)
        self.commander.allow_replanning(True)
        self.commander.set_planner_id("RRTConnect") #RRT
        self.pub_chess_res = rospy.Publisher("/chess_move_res", String, queue_size=20, latch=False)
        # for the gripper
        self.move_client = actionlib.SimpleActionClient("/franka_gripper/move", MoveAction)
        self.grasp_client = actionlib.SimpleActionClient("/franka_gripper/grasp", GraspAction)
        self.homing_client = actionlib.SimpleActionClient('/franka_gripper/homing', HomingAction)
        # Error recovery: a single goal to this action clears a libfranka reflex
        # (e.g. an acceleration-discontinuity abort) and re-enables the
        # controller. We fire it on demand from execute_path, not continuously.
        self.recovery_client = actionlib.SimpleActionClient(
            "/franka_control/error_recovery", ErrorRecoveryAction)
        self.gripper_is_open = True
        self.ignore_grasp_error = True
        # Hand-eye transform (camera frame -> flange), configured in settings.py.
        self._R_flange2zed = HANDEYE_R_FLANGE2ZED
        if CAMERA == "left":
            self._T_flange2zed = HANDEYE_T_FLANGE2ZED_LEFT
        elif CAMERA == "right":
            self._T_flange2zed = HANDEYE_T_FLANGE2ZED_RIGHT
        # configurations to compute
        self._zed_position_world = np.zeros(3)
        self._markers_world = np.zeros((4, 3))
        self.board_grid = dict()
        self.board_corners = np.zeros((4, 3))
        # integrate camera
        self.cam_on = False
        if cam:
            self.cam_on = True
            self.camera = Camera(ip=CAM_IP, port=30000, name="1")
        # logging
        self.logger = setup_logger(
            logger_name="robot_logger",
            log_file=user_data_path("logs", "robot.txt"),
        )

    def shutdown(self):
        rospy.loginfo("Stopping the ChessRoboPlayer node")
        if hasattr(self, "cam_on") and self.cam_on:
            self.camera.close()
        # Shut down MoveIt cleanly and exit the script
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)

    def homing_gripper(self):
        self.homing_client.wait_for_server()
        self.homing_client.send_goal(HomingGoal())
        self.homing_client.wait_for_result(rospy.Duration.from_sec(5.0))

    def update_world_positions(self):
        current_pose = self.commander.get_current_pose().pose
        P = current_pose.position
        Q = current_pose.orientation

        R_arm = Rotation.from_quat([Q.x, Q.y, Q.z, Q.w])
        R_zed = Rotation.from_euler("ZXY", self._R_flange2zed, degrees=True)

        if self.cam_on and len(self.camera.detected_markers) == 4:
            markers = list()
            marker_id = list(self.camera.detected_markers.keys())
            for marker_prop in self.camera.detected_markers.values():
                markers.append(marker_prop["pos2camera"])
            rospy.loginfo("using detected markers")
        else:
            raise Exception("No markers avaliable when updating!")
        for i, marker in enumerate(markers):
            zed2flange = R_zed.apply(marker) + R_zed.apply(self._T_flange2zed)
            flange2base = R_arm.apply(zed2flange) + np.array([P.x, P.y, P.z])
            self._markers_world[i] = flange2base
            rospy.logdebug(f" marker {marker_id[i]} after {self._markers_world[i]}")

    def calculate_board_grids(self):
        """update the positions of grids in the base axis"""
        markers = self._markers_world
        size = 8 * SQUARE_SIZE
        self.board_height = board_geometry.board_height(markers)
        x, y, th = board_geometry.solve_board_pose(markers, size)  # x [m], y [m], th [rad]
        rospy.logdebug(f"fitted board position: {[x, y, th]}")

        self.board_corners = board_geometry.board_corners(
            size, x + X_OFFSET, y + Y_OFFSET, th, self.board_height
        )
        rospy.logdebug(f"Board corners: {self.board_corners}")
        # Expose the localized corners (A1, H1, H8, A8) so the touch-probe
        # calibration tool (collect_board_corners.py) can compute X/Y/Z offset
        # corrections against ground-truth gripper touches.
        rospy.set_param(
            "/open_chess_robot/board_corners", self.board_corners.tolist())

        self.board_grid = board_geometry.square_centers(
            size, x, y, th, self.board_height, SQUARE_SIZE,
            x_offset=X_OFFSET, y_offset=Y_OFFSET,
        )

    def marker_update(self, refine_pos=False, side=CAMERA):
        if self.cam_on:
            current_frame = self.camera.get_img(side)
            corners, ids = self.camera.detect_markers(frame=current_frame)
            attempt = 0
            while len(ids) != 4:
                if len(ids) < 4:
                    rospy.loginfo("marker missing, checking again")
                if len(ids) > 4:
                    rospy.loginfo(f"additional markers detected, checking again, {ids} found")
                corners, ids = self.camera.detect_markers(frame=self.camera.get_img(side))
                attempt += 1
                rospy.sleep(0.05)
                if attempt > 20:
                    raise ValueError(f"expected 4 markers but found markers {ids} in the current image.")
            self.camera.locate_markers(corners, ids)
        if refine_pos:
            self.refine_marker_pose()

    def refine_marker_pose(self):
        avg_detected_markers = self.camera.detected_markers.copy()
        valid_res = max(1, int(rospy.get_param(
            "/open_chess_robot/localization/marker_samples", MARKER_SAMPLES)))
        for _ in range(valid_res - 1):
            self.marker_update(refine_pos=False)
            for marker_id in avg_detected_markers.keys():
                avg_detected_markers[marker_id]["pos2camera"] += self.camera.detected_markers[marker_id]["pos2camera"]
        for marker_id in avg_detected_markers.keys():
            avg_detected_markers[marker_id]["pos2camera"] /= valid_res
        self.camera.detected_markers = avg_detected_markers

    def all_update(self):
        self.marker_update(refine_pos=True)
        self.update_world_positions()
        self.calculate_board_grids()

    def workspace_constraint(self):
        # http://docs.ros.org/en/jade/api/moveit_commander/html/classmoveit__commander_1_1move__group_1_1MoveGroupCommander.html#ad7f6d93d73bf43268ba983afb0dc4f23
        # https://ros-planning.github.io/moveit_tutorials/doc/planning_with_approximated_constraint_manifolds/planning_with_approximated_constraint_manifolds_tutorial.html
        # https://ros-planning.github.io/moveit_tutorials/doc/motion_planning_api/motion_planning_api_tutorial.html
        # https://cram-system.org/tutorials/intermediate/collisions_and_constraints
        # https://python.hotexamples.com/examples/shape_msgs.msg/SolidPrimitive/-/python-solidprimitive-class-examples.html
        # https://answers.ros.org/question/174095/end-effector-pose-constrained-planning/
        # https://groups.google.com/g/moveit-users/c/rPaxNoawSFc

        box_constraint = PositionConstraint()
        box_constraint.header = self.commander.get_current_pose().header
        box_constraint.link_name = self.end_effector_link
        box_constraint.target_point_offset = Vector3(0, 0, 0.15)

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [1.2, 1.0, 1.0]

        pose = Pose()
        pose.position.x = 0.5
        pose.position.y = 0
        pose.position.z = 0.5 + self.board_height
        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = 0
        pose.orientation.w = 1

        bounds = BoundingVolume()
        bounds.primitives = [box]
        bounds.primitive_poses = [pose]
        box_constraint.constraint_region = bounds
        box_constraint.weight = 1

        self.constraints.position_constraints.append(box_constraint)

    def upright_constraints(self):
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header = self.commander.get_current_pose().header
        orientation_constraint.link_name = self.end_effector_link
        orientation_constraint.orientation.x = 0.9238795
        orientation_constraint.orientation.y = -0.3826834
        orientation_constraint.orientation.z = 0
        orientation_constraint.orientation.w = 0
        orientation_constraint.absolute_x_axis_tolerance = 0.05
        orientation_constraint.absolute_y_axis_tolerance = 0.05
        orientation_constraint.absolute_z_axis_tolerance = 0.05  # =3.14 ignores this axis
        orientation_constraint.weight = 1

        self.constraints.orientation_constraints.append(orientation_constraint)

        joint_constraints = [
            JointConstraint(
                joint_name=f"{self.robot_name}_joint1", position=0, tolerance_above=0.785398, tolerance_below=0.785398, weight=0.5
            ),
            JointConstraint(
                joint_name=f"{self.robot_name}_joint7",
                position=0.785398,
                tolerance_above=0.785398,
                tolerance_below=0.785398,
                weight=0.5,
            ),
        ]

        self.constraints.joint_constraints.extend(joint_constraints)

    def init_constraints(self, workspace=True, upright=True):
        self.commander.clear_path_constraints()
        self.constraints = Constraints()
        self.constraints.name = "robot contraints"
        if workspace:
            self.workspace_constraint()
        if upright:
            self.upright_constraints()

    @contextmanager
    def _motion_lock(self):
        """Hold the global is_moving flag for the duration of a motion.

        Using try/finally guarantees the flag is cleared even if planning or
        execution raises, so an error cannot leave the system wedged as
        permanently "moving" (which would hang the commander's wait loops).
        """
        rospy.set_param('is_moving', True)
        try:
            yield
        finally:
            rospy.set_param('is_moving', False)

    def move_ready_state(self, acc=None, vel=None):
        with self._motion_lock():
            if acc is None:
                acc = ChessRoboPlayer.robot_acc
            if vel is None:
                vel = ChessRoboPlayer.robot_vel
            self.move_gripper("open")
            self.commander.set_named_target("ready")
            self.commander.set_max_acceleration_scaling_factor(acc)
            self.commander.set_max_velocity_scaling_factor(vel)
            self.commander.go(wait=True)

    def move_camera_state(self, acc=None, vel=None, joint_position=LOW_CAM_JOINTS):
        with self._motion_lock():
            if acc is None:
                acc = ChessRoboPlayer.robot_acc
            if vel is None:
                vel = ChessRoboPlayer.robot_vel
            joint_goal = joint_position[0:7]
            self.commander.set_max_acceleration_scaling_factor(acc)
            self.commander.set_max_velocity_scaling_factor(vel)
            self.commander.go(joint_goal, wait=True)

    def move_camera_state_low(self, acc=None, vel=None):
        self.move_camera_state(acc, vel, joint_position=LOW_CAM_JOINTS)

    def move_camera_state_high(self, acc=None, vel=None):
        self.move_camera_state(acc, vel, joint_position=HIGH_CAM_JOINTS)

    def move_camera_state_left(self, acc=None, vel=None):
        self.move_camera_state(acc, vel, joint_position=LEFT_CAM_JOINTS)

    def move_camera_state_right(self, acc=None, vel=None):
        self.move_camera_state(acc, vel, joint_position=RIGHT_CAM_JOINTS)

    def move_camera_state_human(self, acc=None, vel=None):
        self.move_camera_state(acc, vel, joint_position=LOOK_AT_HUMAN)

    def move_camera_state_away(self, acc=None, vel=None):
        self.move_camera_state(acc, vel, joint_position=LOOK_AWAY)

    def move_camera_state_away_opposite(self, acc=None, vel=None):
        self.move_camera_state(acc, vel, joint_position=LOOK_AWAY_R)

    def move_camera_state_rotate(self, acc=None, vel=None):
        self.move_camera_state(acc, vel, joint_position=ROTATE_LEFT)

    def recover_from_error(self, timeout=5.0):
        """Send one error-recovery goal to clear a libfranka reflex and re-enable
        the controller, then wait for the result. Returns True if the action
        reported done within ``timeout``. Best-effort: a missing action server is
        logged and treated as a no-op rather than raising."""
        if not self.recovery_client.wait_for_server(rospy.Duration.from_sec(2.0)):
            rospy.logwarn("Error-recovery action server unavailable; cannot recover")
            return False
        rospy.logwarn("Sending franka error-recovery goal")
        self.recovery_client.send_goal(ErrorRecoveryGoal())
        finished = self.recovery_client.wait_for_result(
            rospy.Duration.from_sec(timeout))
        if not finished:
            rospy.logwarn("Error-recovery did not complete within %.1fs", timeout)
        return finished

    def execute_path(self, waypoints, acc=None, vel=None, constrain=True):
        with self._motion_lock():
            if acc is None:
                acc = ChessRoboPlayer.robot_acc
            if vel is None:
                vel = ChessRoboPlayer.robot_vel
            self.commander.set_max_velocity_scaling_factor(value=vel)
            self.commander.set_max_acceleration_scaling_factor(value=acc)
            recoveries = max(0, int(rospy.get_param(
                "/open_chess_robot/execution/recovery_attempts",
                EXECUTION_RECOVERY_ATTEMPTS)))
            tries = recoveries + 1
            for attempt in range(1, tries + 1):
                fraction = 0.0
                maxattempts = 10
                plan_tries = 0
                while fraction < 1.0 and plan_tries < maxattempts:
                    if constrain:
                        (plan, fraction) = self.commander.compute_cartesian_path(
                            waypoints, 0.01, 0.0, path_constraints=self.constraints
                        )
                    else:
                        (plan, fraction) = self.commander.compute_cartesian_path(waypoints, 0.01, 0.0)
                    plan_tries += 1
                if fraction < CARTESIAN_SUCCESS_FRACTION:
                    raise MoveExecutionError(
                        f"Cartesian path only {fraction:.2f} planned after {plan_tries} attempts "
                        f"(need {CARTESIAN_SUCCESS_FRACTION}); refusing to run a partial path"
                    )
                plan = self.commander.retime_trajectory(
                    self.commander.get_current_state(),
                    plan,
                    velocity_scaling_factor=vel,
                    acceleration_scaling_factor=acc,
                    algorithm="time_optimal_trajectory_generation",
                )
                if self.commander.execute(plan, wait=True):
                    return
                rospy.logwarn(
                    "Trajectory execution failed (attempt %s/%s)", attempt, tries)
                if attempt < tries:
                    rospy.logwarn(
                        "Recovering and retrying (recovery %s/%s)", attempt, recoveries)
                    self.recover_from_error()
            raise MoveExecutionError(
                f"Trajectory execution reported failure after {tries} attempts "
                f"({recoveries} recoveries)")

    def move_gripper(self, command="open"):
        self.move_client.wait_for_server()
        if command == "close":
            goal = MoveGoal(width=-GRASP_WIDTH, speed=GOALSPEED)
        elif command == "open":
            goal = MoveGoal(width=GRASP_WIDTH, speed=GOALSPEED)
        else:
            raise ValueError("grasp goal should be close or open")
        self.move_client.send_goal(goal)
        self.move_client.wait_for_result(rospy.Duration.from_sec(5.0))
        return self.move_client.get_result().success

    def grasp(self, command):
        self.grasp_client.wait_for_server()
        if command == "close":
            assert self.gripper_is_open, "gripper is already closed!"
            goal = GraspGoal(width=GOALWIDTH, speed=GOALSPEED, force=GOALFORCE)
            goal.epsilon.inner = goal.epsilon.outer = 0.1

            self.grasp_client.send_goal(goal)
            self.grasp_client.wait_for_result(rospy.Duration.from_sec(5.0))
            res = self.grasp_client.get_result().success
            self.gripper_is_open = False
        elif command == "open":
            assert not self.gripper_is_open, "gripper is already open!"
            res = self.move_gripper("open")
            self.gripper_is_open = True
        else:
            raise ValueError("grasp goal should be close or open")
        if not self.ignore_grasp_error and not res:
            raise rospy.ROSInterruptException(f"{command} failed")
        return res

    def _look_down_ee_pose(self, target_xyz, height, gripper_yaw_deg):
        """EE (link8) pose with the gripper pointing straight down and the
        active lens centred ``height`` above ``target_xyz``.

        The hand-eye rotation ``[45, 2.5, 0]`` is almost a pure roll about the
        optical axis (only ~2.5 deg of tilt), so a gripper-straight-down pose
        already looks essentially nadir; we keep that known-reachable
        orientation and only translate the flange so the *lens* - not the
        flange - sits above the square. The lens centre in world is
        ``R_arm.apply(R_zed.apply(T)) + P`` (from ``update_world_positions``),
        so ``P = lens_centre - R_arm.apply(R_zed.apply(T))``.
        """
        R_zed = Rotation.from_euler("ZXY", self._R_flange2zed, degrees=True)
        phi = np.radians(gripper_yaw_deg)
        x8 = np.array([np.cos(phi), np.sin(phi), 0.0])
        z8 = np.array([0.0, 0.0, -1.0])              # gripper approach: down
        y8 = np.cross(z8, x8)                         # right-handed link8 frame
        R_arm = Rotation.from_matrix(np.column_stack([x8, y8, z8]))
        lens_offset = (R_arm * R_zed).apply(self._T_flange2zed)  # flange->lens
        lens_centre = np.array(
            [target_xyz[0], target_xyz[1], target_xyz[2] + height])
        P = lens_centre - lens_offset
        q = R_arm.as_quat()                          # x, y, z, w
        pose = self.commander.get_current_pose().pose
        pose.position.x, pose.position.y, pose.position.z = P
        (pose.orientation.x, pose.orientation.y,
         pose.orientation.z, pose.orientation.w) = q
        return pose

    def look_down_square(self, square, height=0.3, gripper_yaw=-45):
        """Move so the active lens looks straight down at the centre of
        ``square`` and STAY there - a near-nadir view for measuring the piece's
        grasp offset without perspective. Does not return to ready.

        The gripper stays straight down (reachable, like a pick). ``gripper_yaw``
        is the link8 yaw about the vertical; -45 deg cancels the Franka hand's
        45 deg mount offset so the fingers sit square with the board. The yaw
        barely affects reachability, but 90 deg rotations keep the hand aligned,
        so fall back through them if the first does not plan. Returns the gripper
        yaw (degrees) used - needed to rotate pixel offsets into the base frame.
        """
        yaw_candidates = [gripper_yaw, gripper_yaw + 90,
                          gripper_yaw - 90, gripper_yaw + 180]
        target = self.board_grid[square]
        with self._motion_lock():
            self.commander.set_max_velocity_scaling_factor(self.robot_vel)
            self.commander.set_max_acceleration_scaling_factor(self.robot_acc)
            for yaw in yaw_candidates:
                pose = self._look_down_ee_pose(target, height, yaw)
                self.commander.set_pose_target(pose)
                reached = self.commander.go(wait=True)
                self.commander.stop()
                self.commander.clear_pose_targets()
                if reached:
                    rospy.loginfo(
                        f"look_down_square {square}: gripper yaw {yaw} deg "
                        f"reachable, lens {height:.2f} m above board")
                    return yaw
                rospy.logdebug(f"look_down_square {square}: yaw {yaw} unreachable")
        raise MoveExecutionError(
            f"No reachable look-down pose for {square} over yaws {yaw_candidates}")

    def measure_pick_offset(self, square: str):
        """Vision-based grasp correction for ``square`` (base-frame dx, dy metres).

        Parks the active lens near-nadir over the square (``look_down_square``),
        captures one frame, detects the centre square and the piece's round base,
        and returns the board-plane offset of the piece from the square centre via
        differential back-projection. Because both pixels are projected through
        the same live camera pose, the hand-eye error is common-mode and cancels,
        so the result is hand-eye-independent to first order.

        Returns ``np.array([dx, dy])`` in metres, or ``None`` if the camera is
        off, the look pose is unreachable, or either detection fails (fail-open:
        the caller then descends on the nominal square centre).
        """
        if not self.cam_on:
            return None
        try:
            self.look_down_square(square)
        except MoveExecutionError as exc:
            rospy.logwarn(f"pick offset {square}: look pose unreachable ({exc})")
            return None

        rospy.sleep(rospy.get_param(
            "/open_chess_robot/pick/settle_time", PICK_SETTLE_TIME))
        for _ in range(3):
            img = self.camera.get_img(CAMERA)
        self.camera._load_calibration(CAMERA)
        K = np.asarray(self.camera.camera_matrix, dtype=np.float64)
        dist = np.asarray(self.camera.dist_coeff, dtype=np.float64).reshape(-1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        debug_path = str(user_data_path("calibration", f"pick_offset_{square}.png"))

        sq = piece_offset.detect_center_square(gray)
        if sq is None:
            # Save the raw frame so a square miss can be diagnosed afterwards.
            cv2.imwrite(debug_path, img)
            rospy.logwarn(f"pick offset {square}: centre square not detected "
                          f"(debug image: {debug_path})")
            return None
        corners, sq_centre, side_px = sq
        # Base-circle thresholds default from settings.py but stay live-tunable:
        # a missed detection can be fixed with `rosparam set` instead of a code
        # edit. The check_piece_offset.py bench tool shares the same defaults.
        circ_centre, radius = piece_offset.detect_base_circle(
            gray, corners, sq_centre, side_px,
            r_min_ratio=rospy.get_param(
                "/open_chess_robot/pick/radius_min_ratio", PICK_RADIUS_MIN_RATIO),
            r_max_ratio=rospy.get_param(
                "/open_chess_robot/pick/radius_max_ratio", PICK_RADIUS_MAX_RATIO),
            clahe_clip=rospy.get_param(
                "/open_chess_robot/pick/circle_clahe_clip", PICK_CIRCLE_CLAHE_CLIP),
            hough_param1=rospy.get_param(
                "/open_chess_robot/pick/circle_hough_param1", PICK_CIRCLE_HOUGH_PARAM1),
            hough_param2=rospy.get_param(
                "/open_chess_robot/pick/circle_hough_param2", PICK_CIRCLE_HOUGH_PARAM2))
        # Always save: the annotated square (and the base, if found) is the
        # evidence needed to tune a missed detection.
        vis = piece_offset.draw_offset_annotation(
            img, corners, sq_centre, circ_centre, radius)
        cv2.imwrite(debug_path, vis)
        if circ_centre is None:
            rospy.logwarn(f"pick offset {square}: piece base not detected "
                          f"(square ok, side {side_px:.1f} px; "
                          f"debug image: {debug_path})")
            return None

        pose = self.commander.get_current_pose().pose
        quat = [pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w]
        flange_xyz = np.array(
            [pose.position.x, pose.position.y, pose.position.z])
        R_cam, cam_centre = camera_to_base(
            quat, self._R_flange2zed, self._T_flange2zed, flange_xyz)
        board_z = float(self.board_corners[:, 2].mean())
        offset = piece_offset.differential_offset_base(
            sq_centre, circ_centre, R_cam, cam_centre, K, dist, board_z)
        rospy.loginfo(
            f"pick offset {square}: dx {offset[0]*1000:+.1f} mm, "
            f"dy {offset[1]*1000:+.1f} mm (debug image: {debug_path})")
        return offset

    def pick(self, square: str, x_offset=0, y_offset=0):
        """pick a piece at a certain square from current pose

        Args:
            square (str): square in captial such as A2
        """
        path = []
        start_position = self.board_grid[square]
        rospy.loginfo(f"picking piece at the square: {square}")

        # Optional vision-based grasp correction: nudge the descend toward the
        # piece's actual position. Fail-open - any detection failure leaves the
        # offset at zero so the nominal kinematic grasp is used.
        if rospy.get_param(PICK_VISION_REFINE_PARAM, False):
            offset = self.measure_pick_offset(square)
            if offset is not None:
                cap = PICK_OFFSET_MAX_RATIO * SQUARE_SIZE
                if abs(offset[0]) <= cap and abs(offset[1]) <= cap:
                    x_offset += float(offset[0])
                    y_offset += float(offset[1])
                else:
                    rospy.logwarn(
                        f"pick offset {square}: |{np.round(offset, 4)}| exceeds "
                        f"cap {cap:.3f} m; using nominal grasp")

        # plan for start location
        start_pose = self.commander.get_current_pose().pose
        start_pose.position.x = start_position[0] + x_offset
        start_pose.position.y = start_position[1] + y_offset
        start_pose.position.z = start_position[2] + Z_ABOVE_BOARD
        path.append(copy.deepcopy(start_pose))

        start_pose.position.z -= Z_TO_PIECE
        path.append(copy.deepcopy(start_pose))
        self.execute_path(path)
        self.grasp("close")
        rospy.loginfo(f"{square} picked! waiting...")

        self.pickup_square = square

    def place(self, square: str, x_offset=0, y_offset=0, capture=False, ishigh=False, release=True):
        """pick a piece at a certain square from current pose

        Args:
            square (str): square in captial such as A2
        """
        path = []
        place_position = self.board_grid[square]
        rospy.loginfo(f"placing piece at the square: {square}")

        # plan the placing movement
        start_pose = self.commander.get_current_pose().pose
        if ishigh or capture:
            start_pose.position.z = Z_ABOVE_BOARD
            path.append(copy.deepcopy(start_pose))
        # 2 move
        start_pose.position.x = place_position[0] + x_offset
        start_pose.position.y = place_position[1] + y_offset
        if ishigh or capture:
            start_pose.position.z = place_position[2] + Z_ABOVE_BOARD
        else:
            start_pose.position.z = place_position[2] + Z_ABOVE_BOARD - Z_TO_PIECE + Z_DROP
        path.append(copy.deepcopy(start_pose))
        # 3 down
        if not capture:
            if ishigh:
                start_pose.position.z -= Z_TO_PIECE - Z_DROP  # place higher
                path.append(copy.deepcopy(start_pose))
            # 4 open
            self.execute_path(path)
            if release:
                self.grasp("open")
                # 5 up
                start_pose = self.commander.get_current_pose().pose
                start_pose.position.z = place_position[2] + Z_ABOVE_BOARD
                self.execute_path([start_pose])
        else:
            self.execute_path(path)
            if release:
                self.grasp("open")
        # placed/dropped piece
        rospy.loginfo(f"{square} placed! waiting...")

    def wait_for_operator_confirmation(self, message: str):
        rospy.set_param(PROMOTION_CONFIRMATION_PARAM, False)
        confirm_command = f"rosparam set {PROMOTION_CONFIRMATION_PARAM} true"
        prompt = (
            f"\033[91m{message}\n"
            f"After manual replacement, press Enter here or run:\n"
            f"{confirm_command}\033[0m"
        )
        print(prompt)
        rospy.logwarn(f"{message} Confirm with: {confirm_command}")

        if sys.stdin is not None and sys.stdin.isatty():
            try:
                input("Press Enter after the promotion piece has been replaced...")
                rospy.set_param(PROMOTION_CONFIRMATION_PARAM, True)
            except EOFError:
                pass

        while (
            not rospy.is_shutdown()
            and not rospy.get_param(PROMOTION_CONFIRMATION_PARAM, False)
        ):
            rospy.sleep(0.25)
        rospy.loginfo("Operator confirmed promotion replacement")

    def execute_promotion(self, decoded_move: EncodedMove):
        self.wait_for_operator_confirmation(promotion_prompt(decoded_move))

    def execute_castling(self, start: str, end: str):
        rook_start, rook_end = castling_rook_move(start, end)
        self.pick(rook_start)
        self.place(rook_end, ishigh=True)

    def execute_en_passant(self, start: str, end: str):
        enemy_pawn = en_passant_capture_square(start, end)
        drop = en_passant_capture_drop()
        self.pick(enemy_pawn)
        self.place(
            square=drop.square,
            x_offset=drop.x_offset,
            y_offset=drop.y_offset,
            capture=True,
        )

    def execute_capture(self, target: str):
        drop = normal_capture_drop()
        self.pick(target)
        self.place(
            square=drop.square,
            x_offset=drop.x_offset,
            y_offset=drop.y_offset,
            capture=True,
        )

    def execute_chess_move(self, move: str):
        decoded_move = parse_encoded_move(move)
        self.move_ready_state()
        execution_time = time.time()
        start = decoded_move.start
        end = decoded_move.end

        rospy.loginfo(
            "promotion: %s, en passant: %s, capture: %s, hop: %s",
            int(decoded_move.is_promotion),
            int(decoded_move.is_en_passant),
            int(decoded_move.is_capture),
            int(decoded_move.is_hop),
        )
        rospy.loginfo(f"executing current move: {move}")
        # capture
        capture_time = 0
        if decoded_move.is_capture and not decoded_move.is_en_passant:
            capture_time = time.time()
            self.execute_capture(end)
            capture_time = time.time() - capture_time

        # normal move
        pick_time = time.time()
        self.pick(start)
        pick_time = time.time() - pick_time
        place_time = time.time()
        self.place(end, ishigh=decoded_move.is_hop)
        place_time = time.time() - place_time

        # castling
        castling_time = 0
        if decoded_move.is_castling:
            castling_time = time.time()
            self.execute_castling(start, end)
            castling_time = time.time() - castling_time
        # en_passant
        en_passant_time = 0
        if decoded_move.is_en_passant:
            en_passant_time = time.time()
            self.execute_en_passant(start, end)
            en_passant_time = time.time() - en_passant_time
        # promotion
        promotion_time = 0
        if decoded_move.is_promotion:
            promotion_time = time.time()
            self.execute_promotion(decoded_move)
            promotion_time = time.time() - promotion_time
        execution_time = time.time() - execution_time
        rospy.loginfo(f"{move} executed! Resume to ready state")
        self.logger.info(f'move: {move}; execution time: {execution_time}; capture_time: {capture_time}; pick_time: {pick_time}; place_time: {place_time}; castling_time: {castling_time}; en_passant_time: {en_passant_time}; promotion_time: {promotion_time}')
        self.move_camera_state_low()

    def respond_chess_message(self, chess_move_msg):
        move = chess_move_msg.data
        while rospy.get_param('is_moving'):
            rospy.sleep(0.1)
        try:
            self.execute_chess_move(move)
        except Exception as exc:
            rospy.logerr(f"move execution failed for {move}: {exc}")
            self.pub_chess_res.publish(encode_result(move, success=False))
            return
        self.pub_chess_res.publish(encode_result(move, success=True))

    def run(self):
        self.move_camera_state_high()
        rospy.sleep(0.5)
        self.all_update()
        self.init_constraints()
        rospy.Subscriber("/chess_move", String, self.respond_chess_message)

if __name__ == "__main__":
    try:
        rospy.init_node("chess_robo_player", anonymous=True, log_level=rospy.INFO)
        robo = ChessRoboPlayer(cam=True)
        robo.run()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
