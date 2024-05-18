#!/home/charles/panda/panda_env310/bin/python3.10

import time
import numpy as np
import copy
import cv2
from utili import board_fit
from utili.logger import setup_logger 
from utili.camera_config import Camera
from scipy.spatial.transform import Rotation
from scipy.optimize import curve_fit
import rospy
import moveit_commander
import actionlib
from moveit_commander import MoveGroupCommander
from std_msgs.msg import String, Bool
from move_chess_panda.msg import Live_offset
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

from setup_configurations import (
    ROBOT,
    CAMERA, CAM_IP, SQUARE_SIZE,
    Z_ABOVE_BOARD, Z_TO_PIECE, Z_DROP,
    X_OFFSET, Y_OFFSET, ACC, VEL,
    LOW_CAM_JOINTS, HIGH_CAM_JOINTS, LEFT_CAM_JOINTS, RIGHT_CAM_JOINTS,
    LOOK_AT_HUMAN, LOOK_AWAY, LOOK_AWAY_R, ROTATE_LEFT, ROTATE_RIGHT
)

GRASP_WIDTH = 0.035
GOALWIDTH = 0
GOALSPEED = 1
GOALFORCE = 10

class ChessRoboPlayer(object):
    robot_acc = ACC
    robot_vel = VEL

    def __init__(self, cam=True):
        self.execution = True
        rospy.on_shutdown(self.shutdown)
        self.commander = MoveGroupCommander(ROBOT + "_arm")
        self.commander.allow_replanning(True)
        self.commander.set_planner_id("RRTConnect") #RRT
        self.pub_chess_res = rospy.Publisher("/chess_move_res", String, queue_size=20, latch=False)
        # for the gripper
        self.move_client = actionlib.SimpleActionClient("/franka_gripper/move", MoveAction)
        self.grasp_client = actionlib.SimpleActionClient("/franka_gripper/grasp", GraspAction)
        self.homing_client = actionlib.SimpleActionClient('/franka_gripper/homing', HomingAction)
        self.gripper_is_open = True
        self.ignore_grasp_error = True
        # predefined configurations
        # NOTE: double check
        self._R_flange2zed = [45, 2.5, 0]
        if CAMERA == "left":
            self._T_flange2zed = [-0.06, -0.06, 0.02]
        elif CAMERA == "right":
            self._T_flange2zed = [0.06, -0.06, 0.02]
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
        self.logger = setup_logger(logger_name="robot_logger", log_file="/home/charles/panda/catkin_ws/src/move_chess_panda/scripts/logs/robot.txt")
        # rosparams
        rospy.set_param('x_offset', X_OFFSET)
        rospy.set_param('y_offset', Y_OFFSET)


    def shutdown(self):
        rospy.loginfo("Stopping the ChessRoboPlayer node")
        if hasattr(self, "cam_on") and self.cam_on:
            self.camera.close()
        # Shut down MoveIt cleanly and exit the script
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)

    def homing_gripper(self):
        homing_client = actionlib.SimpleActionClient('/franka_gripper/homing', HomingAction)
        homing_client.wait_for_server()
        homing_client.send_goal(HomingGoal())
        homing_client.wait_for_result(rospy.Duration.from_sec(5.0))

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
        self.board_hight = np.max([0, np.mean(markers[:, 2])])
        size = 8 * SQUARE_SIZE
        var = {"markers": markers, "size": size}
        [x, y, th], _ = curve_fit(
            board_fit.distance, var, [0] * len(markers), bounds=([0, -1, -1], [2, 1, 1])
        )  # x [m], y [m], th [rad]
        rospy.logdebug(f"fitted board position: {[x,y,th]}")

        x_offset = rospy.get_param('x_offset')
        y_offset = rospy.get_param('y_offset')
        self.board_corners = board_fit.board(size, x + x_offset, y + y_offset, th, self.board_hight)
        rospy.logdebug(f"Board corners: {self.board_corners}")
        R_board = Rotation.from_euler("XYZ", [0, 0, th])

        numbers = [1, 2, 3, 4, 5, 6, 7, 8]
        letters = ["A", "B", "C", "D", "E", "F", "G", "H"]

        for letter in letters:
            for number in numbers:
                square = [
                    SQUARE_SIZE * (-0.5 + number),
                    -SQUARE_SIZE * (0.5 + letters.index(letter)),
                    self.board_hight,
                ]
                square_world = [
                    self.board_corners[3, 0],
                    self.board_corners[3, 1],
                    0,
                ] + R_board.apply(square)
                key = letter + str(number)
                self.board_grid[key] = square_world

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
        valid_res = 10
        for _ in range(valid_res - 1):
            self.marker_update(refine_pos=False)
            for id in avg_detected_markers.keys():
                avg_detected_markers[id]["pos2camera"] += self.camera.detected_markers[id]["pos2camera"]
        for id in avg_detected_markers.keys():
            avg_detected_markers[id]["pos2camera"] /= valid_res
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
        box_constraint.link_name = ROBOT + "_link8"
        box_constraint.target_point_offset = Vector3(0, 0, 0.15)

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [1.2, 1.0, 1.0]

        pose = Pose()
        pose.position.x = 0.5
        pose.position.y = 0
        pose.position.z = 0.5 + self.board_hight
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
        orientation_constraint.link_name = ROBOT + "_link8"
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
                joint_name=f"{ROBOT}_joint1", position=0, tolerance_above=0.785398, tolerance_below=0.785398, weight=0.5
            ),
            JointConstraint(
                joint_name=f"{ROBOT}_joint7",
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

        # self.commander.set_path_constraints(self.constraints)

    def move_ready_state(self, acc=None, vel=None):
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

    def execute_path(self, waypoints, acc=None, vel=None):
        if acc is None:
            acc = ChessRoboPlayer.robot_acc
        if vel is None:
            vel = ChessRoboPlayer.robot_vel
        # self.commander.set_max_velocity_scaling_factor(value=vel)
        # self.commander.set_max_acceleration_scaling_factor(value=acc)
        fraction = 0.0
        maxattempts = 10
        attempts = 0
        while fraction < 1.0 and attempts < maxattempts:
            (plan, fraction) = self.commander.compute_cartesian_path(
                waypoints, 0.01, 0.0, path_constraints=self.constraints
            )
            attempts += 1
        plan = self.commander.retime_trajectory(
            self.commander.get_current_state(),
            plan,
            velocity_scaling_factor=vel,
            acceleration_scaling_factor=acc,
            algorithm="time_optimal_trajectory_generation",
        )
        self.commander.execute(plan, wait=True)

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

    def pick(self, square: str, x_offset=0, y_offset=0):
        """pick a piece at a certain square from current pose

        Args:
            square (str): square in captial such as A2
        """
        path = []
        start_position = self.board_grid[square]
        rospy.loginfo(f"picking piece at the square: {square}")

        # plan for start location
        start_pose = self.commander.get_current_pose().pose
        start_pose.position.x = start_position[0]
        start_pose.position.y = start_position[1]
        start_pose.position.z = start_position[2] + Z_ABOVE_BOARD
        path.append(copy.deepcopy(start_pose))

        start_pose.position.z -= Z_TO_PIECE
        path.append(copy.deepcopy(start_pose))
        self.execute_path(path)
        self.grasp("close")
        rospy.loginfo(f"{square} picked! waiting...")

        self.pickup_square = square

    def place(self, square: str, x_offset=0, y_offset=0, capture=False, ishigh=False):
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
            self.grasp("open")
            # 5 up
            start_pose = self.commander.get_current_pose().pose
            start_pose.position.z = place_position[2] + Z_ABOVE_BOARD
            self.execute_path([start_pose])
        else:
            self.execute_path(path)
            self.grasp("open")
        # placed/dropped piece
        rospy.loginfo(f"{square} placed! waiting...")

    def execute_promotion(self, move: str):
        pass

    def execute_castling(self, start: str, end: str):
        assert start[1] == end[1], f"{start} to {end} is not valid castling"
        if ord(start[0]) > ord(end[0]):
            # direction: h >>> a
            rock_start = "A" + start[1]
            rock_end = chr(ord(end[0]) + 1) + start[1]
            self.pick(rock_start)
            self.place(rock_end, ishigh=True)
        else:
            # direction: a >>> h
            rock_start = "H" + start[1]
            rock_end = chr(ord(end[0]) - 1) + start[1]
            self.pick(rock_start)
            self.place(rock_end, ishigh=True)

    def execute_en_passant(self, start: str, end: str):
        enemy_pawn = end[0] + start[1]
        self.pick(enemy_pawn)
        self.place(square="H4", y_offset=-0.2, capture=True)

    def execute_capture(self, target: str, waste: str = "H4"):
        self.pick(target)
        self.place(square=waste, y_offset=-0.1, capture=True)

    def execute_chess_move(self, move: str):
        self.move_ready_state()
        execution_time = time.time()
        start = move.upper()[:2]
        end = move.upper()[2:4]
        is_hop = int(move[-5])
        is_capture = int(move[-4])
        is_castling = int(move[-3])
        is_en_passant = int(move[-2])
        is_promotion = int(move[-1])

        rospy.loginfo(f"promotion: {is_promotion}, en passant: {is_en_passant}, capture: {is_capture}, hop: {is_hop}")
        rospy.loginfo(f"executing current move: {move}")
        # capture
        capture_time = 0
        if is_capture:
            capture_time = time.time()
            self.execute_capture(end)
            capture_time = time.time() - capture_time

        # normal move
        pick_time = time.time()
        self.pick(start)
        pick_time = time.time() - pick_time
        place_time = time.time()
        self.place(end, ishigh=is_hop)
        place_time = time.time() - place_time

        # castling
        castling_time = 0
        if is_castling:
            castling_time = time.time()
            self.execute_castling(start, end)
            castling_time = time.time() - castling_time
        # en_passant
        en_passant_time = 0
        if is_en_passant:
            en_passant_time = time.time()
            self.execute_en_passant(start, end)
            en_passant_time = time.time() - en_passant_time
        # promotion
        if is_promotion:
            raise NotImplementedError("promotion function is not completed!")
        execution_time = time.time() - execution_time
        rospy.loginfo(f"{move} executed! Resume to ready state")
        self.logger.info(f'move: {move}; execution time: {execution_time}; capture_time: {capture_time}; pick_time: {pick_time}; place_time: {place_time}; castling_time: {castling_time}; en_passant_time: {en_passant_time}')
        self.move_camera_state_low()

    def respond_chess_message(self, chess_move_msg):
        self.execute_chess_move(chess_move_msg.data)
        self.pub_chess_res.publish(f"{chess_move_msg.data} is finished")

    def respond_offset_message(self, live_offset_msg):
        print(f"received offset: {live_offset_msg}")
        x_offset, y_offset = live_offset_msg.x_offset, live_offset_msg.y_offset
        rospy.set_param('x_offset', x_offset)
        rospy.set_param('y_offset', y_offset)
        self.calculate_board_grids()

    def run(self):
        self.move_camera_state_high()
        rospy.sleep(0.5)
        self.all_update()
        self.init_constraints()
        rospy.Subscriber("/chess_move", String, self.respond_chess_message)
        rospy.Subscriber("/live_offset", Live_offset, self.respond_offset_message)

if __name__ == "__main__":
    try:
        rospy.init_node("chess_robo_player", anonymous=True, log_level=rospy.INFO)
        robo = ChessRoboPlayer(cam=True)
        robo.run()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
