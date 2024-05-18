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


class ChessRoboPlayer(object):


    def __init__(self, cam=True):
        self.execution = True
        rospy.on_shutdown(self.shutdown)
        self.commander = MoveGroupCommander(ROBOT + "_arm")
        self.commander.allow_replanning(True)
        self.commander.set_planner_id("RRT")
        self.pub_chess_res = rospy.Publisher("/chess_move_res", String, queue_size=20, latch=False)
        # for the gripper
        self.move_client = actionlib.SimpleActionClient("/franka_gripper/move", MoveAction)
        self.grasp_client = actionlib.SimpleActionClient("/franka_gripper/grasp", GraspAction)
        self.homing_client = actionlib.SimpleActionClient('/franka_gripper/homing', HomingAction)

        
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