#!/usr/bin/env python3

from ocr_runtime.script_imports import prefer_source_scripts
prefer_source_scripts(__file__)
from hri_chess_robot import HRIChessRobot
from std_msgs.msg import String, Float32MultiArray
from chess_robo_player import ChessRoboPlayer
import rospy

class EvaChessRobot(HRIChessRobot):

    def __init__(self, cam=True):
        super().__init__(cam)

    def run(self):
        self.move_camera_state_high()
        rospy.sleep(0.5)
        self.all_update()
        self.init_constraints(workspace = False)
        rospy.Subscriber("/chess_move", String, self.respond_chess_message)
        rospy.Subscriber("/change_pose", String, self.respond_pose_message)
        rospy.Subscriber("/change_speed", Float32MultiArray, self.respond_speed_message)

    def respond_speed_message(self, speed_msg):
        # data = [acceleration_scale, velocity_scale]
        print(f"received speed: {speed_msg.data}")
        ChessRoboPlayer.robot_acc, ChessRoboPlayer.robot_vel = speed_msg.data[0], speed_msg.data[1]

if __name__ == "__main__":
    try:
        rospy.init_node("chess_robo_player", anonymous=True, log_level=rospy.DEBUG)
        my_robo_player = EvaChessRobot(cam=True)
        my_robo_player.run()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
