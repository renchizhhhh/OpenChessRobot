#!/home/charles/panda/panda_env310/bin/python3.10

from chess_robo_player import ChessRoboPlayer
from utili.camera_config import Camera
from std_msgs.msg import String, Bool
import rospy
import cv2
import os
from pathlib import Path
from datetime import datetime
import numpy as np


def create_data_dir(folder):
    current_time = datetime.now().strftime("%m_%d_%H_%M")
    data_dir = f"data/collect/{current_time}/{folder}"
    data_dir = os.path.join(Path(__file__).resolve().parent, data_dir)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    return data_dir


class ChessRoboCam(ChessRoboPlayer):
    def __init__(self, multipose=False, cam=True):
        super().__init__(cam)
        self.counter = 0
        self.multipose = multipose
        create_data_dir(folder="label")
        self.data_dir = create_data_dir(folder="img")
        rospy.set_param("data_dir",self.data_dir)

    def save_marker_pos(self, path):
        file = os.path.join(path, "marker.txt")
        with open(file, "w") as f:
            for i, d in self.camera.detected_markers.items():
                f.write(f"{i},{d['pos2img']}")
        f.close()
        rospy.loginfo(f"marker is written")

    def save_img(self, path, subfix="", multipose=False):
        img = self.camera.get_img()
        if subfix:
            file = os.path.join(path, f"{self.counter:02d}_{subfix}.png")
        else:
            file = os.path.join(path, f"{self.counter:02d}.png")
        cv2.imwrite(file, img)
        if not self.multipose:
            self.counter += 1

    def take_photo(self, cam_pose="low"):
        if cam_pose=="high":
            self.move_camera_state_high(acc=0.8)
        elif cam_pose=="low":
            self.move_camera_state_low()
        elif cam_pose=="left":
            self.move_camera_state_left()
        elif cam_pose=="right":
            self.move_camera_state_right()
        else:
            raise Exception(f"Invalid camera pose '{cam_pose}'")
        self.save_img(self.data_dir, subfix=cam_pose)
        rospy.sleep(0.5)

    def take_multi_photo(self):
        # camera_poses = ["high", "low", "left", "right"]
        camera_poses = ["low"]
        for pose in camera_poses:
            self.take_photo(pose)
        if self.multipose:
            self.counter += 1       

    def execute_chess_move(self, move: str):
        super(ChessRoboCam, self).execute_chess_move(move)
        self.take_multi_photo()
        rospy.loginfo(f"img {self.counter:2d} is written")

    def respond_cam_pose(self, pose_msg):
        if pose_msg.data:
            self.take_multi_photo()

    def run(self):
        self.move_camera_state_high()
        self.all_update()
        self.init_constraints(workspace=False)
        input("Press Enter to start...Don't forget the rocks!")
        self.save_marker_pos(path=create_data_dir(folder="label"))
        rospy.Subscriber("/chess_move", String, self.respond_chess_message)
        # rospy.Subscriber("/camera_ready", Bool, self.respond_cam_pose)
        self.take_multi_photo()
        self.move_camera_state_low()

if __name__ == "__main__":
    try:
        rospy.init_node("chess_robo_player", anonymous=True, log_level=rospy.INFO)
        my_robo_cam = ChessRoboCam(multipose=True, cam=True)
        my_robo_cam.run()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
