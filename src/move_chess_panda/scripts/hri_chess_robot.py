#!/home/charles/panda/panda_env310/bin/python3.10

from chess_robo_player import ChessRoboPlayer
# from utili.camera_config import Camera
from std_msgs.msg import String
from move_chess_panda.msg import Live_offset
import rospy
import random

class HRIChessRobot(ChessRoboPlayer):
    def __init__(self, cam=True):
        super().__init__(cam)

    def respond_pose_message(self, pose_msg):
        motion_acc = 0.3
        motion_vel = 0.4
        if pose_msg.data == "ready":
            self.move_ready_state()
        if pose_msg.data == "camera":
            self.move_camera_state()
        if pose_msg.data == "low":
            self.move_camera_state_low()
        if pose_msg.data == "high":
            self.move_camera_state_high()
        if pose_msg.data == "left":
            self.move_camera_state_left()
        if pose_msg.data == "right":
            self.move_camera_state_right()
        if pose_msg.data == "nod":
            self.move_camera_state_low()
            self.move_camera_state_human(acc=motion_acc, vel=motion_vel)
            self.move_camera_state_low(acc=motion_acc, vel=motion_vel)
            # self.move_camera_state_human(acc=motion_acc, vel=motion_vel)
            self.move_camera_state_low()
        if pose_msg.data == "shake":
            self.move_camera_state_low()
            self.move_camera_state_away(acc=motion_acc*2, vel=motion_vel*2)
            self.move_camera_state_away_opposite(acc=motion_acc*2, vel=motion_vel*2)
            # self.move_camera_state_away(acc=motion_acc*2, vel=motion_vel*2)
            self.move_camera_state_low()
            rospy.sleep(1)
        if pose_msg.data == "rotate":
            self.move_camera_state_low()
            self.move_camera_state_human(acc=motion_acc, vel=motion_vel)
            if random.random() < 0.5:
                self.move_camera_state_rotate(acc=motion_acc, vel=motion_vel)
            self.move_camera_state_low(acc=motion_acc, vel=motion_vel)
        if pose_msg.data == "human":
            self.move_camera_state_low()
            self.move_camera_state_human(acc=motion_acc, vel=motion_vel)
            self.move_camera_state_low(acc=motion_acc, vel=motion_vel)
        if pose_msg.data == "stare":
            self.move_camera_state_low()
            self.move_camera_state_human(acc=motion_acc, vel=motion_vel)

    def run(self):
        self.move_camera_state_high()
        rospy.sleep(0.5)
        self.all_update()
        self.init_constraints(workspace = False)
        rospy.Subscriber("/chess_move", String, self.respond_chess_message)
        rospy.Subscriber("/change_pose", String, self.respond_pose_message)
        rospy.Subscriber("/live_offset", Live_offset, self.respond_offset_message)

if __name__ == "__main__":
    try:
        rospy.init_node("chess_robo_player", anonymous=True, log_level=rospy.DEBUG)
        my_robo_player = HRIChessRobot(cam=True)
        my_robo_player.run()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
