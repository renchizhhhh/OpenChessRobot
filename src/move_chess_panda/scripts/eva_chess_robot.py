#!/home/charles/panda/panda_env310/bin/python3.10

from hri_chess_robot import HRIChessRobot
# from utili.camera_config import Camera
from std_msgs.msg import String
from move_chess_panda.msg import Live_offset
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
        rospy.Subscriber("/change_speed", Live_offset, self.respond_speed_message)
        rospy.Subscriber("/live_offset", Live_offset, self.respond_offset_message)

    def respond_speed_message(self, speed_msg):
        print(f"received speed: {speed_msg}")
        ChessRoboPlayer.robot_acc, ChessRoboPlayer.robot_vel = speed_msg.x_offset, speed_msg.y_offset

if __name__ == "__main__":
    try:
        rospy.init_node("chess_robo_player", anonymous=True, log_level=rospy.DEBUG)
        my_robo_player = EvaChessRobot(cam=True)
        my_robo_player.run()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
