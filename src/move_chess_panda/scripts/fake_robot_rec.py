#!/home/charles/panda/panda_env310/bin/python3.10

import threading
import time
import chess
import rospy
from std_msgs.msg import String, Bool

import speech_recognition as sr
from utili.logger import setup_logger

from hri_chess_commander import HRIChessCommander
import numpy as np

if __name__ == "__main__":
    try:
        rospy.init_node("chess_commander", anonymous=True, log_level=rospy.DEBUG)
        pub_chat = rospy.Publisher("/chess_chat", String, queue_size=50, latch=False)
        pub_pose = rospy.Publisher("/change_pose", String,  queue_size=50, latch=False)

        board = chess.Board()
        while not board.is_game_over() and not rospy.is_shutdown():
            move = input()
            
        pub_chat.publish()
    except rospy.ROSInterruptException:
        pass