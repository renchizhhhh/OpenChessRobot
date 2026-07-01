#!/usr/bin/env python3
"""Interactive commander for the LLM-commentary game.

Launched by hri_chess_exe.launch (enable_commentary:=true), or run via rosrun.
Drives the commentary node: publishes the post-move FEN to
/chess_board_after_robot_move (pre-cache) before each robot move, and
"fen:...,move:..." to /chess_chat (speak) after each human move.
"""

import chess
import rospy
from std_msgs.msg import String

from ocr_runtime.script_imports import prefer_source_scripts
prefer_source_scripts(__file__)
from hri_chess_commander import HRIChessCommander
from ocr_runtime.llm_commentary import format_recognized_move


class HRIChessAssistant(HRIChessCommander):
    def __init__(self, fen=chess.STARTING_FEN, cam=True):
        super().__init__(fen, cam)
        self.pub_chat = rospy.Publisher(
            "/chess_chat", String, queue_size=50, latch=False)
        self.pub_pred = rospy.Publisher(
            "/chess_board_after_robot_move", String, queue_size=50, latch=False)

    def pub_recognized_move(self, move):
        self.pub_chat.publish(format_recognized_move(self.board, move))
        rospy.loginfo("chess chat published")

    def test_human_play_analyze(self):
        human_color = chess.BLACK
        input("Press Enter when ready...")
        while not self.board.is_game_over() and not rospy.is_shutdown():
            print(self.board, "\n")

            if human_color == self.board.turn:
                rospy.loginfo("Human moves...")
                input("Press Enter when your move is done...")
                rec_board = self.observe_board()
                move = self.detect_move(self.board, rec_board, human_color)
                self.pub_recognized_move(move)
                print(f"Human move: {move}", "\n")
            else:
                rospy.loginfo("Robot moves...")
                try:
                    move = self.ext_engine.next_move(self.board)
                except Exception:
                    rospy.logwarn(
                        f"Stockfish crashed! Current board is {self.board.fen()}")
                    return ""
                next_board = self.board.copy(stack=False)
                next_board.push(chess.Move.from_uci(str(move)))
                self.pub_pred.publish(next_board.fen())
                # Let any in-flight commentary finish before the arm moves.
                while rospy.get_param('is_speaking') and not rospy.is_shutdown():
                    rospy.sleep(0.1)
                self.robot_execution(move)
                print(f"Engine move: {move}", "\n")

            if move:
                self.board.push(chess.Move.from_uci(str(move)))
            else:
                print("No move detected. Pass.")

            print(self.board.transform(chess.flip_vertical).transform(
                chess.flip_horizontal), "\n")

        if self.board.is_game_over():
            print(self.board.result())


if __name__ == "__main__":
    try:
        rospy.init_node("chess_commander", anonymous=True, log_level=rospy.DEBUG)
        my_chess = HRIChessAssistant(cam=True)
        my_chess.check_initial_state()
        my_chess.test_human_play_analyze()
    except rospy.ROSInterruptException:
        pass
