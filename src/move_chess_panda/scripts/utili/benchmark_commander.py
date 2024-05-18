#!/home/charles/panda/panda_env310/bin/python3.10

from hri_chess_commander import HRIChessCommander
from data_collect_chess_robot import create_data_dir
import rospy
import chess
import os
from pathlib import Path
from std_msgs.msg import String, Bool


class ChessCommanderBenchmark(HRIChessCommander):
    def __init__(self, fen=chess.STARTING_FEN):
        super().__init__(fen)
        self.counter = 0
        self.pub_camera_ready = rospy.Publisher(
            "/camera_ready", Bool, queue_size=50, latch=False
        )
        self.label_dir = create_data_dir(folder="label")
        self.pred_dir = create_data_dir(folder="predict")

    def save_fen(self, path, fen, sufix=""):
        if not sufix:
            file = os.path.join(path, f"{self.counter:02d}.txt")
        else:
            file = os.path.join(path, f"{self.counter:02d}_{sufix}.txt")
        with open(file, 'w') as f:
            f.write(fen)
        f.close()
        rospy.loginfo(f"label {self.counter} is written")

    # def multi_onestep_recog(self, img):
    #     cog_board = self.onestep_recog(
    #         img=img, recognizer=self.chess_rec, color=chess.WHITE
    #     )

    def recognize_move(self, next_turn=chess.WHITE, update_board=True):
        """recognize the move by a human player and update the current board

        Args:
            next_turn (chess.COLOR, optional): the turn on the predict fen. Defaults to chess.WHITE.
            update_board (bool, optional): use the prediction to update the current board or not. Defaults to True.

        Returns:
            [str | chess.BOARD]: return the inferred move or the recognized chess board
        """
        is_valid = False
        failed_time = 0
        single_step = True
        img = self.request_new_img()
        while not is_valid and not rospy.is_shutdown():
            if single_step:
                cog_board = self.onestep_recog(
                    img=img, recognizer=self.chess_rec, color=chess.WHITE
                )
            else:
                img = self.request_new_img()
                cog_board = self.twostep_recog(
                    img=img, recognizer=self.chess_rec, color=chess.WHITE
                )
            # NOTE: not remembering the castling right
            if next_turn == chess.BLACK:
                cog_board.turn = chess.BLACK
                rospy.loginfo(f"Look for the turn: {cog_board.fen()}")
            cog_board.castling_rights = cog_board.clean_castling_rights()
            if cog_board.status() == chess.Status.VALID:
                is_valid = True
                rospy.loginfo(f"Current board status: {cog_board.status().name}")
                rospy.loginfo(f"Current FEN: {cog_board.fen()}")
            elif cog_board.status() == chess.Status.TOO_MANY_KINGS:
                single_step = False
                failed_time += 1
                rospy.loginfo(f"Found: {cog_board.status().name}")
                rospy.loginfo(f"Current FEN: {cog_board.fen()}")
            else:
                failed_time += 1
                rospy.loginfo(f"Invalid state: {cog_board.status().name}")
                rospy.loginfo(f"Current FEN: {cog_board.fen()}")

            if failed_time == 5:
                self.change_robot_pose("left")
                img = self.request_new_img()
            if failed_time == 10:
                self.change_robot_pose("right")
                img = self.request_new_img()
            if failed_time == 15:
                self.change_robot_pose("low")
                rospy.logwarn(
                    f"predicted board is not valid after {failed_time} trails"
                )
                print("*************************")
                print(cog_board)
                print("*************************")
                break
            rospy.sleep(0.2)
        if update_board:
            move = self.infer_move(self.board, cog_board)
            self.board = cog_board
            rospy.loginfo(f"recognized human move: {move}")
            return move
        else:
            return cog_board

    def save_both_fen(self):
        self.save_fen(self.label_dir, self.board.fen())
        recog_fen = self.recognize_move(update_board=False).fen()
        self.save_fen(self.pred_dir, recog_fen, from_recog=True)
        self.counter += 1

    def replay(self, path=str(Path(__file__).parent / "data/games/Adams"), num_game=2, num_step=99, execution=False):
        games = self.load_chess_games(path)
        for game in games[:num_game]:
            print(self.board, "\n")
            self.save_both_fen()
            for move in game[:num_step]:
                if rospy.is_shutdown():
                    raise Exception("ROS down. Stop")
                if not execution:
                    input("Press enter to save FEN and move robot...")
                #self.pub_camera_ready.publish(True)
                print("waiting for taking photo")
                self.replay_move(move, execution)
                print("**********next move**********")
                print(self.board, "\n")
                print("*****************************")
                self.save_both_fen()
            input("Game Over! Reset the game and press enter to continue")

if __name__ == "__main__":
    try:
        rospy.init_node("chess_commander", anonymous=True)
        my_chess = ChessCommanderBenchmark()
        my_chess.replay(num_step=100, execution=True)
    except rospy.ROSInterruptException:
        pass
