#!/home/charles/panda/panda_env310/bin/python3.10

from chess_commander import ChessCommander
import rospy
import chess
import os
import glob
import numpy as np
from pathlib import Path
from std_msgs.msg import String, Bool
from pynput.keyboard import Key, Listener, KeyCode
import time

class ChessCommanderFen(ChessCommander):
    def __init__(self, fen=chess.STARTING_FEN):
        super().__init__(fen)
        self.counter = 0
        self.pause_execution = False
        self.pub_camera_ready = rospy.Publisher(
            "/camera_ready", Bool, queue_size=50, latch=False
        )
        self.data_dir = rospy.get_param("data_dir")[:-3] + "label"

    def on_press(self, key):
        if key.char == 'q':
            self.pause_execution = not self.pause_execution
            if self.pause_execution:
                print("Paused. Press 'q' to resume.")
            else:
                print("Resumed.")

    def save_fen(self, path):
        fen = self.board.fen()
        file = os.path.join(path, f"{self.counter:02d}.txt")
        with open(file, 'w') as f:
            f.write(fen)
        f.close()
        rospy.loginfo(f"label {self.counter} is written")
        self.counter += 1   

    @classmethod
    def parse_pgn(cls, file="Adams.pgn"):
        with open(file) as pgn:
            for i, _ in enumerate(pgn):
                cur_game = chess.pgn.read_game(pgn)  # a generator
                name = str(i) + ".csv"
                moves = list()
                for move in cur_game.mainline_moves():
                    moves.append(move)
                np.savetxt(name, moves, delimiter=",", fmt="% s")

    def load_chess_games(self, folder=""):
        games = list()
        folder = str(Path(__file__).parent.absolute())+folder
        rospy.loginfo(f"loading at path: {folder}")
        games_list = glob.glob(os.path.join(folder, "*.csv"))
        for file in np.sort(games_list):
            games.append(
                np.genfromtxt(os.path.join(folder, file), delimiter=",", dtype=str)
            )
        rospy.loginfo(f"{len(games)} games loaded!")
        return games     

    def replay(self, path, start_id=0, num_step=100, execution=False):
        listener = Listener(on_press=self.on_press)
        listener.start()

        games = self.load_chess_games(path)
        for game in games[start_id:]:
            print(self.board, "\n")
            for move in game[:num_step]:
                # Check the pause state
                while self.pause_execution:
                    time.sleep(0.1)  # Sleep a bit to prevent busy waiting
                # If the key is pressed, pause here untill it's pressed again.                 
                if rospy.is_shutdown():
                    raise Exception("ROS down. Stop")
                if not execution:
                    input("Press enter to save FEN and move robot...")
                self.save_fen(self.data_dir)
                #self.pub_camera_ready.publish(True)
                print("waiting for taking photo")
                self.replay_move(move, execution)
                print("**********next move**********")
                print(self.board, "\n")
                print("*****************************")
            input("Game Over! Reset the game and press enter to continue")

if __name__ == "__main__":
    try:
        rospy.init_node("chess_commander", anonymous=True)
        my_chess = ChessCommanderFen()
        my_chess.replay(path = "/data/games/Adams/", num_step=1000, execution=True, start_id=4)
    except rospy.ROSInterruptException:
        pass
