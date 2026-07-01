#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import time
import re
import numpy as np
import click

import rospy
from std_msgs.msg import String, Bool

import chess
import chess.engine
from engine.wrapper import ChessEngineWrapper

from ocr_runtime.settings import MODE, ELO, DEPTH
from ocr_runtime.move_result import result_matches, result_is_failure

class ChessCommander(object):
    def __init__(self, fen=chess.STARTING_FEN):
        rospy.on_shutdown(self.shutdown)
        self.pub_chess_move = rospy.Publisher(
            "/chess_move", String, queue_size=50, latch=False
        )
        self.pub_change_pose = rospy.Publisher(
            "/change_pose", String, queue_size=50, latch=False
        )
        rospy.Subscriber("/chess_move_res", String, self.update_execution_value)
        # sync the execution
        self.last_move_send = ""
        self.last_move_executed = False
        self.last_move_failed = False
        # game settings
        self.board = chess.Board(fen)
        self.ext_engine = ChessEngineWrapper(mode=MODE, depth=DEPTH, elo=ELO)

    def shutdown(self):
        rospy.loginfo("Stopping the ChessCommander node")
        self.ext_engine.shutdown()
        sys.exit()

    def reset_game(self):
        self.board.reset()

    def update_execution_value(self, msg):
        res = msg.data
        rospy.loginfo("Message received: /chess_move_res")
        # Only a result that acknowledges the move we are waiting on may complete
        # it; a stale or unrelated response is ignored rather than treated as
        # success for the last sent move.
        if not result_matches(self.last_move_send, res):
            rospy.loginfo(f"ignoring result '{res}' while awaiting {self.last_move_send}")
            return
        if result_is_failure(res):
            rospy.logwarn(f"move reported failure: {self.last_move_send}")
            self.last_move_failed = True
        else:
            rospy.loginfo(f"current move executed: {self.last_move_send}")
        self.last_move_executed = True
      

    def is_hop(self, move: chess.Move) -> bool:
        """
        Check if a move from the current pickup square to `square` would be considered a knight hop.

        Args:
            square (str): destination square in capital letter notation (e.g. 'A2')

        Returns:
            bool: True if the move is considered a knight hop and there are pieces blocking the knight's path, False otherwise.
        """
        pick_row, pick_col = divmod(move.from_square, 8)
        dest_row, dest_col = divmod(move.to_square, 8)  
        #assumes 0,0 to be A0
        letters = "abcdefgh"
        numbers = "12345678"

        # Calculate the differences in row and column position between the two squares
        col_diff = abs(dest_col - pick_col)
        row_diff = abs(dest_row - pick_row)


        # If the difference in columns is 2 and the difference in rows is 1,
        # or vice versa, then the move is a legal knight move
        pass_squares = []
        if col_diff == 2 and row_diff == 1:
            # Knight moves two columns and one row
            if dest_col > pick_col:
                pass_squares.append(letters[pick_col+1] + numbers[pick_row])
                pass_squares.append(letters[pick_col+1] + numbers[dest_row])
            else:
                pass_squares.append(letters[pick_col-1] + numbers[pick_row])
                pass_squares.append(letters[pick_col-1] + numbers[dest_row])
        elif col_diff == 1 and row_diff == 2:
            # Knight moves one column and two rows
            if dest_row > pick_row:
                pass_squares.append(letters[pick_col] + numbers[pick_row+1])
                pass_squares.append(letters[dest_col] + numbers[pick_row+1])
            else:
                pass_squares.append(letters[pick_col] + numbers[pick_row-1])
                pass_squares.append(letters[dest_col] + numbers[pick_row-1])
        else:
            return False

        # Check if any of the squares the knight would pass through contain a piece
        for square in pass_squares:
            if self.board.piece_at(chess.parse_square(square)):
                return True
        return False
        
    def encode_move_message(self, pseudo_move: chess.Move, move: str):
        """takes in a chess move and a message string, 
        then encodes additional information to the message string based on the attributes of the move."""

        print (f"move in encode_move_message: {move}")
        move += "1" if self.is_hop(pseudo_move) else "0"
        rospy.loginfo(f"next move {pseudo_move} is a hop") if self.is_hop(pseudo_move) else None

        move += "1" if self.board.is_capture(pseudo_move) else "0"
        rospy.loginfo(f"next move {pseudo_move} is a capture") if self.board.is_capture(pseudo_move) else None
        
        move += "1" if self.board.is_castling(pseudo_move) else "0"
        rospy.loginfo(f"next move {pseudo_move} is castling") if self.board.is_castling(pseudo_move) else None
        
        move += "1" if self.board.is_en_passant(pseudo_move) else "0"
        rospy.loginfo(f"next move {pseudo_move} is en passant") if self.board.is_en_passant(pseudo_move) else None
        
        move += "1" if len(pseudo_move.uci()) == 5 else "0"
        rospy.loginfo(f"next move {pseudo_move} is a promotion") if len(pseudo_move.uci()) == 5 else None
        
        rospy.loginfo(f"move after encoding will be: {move}")
        return move

    def change_robot_pose(self, pose: str, duration: float = 3):
        self.pub_change_pose.publish(pose)
        if pose in ["low", "right", "left", "high"]:
            self.cur_robo_pose = pose
        time.sleep(duration)

    def robot_execution(self, move):
        """encode the move into a string with the length of 8-9:
        [move][is_capture: 0/1][is_castling: 0/1][is_en_passant: 0/1][is_promotion: 0/1]
        and send to robot for execution.
        """
        self.last_move_executed = False
        self.last_move_failed = False
        self.last_move_send = move
        pseudo_move = chess.Move.from_uci(move)
        move = self.encode_move_message(pseudo_move, move)
        self.pub_chess_move.publish(move)
        execution_trial = 0
        timed_out = False
        rospy.loginfo("waiting for execution...")
        try:
            while not self.last_move_executed and not rospy.is_shutdown():
                if execution_trial >= 40:
                    rospy.logwarn("last move not executed successfully")
                    timed_out = True
                    break
                execution_trial += 1
                rospy.sleep(0.5)
            if rospy.is_shutdown():
                self.shutdown
        except KeyboardInterrupt:
            print("interrupted!")
        success = self.last_move_executed and not self.last_move_failed and not timed_out
        self.last_move_executed = False
        return success

    def engine_move(self, execution=False, timeit=False):
        """find the move for the current board status, change the current board by the move,
        and publish a message to let the robot execute the move. Once the robot finished its
        execution, the self.last_move_executed will be changed.
        """
        try:
            engine_time = time.time()
            move = self.ext_engine.next_move(self.board)
            engine_time = time.time() - engine_time
        except Exception:
            rospy.logwarn(f"Stockfish crashed! Current board is {self.board.fen()}")
            return ""
        if execution:
            self.robot_execution(move)
        rospy.loginfo(f"engine move: {move}")
        if timeit:
            return move, engine_time
        return move

    def replay_move(self, move, execution=False):
        """change the current board by the move from replay"""
        if execution:
            self.robot_execution(move)
        self.board.push(chess.Move.from_uci(move))
        return move

    def random_move(self, execution=False):
        """generate a random move and change the current board by the move"""
        import random
        if self.board.legal_moves:
            move = random.choice(list(self.board.legal_moves)).uci()
        else:
            raise Exception("No legal moves!")
        if execution:
            self.robot_execution(move)
        self.board.push(chess.Move.from_uci(move))
        rospy.loginfo(f"random move: {move}")
        return move

    def human_cli_move(self, execution=False):
        """get the move from human player and change the current board by the move"""
        is_valid = None
        while not is_valid:
            ipt = click.prompt("Please enter a move like e2e4")
            is_valid = re.search("[a-h][1-8][a-h][1-8]", ipt)
            if not is_valid:
                click.echo("move format check failed. Please enter again")
        move = is_valid.group()
        if execution:
            self.robot_execution(move)
        self.board.push(chess.Move.from_uci(move))
        rospy.loginfo(f"human move: {move}")
        return move

    def play(self):
        human_color = click.prompt(
            "-Please choose a color by number: white:1 | black:0 \n-In standard rules, white moves first.",
            default=0,
        )
        while not self.board.is_game_over():
            print(self.board, "\n")
            for i in range(4):
                if human_color == self.board.turn:
                    move = self.human_cli_move()
                    print(f"Human move: {move}", "\n")
                else:
                    move = self.engine_move(execution=False)
                    print(f"Engine move: {move}", "\n")
                print(self.board, "\n")

    def load_chess_games(self, folder=""):
        games = list()
        folder = Path(__file__).parent.absolute().joinpath(folder)
        rospy.loginfo(f"loading at path: {folder}")
        games_list = folder.glob("*.csv")
        for file in np.sort(list(games_list)):
            games.append(
                np.genfromtxt(os.path.join(folder, file), delimiter=",", dtype=str)
            )
        rospy.loginfo(f"{len(games)} games loaded!")
        return games    

    def replay(self, path, num_game=1, num_step=10, execution=False, benchmark=False):
        games = self.load_chess_games(path)
        for game in games[num_game:]:
            print(self.board, "\n")
            for move in game[:num_step]:
                if rospy.is_shutdown():
                    raise Exception("ROS down. Stop")
                if not execution:
                    input("Press enter to save FEN and move robot...")
                self.replay_move(move, execution)
                print("**********next move**********")
                print(self.board, "\n")
                print("*****************************")
            input("Game Over! Reset the game and press enter to continue")

if __name__ == "__main__":
    try:
        rospy.init_node("chess_commander", anonymous=True)
        my_chess = ChessCommander()
        my_chess.replay(path = "data/games/Adams/", num_step=10, execution=True, num_game=0)
    except rospy.ROSInterruptException:
        pass
