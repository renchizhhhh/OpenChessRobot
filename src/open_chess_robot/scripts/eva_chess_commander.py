#!/usr/bin/env python3

from ocr_runtime.script_imports import prefer_source_scripts
prefer_source_scripts(__file__)
from hri_chess_commander import HRIChessCommander
import rospy
import chess
import time

import numpy as np
from std_msgs.msg import Float32MultiArray

from ocr_runtime.settings import (
    ACC,
    VEL,
)

class EvaChessCommander(HRIChessCommander):
    def __init__(self, fen=chess.STARTING_FEN):
        super().__init__(fen)
        self.is_fix2hop = False
        self.is_noding = False
        self.is_gazing = False
        self.speed_pub = rospy.Publisher("/change_speed", Float32MultiArray, queue_size=50, latch=False)

    def change_behavior(self, variation):
        # variation: ['speed', 'pickandplace', 'posture', 'gaze', 'speech']
        if variation == "speed":
            # data = [acceleration_scale, velocity_scale]
            self.speed_pub.publish(Float32MultiArray(data=[0.2, 0.2]))
        elif variation == "pickandplace":
            self.is_fix2hop = True
        elif variation == "posture":
            self.is_noding = True
        elif variation == "gaze":
            self.is_gazing = True

    def resume_behavior(self, variation):
        # variation: ['speed', 'pickandplace', 'posture', 'gaze', 'speech']
        if variation == "speed":
            self.speed_pub.publish(Float32MultiArray(data=[ACC, VEL]))
        elif variation == "pickandplace":
            self.is_fix2hop = False
        elif variation == "posture":
            self.is_noding = False
        elif variation == "gaze":
            self.is_gazing = False

    def robot_execution(self, move, fix2hop):
        """encode the move into a string with the length of 8-9:
        [move: xxxx][is_hop: 0/1][is_capture: 0/1][is_castling: 0/1][is_en_passant: 0/1][is_promotion: 0/1]
        and send to robot for execution.
        """
        self.last_move_executed = False
        self.last_move_send = move
        pseudo_move = chess.Move.from_uci(move)
        move = self.encode_move_message(pseudo_move, move)
        # change the pick and place style
        if fix2hop:
            move = move[:4] + '1' + move[5:]
        self.pub_chess_move.publish(move)
        execution_trial = 0
        rospy.loginfo("waiting for execution...")
        try:
            while not self.last_move_executed and not rospy.is_shutdown():
                if execution_trial >= 40:
                    rospy.logwarn("last move not executed successfully")
                    break
                execution_trial += 1
                rospy.sleep(0.5)
            if rospy.is_shutdown():
                self.shutdown
        except KeyboardInterrupt:
            print("interrupted!")
        self.last_move_executed = False
        # NOTE: right now always success
        return True

    def test_evaluation(self, execution=True):
        latin_square = {'1':['A','B','D','C'], '2':['B','C','A','D'], '3':['C','D','B','A'], '4':['D','A','C','B']}
        variations = {'A':'default', 'B':'pickandplace', 'C':'posture', "D":'speed'}
        repetition = 3
        robot_counter = 0

        human_color = chess.BLACK
        move_stack = []
        score_stack = []
        evaluation_stack = []

        resume_the_state = False
        participant_group = input("Input the participant group [1-4]: ")
        order = latin_square[participant_group]
        print(f"current order: {order}")
        current_condition = 'default'
        # while not self.board.is_game_over() and not rospy.is_shutdown():
        while not rospy.is_shutdown():
            print(self.board, "\n")
            detect_time = np.nan
            evaluate_time = np.nan
            human_time = np.nan
            if human_color == self.board.turn:
                rospy.set_param("is_waiting", True)
                rospy.loginfo("Human moves...")
                human_time = time.time()
                input("Press Enter when your move is done...")
                human_time = time.time() - human_time
                detect_time = time.time()
                move = str(self.detect_move(self.board, self.observe_board(), human_color))
                detect_time = time.time() - detect_time
                if not move:
                    print("No move detected. Will check again after one second")
                    continue
                evaluate_time = time.time()
                info = self.ext_engine.evaluate_move(self.board, move)
                score, is_mate = self.ext_engine.get_score_from_info(info)
                evaluate_time = time.time() - evaluate_time
                score *= -1
                msg = f"Human move: {move}; \n"
                msg += f"human_time: {human_time}, detect_time: {detect_time}, evaluate_time: {evaluate_time} \n"
            else:
                # change robot behavior
                if robot_counter % repetition == 0:
                    self.resume_behavior(current_condition)
                    current_condition = variations[order.pop(0)]
                    self.change_behavior(current_condition)
                # activate [speech]
                # TODO: add the voice
                # react to last move [posture]
                posture_time = time.time()
                if self.is_noding and evaluation_stack:
                    if evaluation_stack[-1] in ["blunder", "mistake", "negative", "inaccuracy"]:
                        self.change_robot_pose("shake")
                    elif evaluation_stack[-1] in ["fair", "good", "killer", "possitive"]:
                        self.change_robot_pose("nod")
                posture_time = time.time() - posture_time
                # make a new move [P&P]
                rospy.loginfo("Robot moves...")
                move_time = time.time()
                move, engine_time = self.engine_move(execution, timeit=True)
                move_time = time.time() - move_time
                # after the move, interact [gaze]
                gaze_time = time.time()
                if self.is_gazing:
                    self.change_robot_pose("human")
                gaze_time = time.time() - gaze_time
                info = self.ext_engine.info_dict
                score, is_mate = self.ext_engine.get_score_from_info(info)
                msg = f"Engine move: {move}; \n"
                robot_counter += 1

                msg += f"Current condition: {current_condition} \n"
                msg += f"posture_time: {posture_time}, move_time: {move_time}, gaze_time: {gaze_time} \n"

            # deal with checkmate
            if is_mate:
                if score > 0:
                    mate_side = "White"
                else:
                    mate_side = "Black"
                msg += f"{mate_side} found checkmate in {score} ways!\n"
            else:
                msg += f"Current score is {score}.\n"
            # evaluate the move quality
            if score_stack and not is_mate and -3000 < score < 3000:
                eva = self.ext_engine.classify_opponent_move(score_stack[-1], score)
            else:
                eva = ""
            if human_color == self.board.turn:
                msg += f"Curent evaluation is {eva}\n"

            # logging
            print(msg)
            move_stack.append(move)
            score_stack.append(score)
            evaluation_stack.append(eva)
            self.board.push_uci(move)

            if self.board.is_game_over():
                rospy.logerr(f"GG")

        if self.board.is_game_over():
            print(self.board.result())



if __name__ == "__main__":
    try:
        rospy.init_node("chess_commander", anonymous=True, log_level=rospy.DEBUG)        
        my_chess = EvaChessCommander()
        my_chess.check_initial_state()
        my_chess.test_evaluation()

    except rospy.ROSInterruptException:
        pass
