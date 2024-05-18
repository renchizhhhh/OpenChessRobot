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
import sys

# an error handler to surpass the ALSA warnings. Thanks to
# https://stackoverflow.com/questions/7088672/pyaudio-working-but-spits-out-error-messages-each-time
from ctypes import *
from contextlib import contextmanager
ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
@contextmanager
def noalsaerr():
    asound = cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
    yield
    asound.snd_lib_error_set_handler(None)

class HRIChessAssistant(HRIChessCommander):
    def __init__(self, fen=chess.STARTING_FEN, cam=True):
        super().__init__(fen, cam)
        self.to_explain = False
        self.to_analyze = False
        self.to_continue = False
        self.pub_chat = rospy.Publisher("/chess_chat", String, queue_size=50, latch=False)
    
    def set_continue(self, value):
        print(f"set continue to {value}")
        self.to_continue = value

    def set_explain(self, value):
        print(f"set explain to {value}")
        self.to_explain = value

    def set_analyze(self, value):
        print(f"set analyze to {value}")
        self.to_analyze = value

    def zip_board_info(self, board: chess.Board, multipv: int, move: str="") -> String:
        # "fen:2r3k1/pp1q2pp/5pb1/2r5/5Q2/3P2NP/P1P1RRP1/6K1_b_-_-_2_27,moves:[c5d4,a5a4]"
        """wrap the input for the OpenAI API chat completion

        Args:
            board (chess.Board): the board to analyze
            multipv (int): number of best moves to prob

        Returns:
            String: the encoded chess fen
        """
        if multipv == -1:
            last_board = board.copy()
            if move:
                last_move = last_board.san(move)
            else:
                try:
                    last_move = last_board.san(last_board.pop())
                except IndexError:
                    print("Opps! The move stack is empty!")
            last_fen = last_board.fen().replace(" ", "_")
            return f"'fen':'{last_fen}','move':'{last_move}'"
            # history = [move.uci() for move in last_board.move_stack]
            # return f"'fen':'{last_fen}','move':'{last_move}','history':'{history}'"
        else:
            moves = ""
            fen = board.fen().replace(" ", "_")
            moves = self.ext_engine.multipv(board, multipv)
            return f"fen:{fen},moves:[{moves}]"

    def observe_board(self, next_turn=chess.WHITE):
        """recognize the move by a human player and update the current board"""
        is_valid = False
        failed_time = 0
        single_step = True
        img = self.request_new_img()
        # debug_img(img, "wrong_color_ROCK")
        while not is_valid and not rospy.is_shutdown():
            if single_step:
                cog_board= self.onestep_recog(img=img, recognizer=self.chess_rec, color=chess.WHITE)
            else:
                img = self.request_new_img()
                cog_board= self.twostep_recog(img=img, recognizer=self.chess_rec, color=chess.WHITE)
                single_step = True
            # NOTE: not remembering the castling right
            if next_turn == chess.BLACK:
                cog_board.turn = chess.BLACK
                rospy.loginfo(f"Look for the turn: {cog_board.fen()}")
            cog_board.castling_rights = cog_board.clean_castling_rights()
            if cog_board.status() == chess.Status.VALID or cog_board.status() == chess.STATUS_OPPOSITE_CHECK:
                is_valid = True
                rospy.loginfo(f"Current board status: {cog_board.status().name}")
                rospy.loginfo(f"Current FEN: {cog_board.fen()}")
            # elif cog_board.status() in [chess.Status.TOO_MANY_KINGS, chess.Status.TOO_MANY_BLACK_PIECES]:
            #     single_step = False
            #     failed_time += 1
            #     self.camera.reset()
            #     rospy.loginfo(f"Found: {cog_board.status().name}")
            #     rospy.loginfo(f"Current FEN: {cog_board.fen()}")
            # elif cog_board.status() is None:
            #     single_step = False
            #     failed_time += 1
            #     self.camera.reset()
            #     rospy.loginfo(f"Found: {cog_board.status().name}")
            #     rospy.logwarn(f"Current FEN: {cog_board.fen()}")
            #     input("current game statu is None. Please check again.")
            else:
                failed_time += 1
                rospy.loginfo(f"Invalid state: {cog_board.status().name}")
                rospy.loginfo(f"Current FEN: {cog_board.fen()}")
            # refresh image after 5 times
            if failed_time == 5:
                self.camera.reset()
                img = self.request_new_img()
            # # change pose when it cannot recognize
            # if failed_time == 10:
            #     self.change_robot_pose("right")
            #     img = self.request_new_img()
            # if failed_time == 15:
            #     self.change_robot_pose("left")
            #     img = self.request_new_img()
            if failed_time == 20:
                self.change_robot_pose("low")
                rospy.logwarn(f"predicted board is not valid after {failed_time} trails")
                print("*************************")
                print(self.board)
                print(cog_board)
                print("*************************")
                break
            rospy.sleep(0.2)
        return cog_board

    # def ask_gpt(self, publisher: rospy.Publisher, multipv: int):
    #     """ask gpt to complete the conversation

    #     Args:
    #         publisher (rospy.Publisher): the ROS publisher.
    #         multipv (int): how many moves to predict. Set to -1 to explain the last move.
    #     """
    #     if self.to_explain:
    #         msg = self.zip_board_info(self.board, multipv=-1)
    #         publisher.publish(msg)
    #         self.set_explain(False)
    #     if self.to_analyze:
    #         msg = self.zip_board_info(self.board, multipv)
    #         publisher.publish(msg)
    #         self.set_analyze(False)

    # Jan 24: only detect the occlusion


    def test_gpt_analyze(self):        
        cached = ['Nh3', 'Nf3', 'Nc3', 'Na3', 'h3', 'g3', 'f3', 'e3', 'd3', 'c3',
                    'b3', 'a3', 'h4', 'g4', 'f4', 'e4', 'd4', 'c4', 'b4', 'a4']
        input("Press Enter when ready...")
        for move in cached:
            move = self.board.parse_san(move)
            self.board.push(move)
            move = move.uci() 
            print(f"current move {move}")
            self.robot_execution(move)
            self.change_robot_pose("stare")
            # is_speaking is also set in the audio_gpt.py L137
            rospy.set_param('is_speaking', True)
            while rospy.get_param('is_speaking') and not rospy.is_shutdown():
                rospy.sleep(0.1)
            self.change_robot_pose("low")
            undo_move = move[2:]+move[:2]
            self.robot_execution(undo_move)
            self.board.pop()

class AssistantWrapper:
    def __init__(self, assistant) -> None:
        self.assistant = assistant
        with noalsaerr():
            self.r = sr.Recognizer()
            self.mic = sr.Microphone()
            self.mic_check()
        self.listener_thread = threading.Thread(target=self.listener_loop)
        self.listener_thread.start()

    def mic_check(self):
        try:
            with self.mic as source:  # This makes a lot of text, so I want to get it 
                self.r.adjust_for_ambient_noise(source)  # Out of the way to make messages cleaner
                audio = self.r.listen(source, timeout=1)
                rospy.loginfo('Mic checking')
        except sr.WaitTimeoutError:
            rospy.loginfo('Mic check pass')
        except Exception as e:
            rospy.logerr(f'Mic check failed: {e}')

    def recog_voice(self, assistant: HRIChessAssistant):
        # TODO: improve the listening 
        with self.mic as source:   
            self.r.adjust_for_ambient_noise(source)  
            print("listening start")
            audio = self.r.listen(source, phrase_time_limit=4)
            try:
                speech = self.r.recognize_google(audio) + '\n'
                print(f"captured sentence: {speech}")
                # if "hi panda" or "hey panda" in speech.lower():
                if "analyze" in speech.lower():
                    assistant.set_analyze(True)
                if "explain" in speech.lower():
                    assistant.set_explain(True) 
                    move = assistant.detect_move(assistant.board, assistant.observe_board(), assistant.board.turn)
                    msg = assistant.zip_board_info(assistant.board, multipv=-1, move=move)
                    assistant.pub_chat.publish(msg)
                if "continue" in speech.lower():
                    assistant.set_continue(True)
            except sr.UnknownValueError or UnboundLocalError:
                print("nothing catched...will try again")
            print("listening finished")

    def listener_loop(self):
        while not rospy.is_shutdown():
            try:
                with noalsaerr():
                    self.recog_voice(self.assistant)
            except KeyboardInterrupt:
                sys.exit()

if __name__ == "__main__":
    try:
        rospy.init_node("chess_commander", anonymous=True, log_level=rospy.DEBUG)
        my_chess = HRIChessAssistant(cam=True)      
        my_wrapper = AssistantWrapper(my_chess)
        # my_chess.test_gpt_analyze()
        # my_chess = HRIChessCommander(cam=True)    
        my_chess.check_initial_state()
        my_chess.test_human_play_analyze()
    except rospy.ROSInterruptException:
        pass

