#!/home/charles/panda/panda_env310/bin/python3.10

import sys
import click
from chess_commander import ChessCommander
import rospy
import chess
import logging
import time

import cv2
import numpy as np
from std_msgs.msg import String, Bool

from utili.camera_config import Camera
from utili.recap import URI, CfgNode as CN
from chessrec.preprocessing.detect_corners import find_corners, resize_image, sort_corner_points
from chessrec.recognizer.recognizer import ChessRecognizer

from setup_configurations import MODEL_PATH, CAM_IP, MARKER_TYPE
from utili.camera_config import ARUCO_DICT
from utili.logger import setup_logger

# TODO:
## 2.1 what if miss recog?
# 4. separate the camera functions to another node


def debug_img(img, func_name):
    from pathlib import Path
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    path = Path(__file__).parent.parent / "debug"
    path.mkdir(exist_ok=True)
    cv2.imwrite(str(path / f"{func_name}.png"), img)
    rospy.logdebug(f"debug file written at {path} / {func_name}.png")


def detect_markers(frame: np.ndarray, marker_type: str):
    """detect markers and encode marker position in pixels in to np.ndarray: [id, x, y]

    Args:
        frame (np.ndarray): input image

    Returns:
        np.ndarray: markers in ndarray
    """
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    # detector parameters: https://docs.opencv.org/4.5.3/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html
    params = cv2.aruco.DetectorParameters_create()
    # rgbframe = cv2.cvtColor(gframe, cv2.COLOR_GRAY2BGR)
    try:
        dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT[marker_type])
    except KeyError:
        print("indicated marker is not listed in the ARUCO_DICT")
    corners, ids, _ = cv2.aruco.detectMarkers(gframe, dictionary=dictionary, parameters=params)
    corners = np.array(corners).reshape((-1, 4, 2))
    if ids is not None:
        ids = ids.reshape(-1)
        detected_markers = np.zeros((len(ids), 3))
        for i, id in enumerate(ids):
            detected_markers[i, 0] = id
            detected_markers[i, 1:] = np.mean(corners[i], axis=0, dtype=np.int16)
        return detected_markers
    else:
        return np.zeros((99, 3))


def crop_by_marker(img: np.ndarray, markers: np.ndarray, margin=80):
    """crop the image by markers. Can be moved to image utilities.

    Args:
        img (np.ndarray): input image, shape: (1080, 1920, 3)
        markers (np.ndarray): markers in np.ndarray
        margin (int, optional): margin to the marker. Defaults to 80

    Returns:
        np.ndarray: cropped image
    """
    cropped_img = np.zeros(img.shape, dtype=np.uint8)
    _, xmin, ymin = np.amin(markers, axis=0).astype(int)
    _, xmax, ymax = np.amax(markers, axis=0).astype(int)
    ymin = max(0, ymin - margin)
    ymax = min(img.shape[0], ymax + margin)
    xmin = max(0, xmin - margin)
    xmax = min(img.shape[1], xmax + margin)
    cropped_img[ymin:ymax, xmin:xmax, :] = img[ymin:ymax, xmin:xmax, :]
    return cropped_img


class HRIChessCommander(ChessCommander):
    def __init__(self, fen=chess.STARTING_FEN, cam=True):
        super().__init__(fen)
        # enable classifier
        self.corner_cfg = CN.load_yaml_with_base("config://corner_detection.yaml")
        if cam:
            try:
                self.camera = Camera(ip=CAM_IP, port=30000, name="2")
            except Exception:
                print("Camera not working")
                self.shutdown()
        self.chess_rec = self.init_classifier()
        # updated after each change pose
        self.cur_robo_pose = ""
        # save corner positions per pose
        self.cached_corners = dict()
        # define the current marker type
        self.marker_type = MARKER_TYPE
        self.cached_markers = dict()

    def shutdown(self):
        rospy.loginfo("Stopping the ChessCommander node")
        self.ext_engine.shutdown()
        # if hasattr(self, "camera"):
        #     self.camera.close()
        #     rospy.loginfo("Stopping the camera")
        sys.exit()

    def check_corner_by_marker(self, corners, markers):  # doent work for left right state?
        if corners is None:
            return False
        is_valid = True
        left = False
        right = False

        # [top left, top right, bottom right, bottom left] from robot perspective
        corners = sort_corner_points(corners)
        markers = sort_corner_points(markers[:, 1:])

        if self.cur_robo_pose == "left" and len(markers) != 4:
            left = True
            markers = np.insert(markers, 1, None, axis=0)
        if self.cur_robo_pose == "right" and len(markers) != 4:
            right = True
            markers = np.insert(markers, 0, None, axis=0)
        for i in range(len(corners)):
            if not is_valid:
                break
            m = markers[i]
            c = corners[i]
            if i == 0 and not right:  # [top left, top right, bottom right, bottom left] from robot perspective
                if not (c[0] > m[0] and c[1] > m[1]):
                    is_valid = False
                    rospy.logdebug(f"fault {i}, length: {len(markers)}, side: {self.cur_robo_pose}, m: {m}, c: {c}")
            elif i == 1 and not left:
                if not (c[0] < m[0] and c[1] > m[1]):
                    is_valid = False
                    rospy.logdebug(f"fault {i}, length: {len(markers)}, side: {self.cur_robo_pose}, m: {m}, c: {c}")
            elif i == 2:
                if not (c[0] < m[0] and c[1] < m[1]):
                    is_valid = False
                    rospy.logdebug(f"fault {i}, length: {len(markers)}, side: {self.cur_robo_pose}, m: {m}, c: {c}")
            elif i == 3:
                if not (c[0] > m[0] and c[1] < m[1]):
                    is_valid = False
                    rospy.logdebug(f"fault {i}, length: {len(markers)}, side: {self.cur_robo_pose}, m: {m}, c: {c}")
            # rospy.logdebug(id, c, m, is_valid)
        return is_valid

    def init_classifier(self):
        return ChessRecognizer(URI(MODEL_PATH))

    def request_new_img(self):
        """an image from zed camera in (b,g,r) order

        Returns:
            np.ndarray: img array in (r,g,b) order
        """
        img = self.camera.get_img()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def decide_corner(
        self,
        img: np.ndarray,
        markers: np.ndarray,
        corner_detection_cfg: str,
        pose: str,
        update: Bool = False,
    ) -> np.ndarray:
        """decide the corner positions by saved corner coordinates for the current pose,
           or detect corners for current pose and update the cached corner dict.
           The image here is not resized.

        Args:
            img (np.ndarray): img array
            corner_detection_cfg (str): path for corner detection config
            pose (str): arm pose of the current image
            update (Bool): update the cached dict anyway
        """
        if update or (pose not in self.cached_corners.keys()):
            corners = None
            recheck_counter = 50
            while not self.check_corner_by_marker(corners, markers) and not rospy.is_shutdown():
                recheck_counter -= 1
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                try:
                    corners = find_corners(corner_detection_cfg, img)
                except Exception:
                    rospy.logwarn(f"RANSAC failed at trail {50 - recheck_counter}")
                    if recheck_counter % 5 == 0:
                        debug_img(img, f"decide_corner_{recheck_counter}")
                        self.camera.reset()
                rospy.logdebug("double check corners by marker")
                if hasattr(self, "camera"):
                    img = self.request_new_img()
                    rospy.logdebug("request a new image")
                else:
                    rospy.logdebug("not to request an image because the no-cam mode ")
                if recheck_counter < 0:
                    debug_img(img, f"decide_corner_last_recheck_counter")
                    raise Exception(f"cannot find valid corners after 50 trails")
            self.cached_corners[pose] = corners
        else:
            corners = self.cached_corners[pose]
        return corners

    def preprocess_img(self, img: np.ndarray, pose: str, marker_type: str):
        """preprocess a image from the camera by deciding its corners and resize the image
           to make it ready for recognition. Use the self.coner_cfg as the default config file.

        Args:
            img (np.ndarray): raw image from the camera (after BGR-RGB conversion)
            pose (str): camera pose corresponding to the image
            marker_type (str): the type of markers in images

        Returns:
            np.ndarray, np.ndarray: resized image and corners
        """
        config = self.corner_cfg
        if pose not in self.cached_markers.keys():
            markers = detect_markers(img, marker_type)
            while len(markers) != 4 and not rospy.is_shutdown():
                rospy.logdebug(f"marker is missing. Now found {len(markers)}. ")
                img = self.request_new_img()
                markers = detect_markers(img, marker_type)
                rospy.sleep(0.1)
            self.cached_markers[pose] = markers
        else:
            markers = self.cached_markers[pose]
        img = crop_by_marker(img, markers)
        corners = self.decide_corner(img, markers, config, pose)
        img, img_scale = resize_image(config, img)
        corners = corners * img_scale
        return img, corners

    def twostep_recog(
        self,
        img: np.ndarray,
        recognizer: ChessRecognizer,
        color: chess.COLORS = chess.WHITE,
    ):
        """detect the board by two steps: detect occupancy from the top view and detect pieces from lower views

        Args:
            img (np.ndarray): raw image from the camera at lower views
            recognizer (ChessRecognizer): recognizer class that loads the model to GPU
            color (chess.COLORS, optional): the cloest side to the camera. Defaults to chess.WHITE.

        Returns:
            chess.Board: predicted current board
        """
        piece_img = img
        piece_pose = self.cur_robo_pose
        self.change_robot_pose("high")
        occupancy_img = self.request_new_img()
        occupancy_pose = self.cur_robo_pose
        self.change_robot_pose("low")

        occ_img, occ_corners = self.preprocess_img(occupancy_img, occupancy_pose, self.marker_type)
        pie_img, pie_corners = self.preprocess_img(piece_img, piece_pose, self.marker_type)

        occupancy = recognizer.robo_prepredict(occ_img, occ_corners, color)
        board, *_ = recognizer.robo_postpredict(pie_img, pie_corners, occupancy, color)
        return board

    def onestep_recog(
        self,
        img: np.ndarray,
        recognizer: ChessRecognizer,
        color: chess.COLORS = chess.WHITE,
    ):
        """detect the board from the current camera pose

        Args:
            img (np.ndarray): raw image from the camera
            recognizer (ChessRecognizer): recognizer class that loads the model to GPU
            color (chess.COLORS, optional): the cloest side to the camera. Defaults to chess.WHITE.

        Returns:
            chess.Board: predicted current board
        """
        pose = self.cur_robo_pose
        img, corners = self.preprocess_img(img, pose, self.marker_type)
        board, *_ = recognizer.robo_predict(img, corners, color)            
        return board

    def observe_board(self, next_turn=chess.WHITE, multi_pose=False):
        """recognize the move by a human player and update the current board"""
        if self.cur_robo_pose != "low":  # TODO: replaced by a state machine
            self.change_robot_pose("low")
        is_valid = False
        failed_time = 0
        single_step = True
        while not is_valid and not rospy.is_shutdown():
            img = self.request_new_img()
            if single_step:
                try:
                    cog_board = self.onestep_recog(img=img, recognizer=self.chess_rec, color=chess.WHITE)
                except RuntimeError:
                    rospy.logwarn(f"Runtime error at line 310 hri_chess_commander.py")
                    img = self.request_new_img()
                    cog_board = self.onestep_recog(img=img, recognizer=self.chess_rec, color=chess.WHITE)
            else:
                img = self.request_new_img()
                cog_board = self.twostep_recog(img=img, recognizer=self.chess_rec, color=chess.WHITE)
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
            elif cog_board.status() in [chess.Status.TOO_MANY_KINGS, chess.Status.TOO_MANY_BLACK_PIECES]:
                single_step = False
                failed_time += 1
                self.camera.reset()
                rospy.loginfo(f"Found: {cog_board.status().name}")
                rospy.loginfo(f"Current FEN: {cog_board.fen()}")
            elif cog_board.status() is None:
                single_step = False
                failed_time += 1
                rospy.loginfo(f"Found: {cog_board.status().name}")
                rospy.logwarn(f"Current FEN: {cog_board.fen()}")
                input("current game statu is None. Please check again.")
            else:
                failed_time += 1
                rospy.loginfo(f"Invalid state: {cog_board.status().name}")
                rospy.loginfo(f"Current FEN: {cog_board.fen()}")
            # refresh image after 5 times
            if failed_time != 0 and failed_time % 5 == 0:
                self.camera.reset()
                debug_img(img, f"low_{failed_time}")
            # change pose when it cannot recognize
            if failed_time == 10 and multi_pose:
                self.change_robot_pose("right")
                debug_img(img, f"right_{failed_time}")
            if failed_time == 15 and multi_pose:
                self.change_robot_pose("left")
                debug_img(img, f"left_{failed_time}")
            if failed_time == 20:
                self.change_robot_pose("low")
                rospy.logwarn(f"predicted board is not valid after {failed_time} trails")
                print("*************************")
                print(self.board)
                print(cog_board)
                print("*************************")
                break
            rospy.sleep(0.1)
        return cog_board

    def detect_move(self, prev_board: chess.Board, curr_board: chess.Board, turn: chess.Color):
        """
        Detects the move made from the previous chess board state to the current chess board state.

        Parameters:
        prev_board (chess.Board): The previous chess board state.
        curr_board (chess.Board): The current chess board state.

        Returns:
        chess.Move: The move made from the previous state to the current state, or None if no move is found.
        """
        if curr_board.status == chess.STATUS_OPPOSITE_CHECK:
            return ""
        
        white_kingside = chess.Move.from_uci("e1g1")
        white_queenside = chess.Move.from_uci("e1c1")
        black_kingside = chess.Move.from_uci("e8g8")
        black_queenside = chess.Move.from_uci("e8c8")

        cp_board = prev_board.copy(stack=False)
        cp_board.turn = turn
        for move in cp_board.legal_moves:
            if (
                curr_board.piece_at(move.to_square) == cp_board.piece_at(move.from_square)
                and curr_board.piece_at(move.from_square) == None
            ):
                # found a legal move, check castling
                if turn == chess.BLACK:
                    if chess.square_rank(move.to_square) == 7 and chess.square_rank(move.from_square) == 7:
                        if cp_board.piece_at(move.from_square).piece_type == chess.ROOK:
                            if black_kingside in cp_board.legal_moves:
                                return black_kingside
                            elif black_queenside in cp_board.legal_moves:
                                return black_queenside
                if turn == chess.WHITE:
                    if chess.square_rank(move.to_square) == 0 and chess.square_rank(move.from_square) == 0:
                        if cp_board.piece_at(move.from_square).piece_type == chess.ROOK:
                            if white_kingside in cp_board.legal_moves:
                                return white_kingside
                            elif white_queenside in cp_board.legal_moves:
                                return white_queenside
                return move
        rospy.logwarn("No move was detected")
        return ""

    def check_initial_state(self):
        rospy.loginfo("Checking initial state of the board...")
        rospy.sleep(0.1) # wait a bit for the signal to go
        self.change_robot_pose(pose="low")
        while not rospy.is_shutdown() and not rospy.get_param("is_last_command_successful"):
            rospy.sleep(0.1)
            rospy.logdebug("waiting for the full execution of the last command")
        checked = False

        while checked == False and not rospy.is_shutdown():
            init_board = self.observe_board()
            if init_board == chess.Board():
                rospy.loginfo("Board is in initial state")
                checked = True
            else:
                print(init_board.transform(chess.flip_vertical).transform(chess.flip_horizontal), "\n")
                cont = click.prompt("Board is not in initial state, continue anyway? [y|n]", default="y")
                if cont == "n":
                    input("Reset pieced and press enter to check again...")
                else:
                    checked = True
                turn = click.prompt("Whose turn? [w|b]", default="w")
                if turn == "w":
                    init_board.turn = chess.WHITE
                else:
                    init_board.turn = chess.BLACK

        rospy.loginfo("Initial board state saved")
        self.board = init_board

    # TODO: delete as it's duplicated in gpt_chess_assistant.py
    def zip_fen_and_moves(self, board: chess.Board, multipv: int) -> String:
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
            last_move = last_board.pop()
            last_fen = last_board.fen().replace(" ", "_")
            return f"fen:{last_fen},moves:[{last_move}]"
        else:
            moves = ""
            fen = board.fen().replace(" ", "_")
            moves = self.ext_engine.multipv(board, multipv)
            return f"fen:{fen},moves:[{moves}]"

    def test_analyse(self, infer_move=True, multipv=2):
        pub_fen_and_moves = rospy.Publisher("/chess_fen_and_moves", String, queue_size=50, latch=False)

        print("Start recognition test ...")
        # Choose a color by number: white:1 | black:0 \n-In standard rules, white moves first.
        human_color = chess.BLACK
        robot_color = chess.WHITE
        while not self.board.is_game_over() and not rospy.is_shutdown():
            input("Press Enter when a move is done..")
            recognized_board = self.observe_board()
            if infer_move:
                if human_color == self.board.turn:
                    # recognize white move first
                    move = self.detect_move(self.board, recognized_board, human_color)
                    print(f"Human move: {move}", "\n")
                else:
                    move = self.detect_move(self.board, recognized_board, robot_color)
                    print(f"Robot move: {move}", "\n")
                if move:
                    self.board.push(chess.Move.from_uci(str(move)))
                else:
                    if recognized_board.status == chess.STATUS_OPPOSITE_CHECK:
                        print("You Win!")
                        break
                    else:
                        print("No move detected. Pass.")
            else:
                self.board = recognized_board
            msg = self.zip_fen_and_moves(self.board, multipv)
            pub_fen_and_moves.publish(msg)
            print(self.board.transform(chess.flip_vertical).transform(chess.flip_horizontal), "\n")

        if self.board.is_game_over():
            print(self.board.result())

    def test_human_play(self, execution=True, always_rec=False):
        human_color = chess.BLACK
        input("press Enter when you want to start")
        while not self.board.is_game_over() and not rospy.is_shutdown():
            print(self.board, "\n")
            if always_rec:
                if human_color == self.board.turn:
                    rospy.loginfo("Human moves...")
                    input("Press Enter when your move is done...")
                    rec_board = self.observe_board()
                    move = self.detect_move(self.board, rec_board, human_color)
                    self.board = rec_board
                    print(f"Human move: {move}", "\n")
                else:
                    rospy.loginfo("Robot moves...")
                    move = self.engine_move(execution)
                    self.board.push(chess.Move.from_uci(str(move)))
                    print(f"Engine move: {move}", "\n")
            else:
                if human_color == self.board.turn:
                    rospy.loginfo("Human moves...")
                    input("Press Enter when your move is done...")
                    move = self.detect_move(self.board, self.observe_board(), human_color)
                    print(f"Human move: {move}", "\n")
                else:
                    rospy.loginfo("Robot moves...")
                    move = self.engine_move(execution)
                    print(f"Engine move: {move}", "\n")
                if move:
                    self.board.push(chess.Move.from_uci(str(move)))

            print(self.board.transform(chess.flip_vertical).transform(chess.flip_horizontal), "\n")

            if self.board.is_game_over():
                rospy.logerr(f"GG")

        if self.board.is_game_over():
            print(self.board.result())

    def test_interaction(self, execution=True):
        logger = setup_logger("interaction_logger", "/home/charles/panda/catkin_ws/src/move_chess_panda/scripts/logs/interaction.txt")

        human_color = chess.BLACK
        move_stack = []
        score_stack = []
        evaluation_stack = []

        input("Press Enter when ready...")
        while not self.board.is_game_over() and not rospy.is_shutdown():
            print(self.board, "\n")
            detect_time = np.nan
            evaluate_time = np.nan
            engine_time = np.nan
            human_time = np.nan
            if human_color == self.board.turn:
                rospy.set_param("is_waiting", True)
                rospy.loginfo("Human moves...")
                human_time = time.time()
                input("Press Enter when your move is done...")
                human_time = time.time() - human_time
                while rospy.get_param("is_moving"):
                    rospy.sleep(0.2)
                detect_time = time.time()
                move = str(self.detect_move(self.board, self.observe_board(), human_color))
                detect_time = time.time() - detect_time
                if not move:
                    print("No move detected. Will check again after one second")
                    continue
                evaluate_time = time.time()
                info = self.ext_engine.evaluate_move(self.board, move)
                evaluate_time = time.time() - evaluate_time
                score, is_mate = self.ext_engine.get_score_from_info(info)
                score *= -1
                msg = f"Human move: {move}; "
            else:
                rospy.set_param("is_waiting", False)
                while rospy.get_param("is_moving") and not rospy.is_shutdown():
                    rospy.sleep(0.2)
                # react to last move
                if evaluation_stack:
                    if evaluation_stack[-1] in ["blunder", "mistake"]:
                        self.change_robot_pose("shake")
                    elif evaluation_stack[-1] in ["inaccuracy"]:
                        self.change_robot_pose("human")
                    elif evaluation_stack[-1] in ["fair", "good", "killer"]:
                        self.change_robot_pose("nod")
                # make a new move
                rospy.loginfo("Robot moves...")
                move, engine_time = self.engine_move(execution, timeit=True)
                info = self.ext_engine.info_dict
                score, is_mate = self.ext_engine.get_score_from_info(info)
                msg = f"Engine move: {move};"
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
                msg += f"Curent evaluation is {eva}"

            # logging
            print(msg)
            move_stack.append(move)
            score_stack.append(score)
            evaluation_stack.append(eva)
            self.board.push_uci(move)
            logger.info(
                f"turn: {self.board.turn}, move: {move}; score: {score}; evaluation: {eva}; detect_time: {detect_time}; evaluate_time: {evaluate_time}; engine_time: {engine_time}"
            )
            print(self.board.transform(chess.flip_vertical).transform(chess.flip_horizontal), "\n")
            if self.board.is_game_over():
                rospy.logerr(f"GG")

        if self.board.is_game_over():
            print(self.board.result())

    def test_recognition(self, infer_move=True):
        print("Start recognition test ...")
        # Choose a color by number: white:1 | black:0 \n-In standard rules, white moves first.
        human_color = chess.BLACK
        robot_color = chess.WHITE

        while not self.board.is_game_over() and not rospy.is_shutdown():
            input("Press Enter when a move is done..")
            if infer_move:
                if human_color == self.board.turn:
                    # recognize white move first
                    move = self.detect_move(self.board, self.observe_board(), human_color)
                    print(f"Human move: {move}", "\n")
                else:
                    move = self.detect_move(self.board, self.observe_board(), robot_color)
                    print(f"Robot move: {move}", "\n")
                self.board.push(chess.Move.from_uci(str(move)))
            else:
                self.board = self.observe_board()
            rospy.loginfo("Board updated with detected move")
            print(self.board.transform(chess.flip_vertical).transform(chess.flip_horizontal), "\n")

        if self.board.is_game_over():
            print(self.board.result())

    def test_self_play(self):
        # Choose a color by number: white:1 | black:0 \n-In standard rules, white moves first.
        turn = click.prompt("Who's turn is it? [w|b]", default="w")

        if turn == "w":
            self.board.turn = chess.WHITE
        else:
            self.board.turn = chess.BLACK

        while not self.board.is_game_over() and not rospy.is_shutdown():
            if self.board.turn == chess.WHITE:
                rospy.loginfo("White moves...")
            else:
                rospy.loginfo("Black moves...")
            emove = self.engine_move(execution=True)
            rec_board = self.observe_board()
            rmove = self.detect_move(self.board, rec_board, self.board.turn)

            if not emove and not rmove:
                rospy.logerr(f"GG")
            elif str(emove) != str(rmove) and not self.board.is_castling(chess.Move.from_uci(str(emove))):
                print(f"Engine Move: {emove}")
                print(f"Recognized Move: {rmove}")
                print(f"Current board: ")
                print(self.board.transform(chess.flip_vertical).transform(chess.flip_horizontal), "\n")
                print(f"Rec board: ")
                print(rec_board.transform(chess.flip_vertical).transform(chess.flip_horizontal), "\n")
                input("Engine move and Recognition move do not correspond, please fix...\n")
            else:
                self.board.push(chess.Move.from_uci(str(emove)))

            print(self.board.transform(chess.flip_vertical).transform(chess.flip_horizontal), "\n")

        if self.board.is_game_over():
            print(self.board.result())


if __name__ == "__main__":
    try:
        rospy.init_node("chess_commander", anonymous=True, log_level=rospy.DEBUG)
        my_chess = HRIChessCommander(cam=True)
        if rospy.has_param("board_is_localized"):
            rate = rospy.Rate(60)
            while not rospy.is_shutdown() and not rospy.get_param("board_is_localized"):
                rospy.logdebug("waiting for board localization")
                rate.sleep()
        else:
            rospy.logwarn("{board_is_localized} is not found in the param server. Is the param manager node initialized?")
        my_chess.check_initial_state()
        # my_chess.test_recognition(infer_move=False)
        my_chess.test_human_play(always_rec=False)
        # my_chess.test_human_play_analyze()
        # my_chess.test_analyse()
        # my_chess.test_self_play()
        # my_chess.test_interaction()

    except rospy.ROSInterruptException:
        pass
