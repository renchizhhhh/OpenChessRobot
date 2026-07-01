#!/usr/bin/env python3

from ocr_runtime.script_imports import prefer_source_scripts
prefer_source_scripts(__file__)
from chess_robo_player import ChessRoboPlayer

from std_msgs.msg import String
from open_chess_robot.srv import RecognizeBoard, RecognizeBoardResponse
from ocr_runtime.board_recognition import BoardRecognitionEngine
from ocr_runtime.settings import (
    RECOGNIZE_BOARD_SERVICE,
    LOCALIZATION_MAX_ATTEMPTS,
)
from ocr_runtime.startup_checks import (
    VALID_STATUS_NAMES,
    board_status_name,
)
import rospy
import copy

class HRIChessRobot(ChessRoboPlayer):
    def __init__(self, cam=True):
        super().__init__(cam)
        self.motion_acc = 0.3
        self.motion_vel = 0.4
        self.recognition_engine = None

    def move_to_recognition_pose(self, pose):
        if pose in ("", "low", "camera"):
            self.move_camera_state_low()
        elif pose == "high":
            self.move_camera_state_high()
        elif pose == "left":
            self.move_camera_state_left()
        elif pose == "right":
            self.move_camera_state_right()
        else:
            raise ValueError(f"unsupported recognition camera pose: {pose}")

    def setup_recognition_service(self):
        if not self.cam_on:
            rospy.logwarn("Recognition service disabled because camera is off")
            return
        self.recognition_engine = BoardRecognitionEngine(
            camera=self.camera,
            change_pose=self.move_to_recognition_pose,
        )
        service_name = rospy.get_param(
            "/open_chess_robot/recognition/service_name",
            RECOGNIZE_BOARD_SERVICE,
        )
        rospy.Service(service_name, RecognizeBoard, self.handle_recognize_board)
        rospy.loginfo(f"Board recognition service ready at {service_name}")

    def handle_recognize_board(self, req):
        response = RecognizeBoardResponse()
        response.confidence = -1.0

        if self.recognition_engine is None:
            response.success = False
            response.status = RecognizeBoardResponse.STATUS_UNSUPPORTED_REQUEST
            response.status_label = "RECOGNITION_DISABLED"
            response.message = "Recognition service is not initialized"
            return response

        try:
            board = self.recognition_engine.recognize_board(
                camera_pose=req.camera_pose,
                next_turn=req.next_turn,
                refresh_geometry=req.refresh_geometry,
                camera_side=req.camera_side,
                augment=req.augment,
            )
        except Exception as exc:
            response.success = False
            response.status = RecognizeBoardResponse.STATUS_RECOGNITION_ERROR
            response.status_label = type(exc).__name__
            response.message = str(exc)
            response.debug_image_paths = list(self.recognition_engine.debug_image_paths)
            rospy.logwarn(f"Recognition request failed: {exc}")
            return response

        status_name = board_status_name(board)
        response.fen = board.fen()
        response.status_label = status_name
        response.confidence = self.recognition_engine.last_confidence
        response.ambiguous_squares = list(self.recognition_engine.last_ambiguous_squares)
        response.debug_image_paths = list(self.recognition_engine.debug_image_paths)
        if status_name in VALID_STATUS_NAMES:
            response.success = True
            if response.ambiguous_squares:
                response.status = RecognizeBoardResponse.STATUS_AMBIGUOUS
                response.message = (
                    "Board recognized with low-confidence squares: "
                    + ", ".join(response.ambiguous_squares)
                )
            else:
                response.status = RecognizeBoardResponse.STATUS_OK
                response.message = "Board recognized"
        else:
            response.success = False
            response.status = RecognizeBoardResponse.STATUS_INVALID_BOARD
            response.message = f"Recognized board is invalid: {status_name}"
        return response

    def do_and_undo_a_move(self, move):
        self.move_ready_state()
        start = move.upper()[:2]
        end = move.upper()[2:4]
        is_hop = int(move[-5])
        is_capture = int(move[-4])
        is_castling = int(move[-3])
        is_en_passant = int(move[-2])
        is_promotion = int(move[-1])

        rospy.loginfo(f"promotion: {is_promotion}, en passant: {is_en_passant}, capture: {is_capture}, hop: {is_hop}")
        rospy.loginfo(f"executing current move: {move}")
        # capture cannot undo for now

        # normal move
        self.pick(start)
        self.place(end, ishigh=is_hop, release=False)
        self.place(start, ishigh=is_hop, release=True)
        self.move_ready_state()

    def around_marker(self, z_offset=0.5):
        path = []
        sorted_markers = sorted(self._markers_world, key=lambda x: (x[0], x[1]))
        for m in sorted_markers:
            print(f"current marker: {m}")
            start_pose = self.commander.get_current_pose().pose
            start_pose.position.x = m[0] + 0.05 if m[0] < 0.5 else m[0] - 0.2
            start_pose.position.y = m[1] 
            start_pose.position.z = m[2] + z_offset
            path.append(copy.deepcopy(start_pose))
        self.execute_path(path, constrain=False)
        self.move_ready_state()

    def indicate_square(self, square, z_offset=0.2):
        while rospy.get_param("is_moving"):
            rospy.sleep(0.1)
        self.move_ready_state()
        path = []
        start_position = self.board_grid[square]
        # plan for start location
        print(f"start_position: {start_position}")
        start_pose = self.commander.get_current_pose().pose
        start_pose.position.x = start_position[0]
        start_pose.position.y = start_position[1]
        start_pose.position.z = start_position[2] + z_offset
        path.append(copy.deepcopy(start_pose))
        self.execute_path(path)
        self.move_ready_state()

    def respond_pose_message(self, pose_msg):
        if "undo" == pose_msg.data[:4]:
            self.do_and_undo_a_move(pose_msg.data[4:])
        if "look" in pose_msg.data:
            # e.g. "look A2" - near-nadir lens view over the square, stay put
            self.look_down_square(pose_msg.data[-2:])
        elif "square" in pose_msg.data:
            # e.g. square A2
            self.indicate_square(pose_msg.data[-2:])
        if pose_msg.data == "markers":
            self.around_marker()
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
            self.move_camera_state_human(acc=self.motion_acc, vel=self.motion_vel)
            self.move_camera_state_low(acc=self.motion_acc, vel=self.motion_vel)
            self.move_camera_state_low()
        if pose_msg.data == "shake":
            self.move_camera_state_low()
            self.move_camera_state_away(
                acc=self.motion_acc * 2, vel=self.motion_vel * 2
            )
            self.move_camera_state_away_opposite(
                acc=self.motion_acc * 2, vel=self.motion_vel * 2
            )
            self.move_camera_state_low()
            rospy.sleep(1)
        if pose_msg.data == "rotate":
            self.move_camera_state_low()
            self.move_camera_state_rotate(acc=self.motion_acc, vel=self.motion_vel)
            self.move_camera_state_low(acc=self.motion_acc, vel=self.motion_vel)
        if pose_msg.data == "human":
            self.move_camera_state_low()
            self.move_camera_state_human(acc=self.motion_acc, vel=self.motion_vel)
            self.move_camera_state_low(acc=self.motion_acc, vel=self.motion_vel)
        if pose_msg.data == "stare":
            self.move_camera_state_low()
            self.move_camera_state_human(acc=self.motion_acc, vel=self.motion_vel)

    def localize_board(self):
        """Localize the board, retrying the full marker pipeline on failure.

        A single intermittent marker dropout otherwise raises out of
        ``all_update`` and kills the node, leaving the commander waiting on
        ``board_is_localized`` forever. Retrying with a camera reset and a fresh
        high-pose view recovers the common transient case.
        """
        attempts = max(1, int(rospy.get_param(
            "/open_chess_robot/localization/max_attempts",
            LOCALIZATION_MAX_ATTEMPTS)))
        for attempt in range(1, attempts + 1):
            try:
                self.all_update()
                return
            except Exception as exc:
                rospy.logwarn(
                    "Board localization attempt %s/%s failed: %s",
                    attempt, attempts, exc)
                if attempt == attempts:
                    rospy.logerr(
                        "Board localization failed after %s attempts", attempts)
                    raise
                if self.cam_on:
                    self.camera.reset()
                self.move_camera_state_high()
                rospy.sleep(0.5)

    def run(self):
        startup_mode = rospy.get_param("/open_chess_robot/startup_mode", "game")
        if startup_mode not in ("game", "localize_only"):
            rospy.logwarn(
                "Unknown startup_mode '%s'; falling back to game mode",
                startup_mode,
            )
            startup_mode = "game"

        self.move_camera_state_high()
        rospy.sleep(0.5)
        self.localize_board()
        self.init_constraints(workspace=False)
        if startup_mode == "game":
            self.setup_recognition_service()
            rospy.Subscriber("/chess_move", String, self.respond_chess_message)
        else:
            rospy.loginfo(
                "Localize-only mode active: board grid is available, "
                "commander and chess move execution are disabled."
            )
        rospy.Subscriber("/change_pose", String, self.respond_pose_message)
        if rospy.has_param("board_is_localized"):
            rospy.logdebug("Set board localization")
            rospy.set_param("board_is_localized", True)
        else:
            rospy.logwarn(
                "{board_is_localized} is not found in the param server. Is the param manager node initialized?"
            )


if __name__ == "__main__":
    try:
        rospy.init_node("chess_robo_player", anonymous=True, log_level=rospy.DEBUG)
        my_robo_player = HRIChessRobot(cam=True)
        my_robo_player.run()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
