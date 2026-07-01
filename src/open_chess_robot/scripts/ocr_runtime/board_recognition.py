import cv2
import numpy as np
import rospy
import chess

from chessrec.preprocessing.detect_corners import (
    find_corners,
    resize_image,
    sort_corner_points,
)
from chessrec.recognizer.recognizer import ChessRecognizer
from ocr_runtime.paths import user_data_path
from ocr_runtime.recognition_confidence import summarize_confidence
from ocr_runtime.recognition_augment import apply_augmentation
from ocr_runtime.settings import (
    CAMERA,
    MARKER_TYPE,
    MODEL_PATH,
    CORNER_MAX_TRIALS,
    CORNER_RETRY_DELAY,
    CORNER_MARKER_TOL_RATIO,
    CONFIDENCE_AMBIGUOUS_THRESHOLD,
    RECOGNITION_MARKER_MAX_ATTEMPTS,
)
from ocr_runtime.camera_config import (
    ARUCO_DICT,
    make_detector_params,
    detect_markers_filtered,
)
from utili.recap import URI, CfgNode as CN


def debug_img(img, func_name):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    path = user_data_path("debug", f"{func_name}.png")
    cv2.imwrite(str(path), img)
    rospy.logdebug(f"debug file written at {path}")
    return str(path)


def detect_markers(frame: np.ndarray, marker_type: str):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    params = make_detector_params()
    try:
        dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT[marker_type])
    except KeyError:
        raise ValueError("indicated marker is not listed in the ARUCO_DICT")
    corners, ids, _ = detect_markers_filtered(gray, dictionary, params)
    corners = np.array(corners).reshape((-1, 4, 2))
    if ids is None:
        return np.zeros((0, 3))
    ids = ids.reshape(-1)
    detected_markers = np.zeros((len(ids), 3))
    for i, marker_id in enumerate(ids):
        detected_markers[i, 0] = marker_id
        detected_markers[i, 1:] = np.mean(corners[i], axis=0, dtype=np.int16)
    return detected_markers


def crop_by_marker(img: np.ndarray, markers: np.ndarray, margin=80):
    cropped_img = np.zeros(img.shape, dtype=np.uint8)
    _, xmin, ymin = np.amin(markers, axis=0).astype(int)
    _, xmax, ymax = np.amax(markers, axis=0).astype(int)
    ymin = max(0, ymin - margin)
    ymax = min(img.shape[0], ymax + margin)
    xmin = max(0, xmin - margin)
    xmax = min(img.shape[1], xmax + margin)
    cropped_img[ymin:ymax, xmin:xmax, :] = img[ymin:ymax, xmin:xmax, :]
    return cropped_img


class BoardRecognitionEngine:
    def __init__(self, camera, change_pose=None, marker_type=MARKER_TYPE):
        self.camera = camera
        self.change_pose = change_pose
        self.corner_cfg = CN.load_yaml_with_base("config://corner_detection.yaml")
        self.chess_rec = ChessRecognizer(URI(MODEL_PATH))
        self.marker_type = marker_type
        self.cached_corners = {}
        self.cached_markers = {}
        self.cur_robo_pose = ""
        self.cur_side = CAMERA
        self.debug_image_paths = []
        self.corner_max_trials = max(1, int(rospy.get_param(
            "/open_chess_robot/recognition/corner_max_trials", CORNER_MAX_TRIALS)))
        self.corner_retry_delay = float(rospy.get_param(
            "/open_chess_robot/recognition/corner_retry_delay", CORNER_RETRY_DELAY))
        self.corner_marker_tol_ratio = float(rospy.get_param(
            "/open_chess_robot/recognition/corner_marker_tol_ratio",
            CORNER_MARKER_TOL_RATIO))
        self.marker_max_attempts = max(1, int(rospy.get_param(
            "/open_chess_robot/recognition/marker_max_attempts",
            RECOGNITION_MARKER_MAX_ATTEMPTS)))
        self.confidence_threshold = float(rospy.get_param(
            "/open_chess_robot/recognition/confidence_threshold",
            CONFIDENCE_AMBIGUOUS_THRESHOLD))
        self.last_confidence = -1.0
        self.last_ambiguous_squares = []

    def request_new_img(self):
        img = self.camera.get_img(self.cur_side)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _geom_key(self, pose):
        """Geometry cache key. Left/right lenses see the board with a parallax
        shift, so their markers/corners must not share a cache entry."""
        return f"{pose}:{self.cur_side}"

    def move_to_pose(self, pose):
        if not pose:
            return
        if self.change_pose is not None and pose != self.cur_robo_pose:
            self.change_pose(pose)
        self.cur_robo_pose = pose

    def check_corner_by_marker(self, corners, markers):
        if corners is None or len(markers) != 4:
            return False
        is_valid = True
        left = False
        right = False

        corners = sort_corner_points(corners)
        markers = sort_corner_points(markers[:, 1:])

        # Allow a small margin so corners that sit marginally outside their marker
        # quadrant are not rejected. The margin scales with the marker span, so it
        # is independent of image resolution and apparent board size.
        tol = self.corner_marker_tol_ratio * float(np.linalg.norm(np.ptp(markers, axis=0)))

        if self.cur_robo_pose == "left" and len(markers) != 4:
            left = True
            markers = np.insert(markers, 1, None, axis=0)
        if self.cur_robo_pose == "right" and len(markers) != 4:
            right = True
            markers = np.insert(markers, 0, None, axis=0)
        for i in range(len(corners)):
            if not is_valid:
                break
            marker = markers[i]
            corner = corners[i]
            if i == 0 and not right:
                if not (corner[0] > marker[0] - tol and corner[1] > marker[1] - tol):
                    is_valid = False
            elif i == 1 and not left:
                if not (corner[0] < marker[0] + tol and corner[1] > marker[1] - tol):
                    is_valid = False
            elif i == 2:
                if not (corner[0] < marker[0] + tol and corner[1] < marker[1] + tol):
                    is_valid = False
            elif i == 3:
                if not (corner[0] > marker[0] - tol and corner[1] < marker[1] + tol):
                    is_valid = False
        return is_valid

    def decide_corner(self, img, markers, pose, update=False):
        key = self._geom_key(pose)
        if not update and key in self.cached_corners:
            return self.cached_corners[key]

        corners = None
        trial = 0
        while not rospy.is_shutdown():
            trial += 1
            try:
                corners = find_corners(self.corner_cfg, img)
            except Exception:
                rospy.logwarn(f"corner detection failed at trial {trial}")
                corners = None
                if trial % 5 == 0:
                    path = debug_img(img, f"decide_corner_{trial}")
                    self.debug_image_paths.append(path)
                    self.camera.reset()
            # find_corners can intermittently return an unstable / whole-square-
            # shifted grid on a fully-occupied board. Validate it against the
            # marker quadrants and retry on a fresh frame if it disagrees - the
            # same double-check detect mode uses. In project mode the markers are
            # projected, so they carry hand-eye error; if good corners get
            # falsely rejected, loosen corner_marker_tol_ratio.
            if self.check_corner_by_marker(corners, markers):
                break
            if trial >= self.corner_max_trials:
                path = debug_img(img, "decide_corner_last_trial")
                self.debug_image_paths.append(path)
                raise RuntimeError(
                    f"cannot find valid corners after {self.corner_max_trials} trials")
            # Re-crop every retry frame to the marker bounding box. Without this the
            # next find_corners would run on the full uncropped frame (markers,
            # table, background), which produces more lines and makes failures
            # cascade instead of recover.
            img = crop_by_marker(self.request_new_img(), markers)
            rospy.sleep(self.corner_retry_delay)
        self.cached_corners[key] = corners
        return corners

    def _detected_markers(self, img):
        """Find all four markers in the image (ArUco), bounded by retries."""
        markers = detect_markers(img, self.marker_type)
        marker_attempts = 0
        while len(markers) != 4 and not rospy.is_shutdown():
            marker_attempts += 1
            if marker_attempts > self.marker_max_attempts:
                raise RuntimeError(
                    f"expected 4 markers but found {len(markers)} after "
                    f"{self.marker_max_attempts} attempts"
                )
            rospy.logdebug(f"marker is missing. Now found {len(markers)}.")
            img = self.request_new_img()
            markers = detect_markers(img, self.marker_type)
            rospy.sleep(0.1)
        return markers

    def preprocess_img(self, img, pose, refresh_geometry=False):
        key = self._geom_key(pose)
        if refresh_geometry:
            self.cached_markers.pop(key, None)
            self.cached_corners.pop(key, None)

        if key not in self.cached_markers:
            markers = self._detected_markers(img)
            self.cached_markers[key] = markers
        else:
            markers = self.cached_markers[key]

        img = crop_by_marker(img, markers)
        corners = self.decide_corner(img, markers, pose, update=refresh_geometry)
        img, img_scale = resize_image(self.corner_cfg, img)
        corners = corners * img_scale
        return img, corners

    def onestep_recog(self, img, color=chess.WHITE, refresh_geometry=False, augment="none"):
        pose = self.cur_robo_pose
        img, corners = self.preprocess_img(img, pose, refresh_geometry)
        # Augment only the classifier input; geometry above was solved on the
        # clean image so corners stay valid.
        img = apply_augmentation(img, augment)
        board, *_ = self.chess_rec.robo_predict(img, corners, color)
        return board

    def recognize_board(
        self,
        camera_pose="low",
        next_turn="",
        refresh_geometry=False,
        camera_side="",
        augment="none",
    ):
        self.debug_image_paths = []
        self.last_confidence = -1.0
        self.last_ambiguous_squares = []
        # Select the lens for every capture in this recognition (left default).
        self.cur_side = camera_side or CAMERA
        self.move_to_pose(camera_pose or "low")
        img = self.request_new_img()
        color = chess.BLACK if next_turn == "b" else chess.WHITE

        board = self.onestep_recog(
            img, color=color, refresh_geometry=refresh_geometry, augment=augment
        )

        self.last_confidence, self.last_ambiguous_squares = summarize_confidence(
            self.chess_rec.occupancy_confidence,
            self.chess_rec.piece_confidence,
            self.confidence_threshold,
            squares=self.chess_rec._squares,
        )

        if next_turn == "b":
            board.turn = chess.BLACK
        elif next_turn == "w":
            board.turn = chess.WHITE
        board.castling_rights = board.clean_castling_rights()
        return board
