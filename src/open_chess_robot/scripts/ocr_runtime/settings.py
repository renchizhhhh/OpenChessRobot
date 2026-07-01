from .paths import resource_path


# for chess_robot_player.py
CAMERA = "left"
SQUARE_SIZE = 0.050
Z_ABOVE_BOARD = 0.25  # 0.25
Z_TO_PIECE = 0.095  # 0.095  # 0.15
Z_DROP = 0.005
X_OFFSET = -0.0388 #-0.0539
Y_OFFSET = 0.0038 # +0.0086

# Vision-based grasp correction (opt-in, see pick()/measure_pick_offset). These
# are the defaults for the detection tuning; each is still read via
# rospy.get_param("/open_chess_robot/pick/...", <constant>) so it can be overridden
# live with `rosparam set` during on-robot bring-up. The on/off switch itself is a
# launch arg (hri_chess_exe.launch vision_refine:=true), not a constant here.
# PICK_SETTLE_TIME: settle (s) after the look-down move before grabbing the frame,
# so the ZED returns a sharp (un-blurred) image for base-circle detection.
PICK_SETTLE_TIME = 0.1
# Base-circle search band as a fraction of the detected square side.
PICK_RADIUS_MIN_RATIO = 0.22
PICK_RADIUS_MAX_RATIO = 0.35
# CLAHE contrast clip and HoughCircles accumulator thresholds for the base.
PICK_CIRCLE_CLAHE_CLIP = 3.0
PICK_CIRCLE_HOUGH_PARAM1 = 100
PICK_CIRCLE_HOUGH_PARAM2 = 27

# Hand-eye calibration: fixed transform from the ZED camera frame to the robot
# flange, used to lift detected marker positions into the arm base frame
# (chess_robo_player.update_world_positions). These are physically measured
# values and are the single authoritative source for the transform; testing
# showed they are more accurate than the easy_handeye calibration output.
# Re-measure and edit here if the camera mount changes. R is ZXY Euler degrees;
# T is metres and depends on which ZED lens (left/right) is in use.
HANDEYE_R_FLANGE2ZED = [45, 2.5, 0]
HANDEYE_T_FLANGE2ZED_LEFT = [-0.06, -0.06, 0.02]
HANDEYE_T_FLANGE2ZED_RIGHT = [0.06, -0.06, 0.02]

# Minimum Cartesian-path fraction that counts as a successful plan. compute_
# cartesian_path returns the fraction of the requested path it could plan; below
# this, execute_path treats the move as failed instead of running a partial path.
CARTESIAN_SUCCESS_FRACTION = 0.99
# fast
ACC = 0.6  # 0.8
VEL = 0.6  # 0.7

# for camera_config.py
MARKER_SIZE = 0.025
MARKER_TYPE = "DICT_4X4_50"
CAM_IP = "192.168.0.106"

# for chess_commander.py
MODEL_PATH = str(resource_path("scripts", "chessrec", "runs", "runtime"))

MODE = "stockfish15"  # stockfish15 | maia | stockfish16

ELO = 2000
DEPTH = 15

# Startup recognition defaults for hri_chess_commander.py
INITIAL_BOARD_MAX_ATTEMPTS = 3
RECOGNITION_MAX_FAILURES = 20
RECOGNITION_CAMERA_RESET_INTERVAL = 5
STARTUP_POSE_WAIT_TIMEOUT = 10.0
RECOGNITION_BACKEND = "service"  # service | direct
RECOGNIZE_BOARD_SERVICE = "/recognize_board"
RECOGNIZE_BOARD_TIMEOUT = 10.0

# Recognition marker detection: how many fresh frames a single recognition
# attempt will try to find all four markers before giving up and letting the
# fallback ladder move to the next attempt. The low recognition pose decodes
# markers less reliably than the high localization pose (shallow angle), so a low
# cap fails fast instead of burning ~9s of re-captures on a frame set that will
# not recover.
RECOGNITION_MARKER_MAX_ATTEMPTS = 8

# Recognition retry fallback. When a recognition attempt yields an invalid board,
# the commander retries down this ladder of (camera_side, augmentation) instead
# of escalating to the two-step occupancy pipeline (which testing showed degrades
# the near/white half). All attempts stay at the low pose the models were trained
# on. The right lens is a parallax second opinion that can resolve confident
# king-vs-queen confusions; the augmentations are a weaker non-color safety net.
# Each combination is tried once - re-running an identical input is deterministic.
RECOGNITION_FALLBACK_PLAN = [
    ("left", "none"),
    ("right", "none"),
    ("left", "contrast"),
    ("right", "contrast"),
    ("left", "unsharp"),
    ("right", "unsharp"),
]

# Board localization: number of marker observations averaged per update.
MARKER_SAMPLES = 5

# The four board ArUco marker ids. Detections with any other id (spurious decodes
# from glare/noise) are discarded so they cannot corrupt board localization.
EXPECTED_MARKER_IDS = (1, 2, 3, 4)

# Marker detection robustness: when the cheap single-pass detection is missing an
# expected marker, retry on a few light image variants (contrast-equalized,
# sharpened) and union the results. The common case stays single-pass; spurious
# ids decoded from the variants are dropped by the EXPECTED_MARKER_IDS filter.
MARKER_DETECTION_AUGMENT = True

# Board localization: how many times startup retries the full marker-detection
# pipeline (with a camera reset between attempts) before aborting, so a transient
# single-marker dropout does not crash the robot node.
LOCALIZATION_MAX_ATTEMPTS = 3

# Trajectory execution recovery: how many times a single cartesian move, after a
# libfranka reflex aborts it (e.g. acceleration_discontinuity), instantly sends an
# error-recovery goal to clear the transient fault and replans/retries from where
# the arm stopped. So the move is attempted up to 1 + this many times. This only
# papers over the *small* recoverable faults - once these recoveries are spent we
# give up and raise so a genuine problem is not masked.
EXECUTION_RECOVERY_ATTEMPTS = 3

# Corner detection retry policy. find_corners is a heavy internal RANSAC, and the
# scene is static, so a small number of re-cropped retries is enough; a high cap
# only burns frames when the detection systematically disagrees with the markers.
CORNER_MAX_TRIALS = 12
CORNER_RETRY_DELAY = 0.1

# Tolerance for validating detected board corners against marker positions,
# expressed as a fraction of the marker bounding-box span so it scales with
# image resolution and how large the board appears in frame.
CORNER_MARKER_TOL_RATIO = 0.02

# Recognition confidence: squares whose softmax confidence falls below this value
# are reported as ambiguous in the recognition service response.
CONFIDENCE_AMBIGUOUS_THRESHOLD = 0.7

# Poses for chess robot execution
LOW_CAM_JOINTS = [
    -0.0002861509839998449,
    -1.5649661966610398,
    0.001571490184083701,
    -2.441184105685472,
    0.009147199263040286,
    1.397496153041122,
    0.785220506046326,
    0.017489660531282425,
    0.017489660531282425,
]

HIGH_CAM_JOINTS = [
    0.00015898450553431174,
    -0.2909451232551228,
    0.00025142004909806066,
    -1.5919965450139177,
    0.0003605799753592121,
    1.3144894863848386,
    0.785866691948467,
    0.01748703420162201,
    0.01748703420162201,
]

LEFT_CAM_JOINTS = [
    -0.06616944939147938,
    -1.5526289022237463,
    0.5461460406923571,
    -2.424159722067252,
    0.2610135964060131,
    1.4384150299219005,
    1.0994914650048637,
    0.0407392717897892,
    0.0407392717897892,
]

RIGHT_CAM_JOINTS = [
    0.030806058988522757,
    -1.4778349875477093,
    -0.6622232789328013,
    -2.420708685489011,
    -0.30581781720616746,
    1.4800959099526354,
    0.2875588180604471,
    0.040736645460128784,
    0.040736645460128784,
]

LOOK_AT_HUMAN = [
    -0.013366096853392505,
    -1.5770862280739022,
    0.015311006135396885,
    -2.4558152854993334,
    0.009796439154703237,
    1.7455670635771343,
    0.86815584582908,
    0.04074616730213165,
    0.04074616730213165,
]

LOOK_AWAY = [
    -0.013417151374830709,
    -1.5770978355178171,
    0.015234067152183702,
    -2.455841507907426,
    -0.4892774688835677,
    1.397887450594359,
    0.8681612076450472,
    0.04074616730213165,
    0.04074616730213165,
]

LOOK_AWAY_R = [
    -0.013417151374830709,
    -1.5770978355178171,
    0.015234067152183702,
    -2.455841507907426,
    0.4892774688835677,
    1.397887450594359,
    0.8681612076450472,
    0.04074616730213165,
    0.04074616730213165,
]

ROTATE_LEFT = [
    0.00426556653928077,
    -1.5657198262465628,
    0.0070904803803484686,
    -2.4536077158153278,
    0.00771298555822836,
    1.4017696044445036,
    0.3001453456124784,
]


ROTATE_RIGHT = [
    0.00426556653928077,
    -1.5657198262465628,
    0.0070904803803484686,
    -2.4536077158153278,
    0.00771298555822836,
    1.4017696044445036,
    0.9822690132626678,
]
