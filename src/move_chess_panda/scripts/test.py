import matplotlib.pyplot as plt

from utili.camera_config import Camera
from utili.recap import CfgNode as CN

import chess
import cv2
from pathlib import Path

from chessrec.recognizer.recognizer import ChessRecognizer
from chessrec.preprocessing.detect_corners import resize_image, find_corners

camera = Camera(ip="192.168.0.106", port=30000, name="2")

img = camera.get_img()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



model_path = Path("/home/charles/panda/catkin_ws/src/move_chess_panda/scripts/chessrec/runs/aftermove")
recognizer = ChessRecognizer(model_path)

corner_detection_cfg = CN.load_yaml_with_base("config://corner_detection.yaml")

img, img_scale = resize_image(corner_detection_cfg, img)
corners = find_corners(corner_detection_cfg, img)
plt.imshow(img)
plt.scatter(*corners.T, c="r")
plt.axis("off")
plt.show()


board, *_ = recognizer.robo_predict(img, corners, chess.WHITE)

print(board)
print()