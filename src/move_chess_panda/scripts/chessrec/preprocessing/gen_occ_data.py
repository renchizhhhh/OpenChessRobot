"""Script to create the occupancy classification dataset.

Based on the four chessboard corner points which are supplied as labels, this module is responsible for warping the image and cutting out the squares.
Note that before running this module requires the rendered dataset to be downloaded and split (see the :mod:`chesscog.data_synthesis` module for more information).

.. code-block:: console

    $ python -m chesscog.occupancy_classifier.create_dataset --help
    usage: create_dataset.py [-h]
    
    Create the dataset for occupancy classification.
    
    optional arguments:
      -h, --help  show this help message and exit
"""

from pathlib import Path
import cv2
from PIL import Image
import json
import numpy as np
import chess
import os
import shutil
from utili.recap import URI
import argparse

from tqdm import tqdm

from chessrec.preprocessing import sort_corner_points

RENDERS_DIR = URI("data://render")
OUT_DIR = URI("data://out_occupancy")
SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE
IMG_SIZE = BOARD_SIZE + 2 * SQUARE_SIZE

def crop_square(img: np.ndarray, square: chess.Square, turn: chess.Color) -> np.ndarray:
    """Crop a chess square from the warped input image for occupancy classification.

    Args:/
        img (np.ndarray): the warped input image
        square (chess.Square): the square to crop
        turn (chess.Color): the current player

    Returns:
        np.ndarray: the cropped square
    """
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    if turn == chess.WHITE:
        row, col = 7 - rank, file
    else:
        row, col = rank, 7 - file
    return img[int(SQUARE_SIZE * (row + .5)): int(SQUARE_SIZE * (row + 2.5)),
               int(SQUARE_SIZE * (col + .5)): int(SQUARE_SIZE * (col + 2.5))]


def warp_chessboard_image(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp the image of the chessboard onto a regular grid.

    Args:
        img (np.ndarray): the image of the chessboard
        corners (np.ndarray): pixel locations of the four corner points

    Returns:
        np.ndarray: the warped image
    """

    src_points = sort_corner_points(corners)
    dst_points = np.array([[SQUARE_SIZE, SQUARE_SIZE],  # top left
                           [BOARD_SIZE + SQUARE_SIZE, SQUARE_SIZE],  # top right
                           [BOARD_SIZE + SQUARE_SIZE, BOARD_SIZE + \
                            SQUARE_SIZE],  # bottom right
                           [SQUARE_SIZE, BOARD_SIZE + SQUARE_SIZE]  # bottom left
                           ], dtype=np.float32)
    transformation_matrix, _ = cv2.findHomography(src_points, dst_points)
    return cv2.warpPerspective(img, transformation_matrix, (IMG_SIZE, IMG_SIZE))


def _extract_squares_from_img(id: str, subset: str = "", input_dir: Path = RENDERS_DIR, output_dir: Path = OUT_DIR, save_color: bool = False):
    """Extract squares from a single image sample.

    Args:
        id (str): the id of the sample
        subset (str, optional): the subset of the sample. Defaults to "".
        input_dir (Path, optional): the directory of the sample. Defaults to RENDERS_DIR.
        output_dir (Path, optional): the directory to save the output. Defaults to OUT_DIR.
        save_color (bool, optional): whether to save the color of the subimage. Defaults to False.
    """
    img = cv2.imread(str(input_dir / subset / (id + ".png")))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with (input_dir / subset / (id + ".json")).open("r") as f:
        label = json.load(f)

    corners = np.array(label["corners"], dtype=np.float32)
    unwarped = warp_chessboard_image(img, corners)
    board = chess.Board(label["fen"])

    for square in chess.SQUARES:
        if save_color:
            if board.piece_at(square) is None:
                target_class = "empty" 
            elif board.color_at(square) == chess.WHITE:
                target_class = "white" 
            else:
                target_class = "black"
        else:
            target_class = "empty" if board.piece_at(square) is None else "occupied"
        piece_img = crop_square(unwarped, square, label["white_turn"])
        with Image.fromarray(piece_img, "RGB") as piece_img:
            piece_img.save(output_dir / subset / target_class /
                           f"{id}_{chess.square_name(square)}.png")


def create_dataset(input_dir: Path = RENDERS_DIR, output_dir: Path = OUT_DIR, save_color: bool = False, by_pose: bool = False):
    """Create the occupancy classification dataset.

    Args:
        input_dir (Path, optional): the input folder of the rendered images. Defaults to ``data://render``.
        output_dir (Path, optional): the output folder. Defaults to ``data://occupancy``.
        save_color (bool, optional): if save to 3 classes [empty, white, black]
        by_pose (bool, optional): create subfolders by camera poses. Defaults is False. 
    """
    if not os.path.exists(input_dir):
        raise IndexError("Input data path not exist")
    if by_pose:
        folders = ("low", "left", "right")
    else:
        folders = ("train", "val", "test")
    for subset in folders:
        samples = list((input_dir / subset).glob("*.png"))
        if save_color: 
            for c in ("empty", "white", "black"):
                folder = output_dir / subset / c
                shutil.rmtree(folder, ignore_errors=True)
                os.makedirs(folder, exist_ok=True)
        else:
            for c in ("empty", "occupied"):
                folder = output_dir / subset / c
                shutil.rmtree(folder, ignore_errors=True)
                os.makedirs(folder, exist_ok=True)
        for img_file in tqdm(samples): 
            _extract_squares_from_img(img_file.stem, subset,
                                        input_dir, output_dir, save_color)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Create the dataset for occupancy classification.")
    # parser.add_argument("--savecolor", help="whether to save the colors in occupancy (default: False)",
    #                     type=bool, choices=[True, False], default=False)
    # args = parser.parse_args()
    # create_dataset(save_color = args.savecolor)
    create_dataset(save_color = True)
