"""Module that brings together the whole recognition pipeline into a single class so it can be conveniently executed.

This module simultaneously acts as a script to perform a single inference:

.. code-block:: console

    $ python -m chessrec.recognition.recognition --help
    usage: recognition.py [-h] [--white] [--black] file
    
    Run the chess recognition pipeline on an input image
    
    positional arguments:
      file        path to the input image
    
    optional arguments:
      -h, --help  show this help message and exit
      --white     indicate that the image is from the white player's
                  perspective (default)
      --black     indicate that the image is from the black player's
                  perspective
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import chess
from chess import Status
import torch
from PIL import Image
import functools
import cv2
import argparse
import typing
from utili.recap import URI, CfgNode as CN

from chessrec.preprocessing import find_corners, resize_image
from chessrec.preprocessing import gen_occ_data, gen_pie_data

from chessrec.core import device, DEVICE
from chessrec.core.dataset import build_transforms, Datasets
from chessrec.core.dataset import name_to_piece
from chessrec.core.models import build_model

import chessrec.classifier.occ_models
# import chessrec.classifier.color_models
import chessrec.classifier.piece_models

class ChessRecognizer:
    """A class implementing the entire chess inference pipeline.

    Once you create an instance of this class, the CNNs are loaded into memory (possibly the GPU if available), so if you want to perform multiple inferences, they should all use one instance of this class for performance purposes.
    """

    _squares = list(chess.SQUARES)

    def __init__(self, classifiers_folder: Path = URI("models://")):
        """Constructor.
        Args:
            classifiers_folder (Path, optional): the path to the classifiers (supplying a different path is especially useful because the transfer learning classifiers are located at ``models://transfer_learning``). Defaults to ``models://``.
        """
        self._corner_detection_cfg = CN.load_yaml_with_base(
            "config://corner_detection.yaml")
        self._occupancy_cfg, self._occupancy_model = self._load_classifier(
            classifiers_folder / "occupancy_classifier", epoch="10")
        self._occupancy_transforms = build_transforms(
            self._occupancy_cfg, mode=Datasets.TEST)
        self._pieces_cfg, self._pieces_model = self._load_classifier(
            classifiers_folder / "piece_classifier", epoch="40")
        self._pieces_transforms = build_transforms(
            self._pieces_cfg, mode=Datasets.TEST)
        self._piece_classes = np.array(list(map(name_to_piece,
                                                self._pieces_cfg.DATASET.CLASSES)))

    @classmethod
    def _load_classifier(cls, path: Path, epoch: str):
        """Load model by state_dict
        """
        if epoch == "":
            model_file = next(iter(path.glob("*.pt")))
            yaml_file = next(iter(path.glob("*.yaml")))
        else:
            temp = path.glob(f"*Epoch{epoch}.pt")
            model_file = next(iter(temp))
            yaml_file = next(iter(path.glob("*.yaml")))
        cfg = CN.load_yaml_with_base(yaml_file)
        model = build_model(cfg)
        model.load_state_dict(torch.load(model_file, map_location=DEVICE))
        print(f"loading model : {model_file.__str__()}")
        model = device(model)
        model.eval()
        return cfg, model
    
    def _classify_occupancy(self, img: np.ndarray, turn: chess.Color, corners: np.ndarray) -> np.ndarray:
        warped = gen_occ_data.warp_chessboard_image(
            img, corners)
        square_imgs = map(functools.partial(
            gen_occ_data.crop_square, warped, turn=turn), self._squares)
        square_imgs = map(Image.fromarray, square_imgs)
        square_imgs = map(self._occupancy_transforms, square_imgs)
        square_imgs = list(square_imgs)
        square_imgs = torch.stack(square_imgs)
        square_imgs = device(square_imgs)
        occupancy = self._occupancy_model(square_imgs)
        occupancy = occupancy.argmax(
            axis=-1) == self._occupancy_cfg.DATASET.CLASSES.index("occupied")
        occupancy = occupancy.cpu().numpy()
        return occupancy

    def _classify_pieces(self, img: np.ndarray, turn: chess.Color, corners: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
        occupied_squares = np.array(self._squares)[occupancy]
        warped = gen_pie_data.warp_chessboard_image(
            img, corners)
        piece_imgs = map(functools.partial(
            gen_pie_data.crop_square, warped, turn=turn), occupied_squares)
        piece_imgs = map(Image.fromarray, piece_imgs)
        piece_imgs = map(self._pieces_transforms, piece_imgs)
        piece_imgs = list(piece_imgs)
        piece_imgs = torch.stack(piece_imgs)
        piece_imgs = device(piece_imgs)
        pieces = self._pieces_model(piece_imgs)
        pieces = pieces.argmax(axis=-1).cpu().numpy()
        pieces = self._piece_classes[pieces]
        all_pieces = np.full(len(self._squares), None, dtype=np.object_)
        all_pieces[occupancy] = pieces
        return all_pieces

    def prepredict(self, img: np.ndarray, turn: chess.Color = chess.WHITE) -> typing.Tuple[chess.Board, np.ndarray]:
        """Perform an inference.

        Args:
            img (np.ndarray): the input image (RGB)
            turn (chess.Color, optional): the current player. Defaults to chess.WHITE.

        Returns:
            typing.Tuple[chess.Board, np.ndarray]: the predicted position on the board and the four corner points
        """
        with torch.no_grad():
            img, _ = resize_image(self._corner_detection_cfg, img)
            corners = find_corners(self._corner_detection_cfg, img)
            occupancy = self._classify_occupancy(img, turn, corners)
            return occupancy

    def predict(self, img: np.ndarray, turn: chess.Color = chess.WHITE) -> typing.Tuple[chess.Board, np.ndarray]:
        """Perform an inference.

        Args:
            img (np.ndarray): the input image (RGB)
            turn (chess.Color, optional): the current player. Defaults to chess.WHITE.

        Returns:
            typing.Tuple[chess.Board, np.ndarray]: the predicted position on the board and the four corner points
        """
        with torch.no_grad():
            img, img_scale = resize_image(self._corner_detection_cfg, img)
            corners = find_corners(self._corner_detection_cfg, img)
            occupancy = self._classify_occupancy(img, turn, corners)
            pieces = self._classify_pieces(img, turn, corners, occupancy)

            board = chess.Board()
            board.clear_board()
            for square, piece in zip(self._squares, pieces):
                if piece:
                    board.set_piece_at(square, piece)
            corners = corners / img_scale
            return board, corners, occupancy


    def robo_prepredict(self, img: np.ndarray, corners: np.ndarray, turn: chess.Color = chess.WHITE) -> typing.Tuple[chess.Board, np.ndarray]:
        """Perform an inference.

        Args:
            img (np.ndarray): the input image (RGB)
            turn (chess.Color, optional): the current player. Defaults to chess.WHITE.

        Returns:
            typing.Tuple[chess.Board, np.ndarray]: the predicted position on the board and the four corner points
        """
        with torch.no_grad():
            img, _ = resize_image(self._corner_detection_cfg, img)
            corners = find_corners(self._corner_detection_cfg, img)
            occupancy = self._classify_occupancy(img, turn, corners)
            return occupancy

    def robo_postpredict(self, img: np.ndarray, corners: np.ndarray, occupancy: np.ndarray, turn: chess.Color = chess.WHITE) -> typing.Tuple[chess.Board, np.ndarray]:
        """Perform an inference.

        Args:
            img (np.ndarray): the input image (RGB)
            turn (chess.Color, optional): the current player. Defaults to chess.WHITE.

        Returns:
            typing.Tuple[chess.Board, np.ndarray]: the predicted position on the board and the four corner points
        """
        with torch.no_grad():
            pieces = self._classify_pieces(img, turn, corners, occupancy)

            board = chess.Board()
            board.clear_board()
            for square, piece in zip(self._squares, pieces):
                if piece:
                    board.set_piece_at(square, piece)
            return board, occupancy


    def robo_predict(self, img: np.ndarray, corners: np.ndarray, turn: chess.Color = chess.WHITE) -> typing.Tuple[chess.Board, np.ndarray]:
        """Perform an inference.

        Args:
            img (np.ndarray): the input image (RGB)
            turn (chess.Color, optional): the current player. Defaults to chess.WHITE.

        Returns:
            typing.Tuple[chess.Board, np.ndarray]: the predicted position on the board and the four corner points
        """
        with torch.no_grad():
            occupancy = self._classify_occupancy(img, turn, corners)
            pieces = self._classify_pieces(img, turn, corners, occupancy)

            board = chess.Board()
            board.clear_board()
            for square, piece in zip(self._squares, pieces):
                if piece:
                    board.set_piece_at(square, piece)
            return board, occupancy

class TimedChessRecognizer(ChessRecognizer):
    """A subclass of :class:`ChessRecognizer` that additionally records the time taken for each step of the pipeline during inference.
    """

    def predict(self, img: np.ndarray, turn: chess.Color = chess.WHITE) -> typing.Tuple[chess.Board, np.ndarray, dict]:
        """Perform an inference.

        Args:
            img (np.ndarray): the input image (RGB)
            turn (chess.Color, optional): the current player. Defaults to chess.WHITE.

        Returns:
            typing.Tuple[chess.Board, np.ndarray, dict]: the predicted position on the board, the four corner points, and a dict containing the time taken for each stage of the inference pipeline
        """

        from timeit import default_timer as timer
        with torch.no_grad():
            t1 = timer()
            img, img_scale = resize_image(self._corner_detection_cfg, img)
            corners = find_corners(self._corner_detection_cfg, img)
            t2 = timer()
            occupancy = self._classify_occupancy(img, turn, corners)
            t3 = timer()
            pieces = self._classify_pieces(img, turn, corners, occupancy)
            t4 = timer()

            board = chess.Board()
            board.clear()
            board.turn = turn
            for square, piece in zip(self._squares, pieces):
                if piece:
                    board.set_piece_at(square, piece)
            corners = corners / img_scale
            t5 = timer()

            times = {
                "corner_detection": t2-t1,
                "occupancy_classification": t3-t2,
                "piece_classification": t4-t3,
                "prepare_results": t5-t4
            }

            return board, corners, times


def main(classifiers_folder: Path = URI("models://")):
    """Main method for running inference from the command line.

    Args:
        classifiers_folder (Path, optional): the path to the classifiers (supplying a different path is especially useful because the transfer learning classifiers are located at ``models://transfer_learning``). Defaults to ``models://``.
    """

    parser = argparse.ArgumentParser(
        description="Run the chess recognition pipeline on an input image")
    parser.add_argument("file", help="path to the input image", type=str)
    parser.add_argument(
        "--white", help="indicate that the image is from the white player's perspective (default)", action="store_true", dest="color")
    parser.add_argument(
        "--black", help="indicate that the image is from the black player's perspective", action="store_false", dest="color")
    parser.set_defaults(color=True)
    args = parser.parse_args()

    img = cv2.imread(str(URI(args.file)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    recognizer = ChessRecognizer(classifiers_folder)
    board, *_ = recognizer.predict(img, args.color)

    print(board)
    print()
    # print(
    #     f"You can view this position at https://lichess.org/editor/{board.board_fen()}")

    if board.status() != Status.VALID:
        print()
        print("WARNING: The predicted chess position is not legal according to the rules of chess.")
        print("         You might want to try again with another picture.")

if __name__ == "__main__":
    main()
