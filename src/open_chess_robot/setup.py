#!/usr/bin/env python3

from catkin_pkg.python_setup import generate_distutils_setup
from distutils.core import setup


setup_args = generate_distutils_setup(
    packages=[
        "chessrec",
        "chessrec.classifier",
        "chessrec.core",
        "chessrec.core.dataset",
        "chessrec.core.training",
        "chessrec.preprocessing",
        "chessrec.recognizer",
        "engine",
        "ocr_runtime",
        "utili",
        "utili.recap",
    ],
    package_dir={"": "scripts"},
    py_modules=[
        "chess_commander",
        "chess_robo_player",
        "data_collect_chess_commander",
        "data_collect_chess_robot",
        "eva_chess_commander",
        "eva_chess_robot",
        "llm_chess_commander",
        "llm_commentary_manager",
        "hri_chess_commander",
        "hri_chess_robot",
        "ros_param_manager",
    ],
)

setup(**setup_args)
