#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ocr_runtime.move_sequences import (
    CAPTURE_BOX_SQUARE,
    CAPTURE_BOX_Y_OFFSET,
    EN_PASSANT_CAPTURE_BOX_Y_OFFSET,
    castling_rook_move,
    en_passant_capture_drop,
    en_passant_capture_square,
    normal_capture_drop,
    parse_encoded_move,
    promotion_prompt,
)
from ocr_runtime.startup_checks import (
    InitialBoardDecision,
    STARTING_FEN,
    classify_initial_board,
)


class FakeStatus:
    def __init__(self, name):
        self.name = name


class FakeBoard:
    def __init__(self, fen, status_name):
        self._fen = fen
        self._status = FakeStatus(status_name)

    def fen(self):
        return self._fen

    def status(self):
        return self._status


class EncodedMoveTests(unittest.TestCase):
    def test_parse_normal_move(self):
        move = parse_encoded_move("e2e400000")

        self.assertEqual(move.uci, "e2e4")
        self.assertEqual(move.start, "E2")
        self.assertEqual(move.end, "E4")
        self.assertFalse(move.is_capture)
        self.assertFalse(move.is_promotion)
        self.assertIsNone(move.promotion_piece)

    def test_parse_promotion_move(self):
        move = parse_encoded_move("e7e8q00001")

        self.assertEqual(move.uci, "e7e8q")
        self.assertEqual(move.start, "E7")
        self.assertEqual(move.end, "E8")
        self.assertTrue(move.is_promotion)
        self.assertEqual(move.promotion_piece, "q")

    def test_parse_knight_hop_flag(self):
        move = parse_encoded_move("g1f310000")

        self.assertTrue(move.is_hop)
        self.assertFalse(move.is_capture)

    def test_parse_capture_flag(self):
        move = parse_encoded_move("e4d501000")

        self.assertTrue(move.is_capture)
        self.assertFalse(move.is_en_passant)

    def test_parse_castling_flag(self):
        move = parse_encoded_move("e1g100100")

        self.assertTrue(move.is_castling)
        self.assertEqual(move.start, "E1")
        self.assertEqual(move.end, "G1")

    def test_parse_en_passant_flags(self):
        move = parse_encoded_move("e5d601010")

        self.assertTrue(move.is_capture)
        self.assertTrue(move.is_en_passant)

    def test_reject_promotion_without_piece(self):
        with self.assertRaises(ValueError):
            parse_encoded_move("e7e800001")

    def test_reject_promotion_piece_without_flag(self):
        with self.assertRaises(ValueError):
            parse_encoded_move("e7e8q00000")


class PhysicalSequenceHelperTests(unittest.TestCase):
    def test_castling_rook_moves(self):
        self.assertEqual(castling_rook_move("E1", "G1"), ("H1", "F1"))
        self.assertEqual(castling_rook_move("E1", "C1"), ("A1", "D1"))
        self.assertEqual(castling_rook_move("E8", "G8"), ("H8", "F8"))
        self.assertEqual(castling_rook_move("E8", "C8"), ("A8", "D8"))

    def test_en_passant_capture_square(self):
        self.assertEqual(en_passant_capture_square("E5", "D6"), "D5")
        self.assertEqual(en_passant_capture_square("D4", "E3"), "E4")

    def test_capture_box_drop_policy(self):
        normal = normal_capture_drop()
        en_passant = en_passant_capture_drop()

        self.assertEqual(normal.square, CAPTURE_BOX_SQUARE)
        self.assertEqual(normal.y_offset, CAPTURE_BOX_Y_OFFSET)
        self.assertEqual(en_passant.square, CAPTURE_BOX_SQUARE)
        self.assertEqual(en_passant.y_offset, EN_PASSANT_CAPTURE_BOX_Y_OFFSET)

    def test_promotion_prompt(self):
        white_queen = parse_encoded_move("e7e8q00001")
        black_knight = parse_encoded_move("b2b1n00001")

        self.assertIn("E8", promotion_prompt(white_queen))
        self.assertIn("WHITE QUEEN", promotion_prompt(white_queen))
        self.assertIn("B1", promotion_prompt(black_knight))
        self.assertIn("BLACK KNIGHT", promotion_prompt(black_knight))


class StartupDecisionTests(unittest.TestCase):
    def test_standard_initial_board_is_accepted(self):
        board = FakeBoard(STARTING_FEN, "VALID")

        check = classify_initial_board(board)

        self.assertEqual(check.decision, InitialBoardDecision.INITIAL_BOARD_OK)
        self.assertEqual(check.status_name, "VALID")

    def test_valid_custom_board_requires_operator_review(self):
        board = FakeBoard("8/8/8/8/8/8/8/K6k w - - 0 1", "VALID")

        check = classify_initial_board(board)

        self.assertEqual(check.decision, InitialBoardDecision.REVIEW_CUSTOM_BOARD)

    def test_invalid_board_retries_recognition(self):
        board = FakeBoard("8/8/8/8/8/8/8/8 w - - 0 1", "NO_WHITE_KING")

        check = classify_initial_board(board)

        self.assertEqual(check.decision, InitialBoardDecision.RETRY_RECOGNITION)
        self.assertEqual(check.status_name, "NO_WHITE_KING")


class LaunchModeTests(unittest.TestCase):
    def test_all_launch_files_parse(self):
        for launch_file in sorted((ROOT / "launch").glob("*.launch")):
            with self.subTest(launch_file=launch_file.name):
                ET.parse(launch_file)
        for launch_file in sorted((ROOT / "launch" / "includes").glob("*.launch")):
            with self.subTest(launch_file=launch_file.name):
                ET.parse(launch_file)

    def test_deleted_movement_manager_is_not_launched(self):
        for launch_file in sorted((ROOT / "launch").glob("*.launch")):
            with self.subTest(launch_file=launch_file.name):
                text = launch_file.read_text()
                self.assertNotIn("chess_movement_manager.py", text)

    def test_recovery_is_optional_in_legacy_launches(self):
        for filename in ("data_collection.launch", "eva_chess.launch"):
            with self.subTest(filename=filename):
                root = ET.parse(ROOT / "launch" / filename).getroot()
                args = {
                    element.attrib["name"]: element.attrib
                    for element in root.findall("arg")
                }
                self.assertEqual(args["enable_recovery"]["default"], "false")
                recovery_nodes = [
                    element for element in root.findall("node")
                    if element.attrib.get("type") == "chess_robot_recovery.py"
                ]
                self.assertEqual(len(recovery_nodes), 1)
                self.assertEqual(recovery_nodes[0].attrib["if"], "$(arg enable_recovery)")

    def test_hri_executable_nodes_are_selectable(self):
        root = ET.parse(ROOT / "launch" / "hri_chess_exe.launch").getroot()
        args = {
            element.attrib["name"]: element.attrib["default"]
            for element in root.findall("arg")
        }
        params = {
            element.attrib["name"]: element.attrib["value"]
            for element in root.findall("param")
        }
        commander_nodes = [
            element for element in root.findall("node")
            if element.attrib.get("type") == "hri_chess_commander.py"
        ]
        self.assertEqual(args["launch_param_manager"], "true")
        self.assertEqual(args["launch_robot"], "true")
        self.assertEqual(args["launch_commander"], "true")
        self.assertEqual(args["startup_mode"], "game")
        self.assertEqual(args["recognition_backend"], "service")
        # Numeric recognition/localization tuning now lives in settings.py and is
        # intentionally not duplicated as launch args/params.
        self.assertNotIn("recognition_max_failures", args)
        self.assertNotIn("initial_board_max_attempts", args)
        self.assertEqual(
            params["/open_chess_robot/startup_mode"],
            "$(arg startup_mode)",
        )
        self.assertEqual(
            params["/open_chess_robot/recognition/backend"],
            "$(arg recognition_backend)",
        )
        self.assertEqual(len(commander_nodes), 1)
        self.assertIn("startup_mode", commander_nodes[0].attrib["if"])
        self.assertIn("game", commander_nodes[0].attrib["if"])

    def test_localize_only_robot_does_not_subscribe_to_chess_moves(self):
        robot_text = (ROOT / "scripts" / "hri_chess_robot.py").read_text()

        self.assertIn('"localize_only"', robot_text)
        self.assertIn('startup_mode == "game"', robot_text)
        self.assertIn('rospy.Subscriber("/chess_move"', robot_text)


if __name__ == "__main__":
    unittest.main()
