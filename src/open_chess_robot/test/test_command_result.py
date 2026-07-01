#!/usr/bin/env python3
"""Tests for the /chess_move_res acknowledgement protocol (Phase C.4 task 1).

Pins the rules the robot publisher and commander subscriber share: a result only
completes the move it actually acknowledges, promotion (5-char) moves match, and
failure results are distinguishable from success. No ROS.
"""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ocr_runtime.move_result import (
    encode_result,
    result_matches,
    result_is_failure,
)

# A normal and a promotion move as the robot would publish them: UCI + 5 flags.
NORMAL = "e2e400000"
PROMOTION = "e7e8q00001"


class EncodeResultTests(unittest.TestCase):
    def test_success_string_is_unchanged_legacy_format(self):
        self.assertEqual(encode_result(NORMAL, success=True), "e2e400000 is finished")

    def test_failure_string_is_distinct(self):
        self.assertEqual(encode_result(NORMAL, success=False), "e2e400000 failed")


class ResultMatchesTests(unittest.TestCase):
    def test_matches_normal_move(self):
        self.assertTrue(result_matches("e2e4", encode_result(NORMAL)))

    def test_matches_promotion_move(self):
        # The latent bug this fixes: a 5-char UCI must still match its result.
        self.assertTrue(result_matches("e7e8q", encode_result(PROMOTION)))

    def test_rejects_unrelated_move(self):
        # The reported bug: a result for a different move must not complete ours.
        self.assertFalse(result_matches("d2d4", encode_result(NORMAL)))

    def test_empty_pending_move_never_matches(self):
        self.assertFalse(result_matches("", encode_result(NORMAL)))

    def test_failure_result_still_matches_its_move(self):
        self.assertTrue(result_matches("e2e4", encode_result(NORMAL, success=False)))


class ResultIsFailureTests(unittest.TestCase):
    def test_success_is_not_failure(self):
        self.assertFalse(result_is_failure(encode_result(NORMAL, success=True)))

    def test_failure_is_detected(self):
        self.assertTrue(result_is_failure(encode_result(NORMAL, success=False)))

    def test_failure_detected_for_promotion(self):
        self.assertTrue(result_is_failure(encode_result(PROMOTION, success=False)))


if __name__ == "__main__":
    unittest.main()
