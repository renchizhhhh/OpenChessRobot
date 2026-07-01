#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ocr_runtime.recognition_confidence import (
    square_name,
    summarize_confidence,
)


class SquareNameTests(unittest.TestCase):
    def test_known_square_indices(self):
        # python-chess convention: A1=0, B1=1, ..., E2=12, H8=63.
        self.assertEqual(square_name(0), "A1")
        self.assertEqual(square_name(12), "E2")
        self.assertEqual(square_name(63), "H8")


class SummarizeConfidenceTests(unittest.TestCase):
    def test_all_confident_board_has_no_ambiguous_squares(self):
        occupancy = np.full(64, 0.95)
        pieces = np.zeros(64)
        pieces[:16] = 0.9  # first 16 squares occupied and confident

        overall, ambiguous = summarize_confidence(occupancy, pieces, threshold=0.7)

        self.assertEqual(ambiguous, [])
        # 16 occupied squares use min(0.95, 0.9)=0.9; the other 48 stay at 0.95.
        self.assertAlmostEqual(overall, (16 * 0.9 + 48 * 0.95) / 64, places=5)

    def test_low_occupancy_confidence_marks_square_ambiguous(self):
        occupancy = np.full(64, 0.95)
        pieces = np.zeros(64)
        occupancy[0] = 0.55  # a1 below threshold

        overall, ambiguous = summarize_confidence(occupancy, pieces, threshold=0.7)

        self.assertEqual(ambiguous, ["A1"])
        self.assertLess(overall, 0.95)

    def test_occupied_square_uses_weaker_of_occupancy_and_piece(self):
        occupancy = np.full(64, 0.99)
        pieces = np.zeros(64)
        pieces[12] = 0.40  # e2 occupied but piece classification weak

        overall, ambiguous = summarize_confidence(occupancy, pieces, threshold=0.7)

        self.assertEqual(ambiguous, ["E2"])

    def test_empty_square_ignores_zero_piece_confidence(self):
        occupancy = np.full(64, 0.8)
        pieces = np.zeros(64)  # every square empty

        overall, ambiguous = summarize_confidence(occupancy, pieces, threshold=0.7)

        # Empty squares must not be flagged just because piece confidence is 0.
        self.assertEqual(ambiguous, [])
        self.assertAlmostEqual(overall, 0.8, places=5)

    def test_square_names_follow_index_order(self):
        occupancy = np.full(64, 0.95)
        pieces = np.zeros(64)
        occupancy[63] = 0.10  # h8

        _, ambiguous = summarize_confidence(occupancy, pieces, threshold=0.7)

        self.assertEqual(ambiguous, ["H8"])

    def test_mismatched_inputs_return_sentinel(self):
        overall, ambiguous = summarize_confidence(
            np.full(64, 0.9), np.zeros(10), threshold=0.7)

        self.assertEqual(overall, -1.0)
        self.assertEqual(ambiguous, [])

    def test_empty_inputs_return_sentinel(self):
        overall, ambiguous = summarize_confidence([], [], threshold=0.7)

        self.assertEqual(overall, -1.0)
        self.assertEqual(ambiguous, [])


if __name__ == "__main__":
    unittest.main()
