#!/usr/bin/env python3
"""Offline tests for centre-square and piece-base detection."""

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ocr_runtime.piece_offset import (
    detect_base_circle,
    detect_center_square,
)


def synthetic_look_image(omit_vertical=None, omit_horizontal=None):
    """A near-nadir look image with realistic duplicate and distractor edges."""
    height, width, pitch = 720, 960, 150
    x_origin, y_origin = 105, -15
    image = np.full((height, width, 3), 125, dtype=np.uint8)

    for row in range(6):
        for column in range(7):
            value = 205 if (row + column) % 2 else 65
            cv2.rectangle(
                image,
                (x_origin + column * pitch, y_origin + row * pitch),
                (x_origin + (column + 1) * pitch,
                 y_origin + (row + 1) * pitch),
                (value, value, value), -1)
    for x in range(x_origin, x_origin + 7 * pitch, pitch):
        cv2.line(image, (x, 0), (x, height - 1), (20, 20, 20), 3)
    for y in range(y_origin, y_origin + 6 * pitch, pitch):
        cv2.line(image, (0, y), (width - 1, y), (20, 20, 20), 3)

    # Remove both the explicit line and checker transition around an occluded
    # boundary. The lattice should reconstruct it from neighbouring lines.
    if omit_vertical is not None:
        cv2.rectangle(image, (omit_vertical - 5, 0),
                      (omit_vertical + 5, height - 1), (125, 125, 125), -1)
    if omit_horizontal is not None:
        cv2.rectangle(image, (0, omit_horizontal - 5),
                      (width - 1, omit_horizontal + 5), (125, 125, 125), -1)

    # Long near-horizontal/vertical edges imitate a piece or scene edge. They
    # are closer to the image centre than the true square borders.
    cv2.line(image, (490, 180), (500, 550), (240, 240, 240), 5)
    cv2.line(image, (250, 340), (720, 350), (240, 240, 240), 5)

    cv2.circle(image, (492, 366), 52, (35, 35, 35), -1)
    cv2.circle(image, (492, 366), 47, (170, 170, 170), 4)
    return image


def synthetic_edge_square_image():
    """An edge square where the board frame duplicates its top/right lines."""
    height, width = 720, 960
    image = np.full((height, width, 3), 130, dtype=np.uint8)
    # Target square: x=390..570, y=270..450. The 20 px frame lines above and
    # right must be merged toward the playable area, as in the real H8 frame.
    cv2.rectangle(image, (390, 270), (570, 450), (75, 75, 75), -1)
    for y in (250, 270, 450):
        cv2.line(image, (80, y), (590, y), (225, 225, 225), 4)
    for x in (390, 570, 590):
        cv2.line(image, (x, 100), (x, 650), (225, 225, 225), 4)
    cv2.circle(image, (495, 360), 45, (30, 30, 30), -1)
    cv2.circle(image, (495, 360), 42, (170, 170, 170), 3)
    return image


class CenterSquareTests(unittest.TestCase):
    def assert_square(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = detect_center_square(gray)
        self.assertIsNotNone(result)
        corners, centre, side = result
        np.testing.assert_allclose(centre, [480, 360], atol=4.0)
        self.assertAlmostEqual(side, 150.0, delta=4.0)
        self.assertEqual(corners.shape, (4, 2))
        return gray, result

    def test_ignores_long_distractor_edges(self):
        self.assert_square(synthetic_look_image())

    def test_infers_missing_centre_boundaries_from_grid(self):
        self.assert_square(synthetic_look_image(
            omit_vertical=405, omit_horizontal=285))

    def test_edge_square_uses_inner_frame_boundaries(self):
        gray = cv2.cvtColor(synthetic_edge_square_image(), cv2.COLOR_BGR2GRAY)
        result = detect_center_square(gray)
        self.assertIsNotNone(result)
        corners, centre, side = result
        np.testing.assert_allclose(centre, [480, 360], atol=4.0)
        # Hough returns the inner Canny edge of each 4 px synthetic line, so the
        # measured playable area is a few pixels smaller than centreline pitch.
        self.assertAlmostEqual(side, 180.0, delta=6.0)
        np.testing.assert_allclose(
            corners, [[390, 270], [570, 270], [570, 450], [390, 450]],
            atol=5.0)

    def test_detects_piece_base_inside_selected_square(self):
        gray, (corners, centre, side) = self.assert_square(
            synthetic_look_image())
        circle_centre, radius = detect_base_circle(
            gray, corners, centre, side)
        self.assertIsNotNone(circle_centre)
        np.testing.assert_allclose(circle_centre, [492, 366], atol=4.0)
        self.assertAlmostEqual(radius, 49.0, delta=5.0)

    def test_asymmetric_head_does_not_enlarge_base_circle(self):
        image = synthetic_look_image()
        # Replace the original round piece with a circular base plus an
        # overlapping, narrow head protruding toward the top-left.
        cv2.rectangle(image, (405, 285), (555, 435), (205, 205, 205), -1)
        cv2.line(image, (405, 285), (555, 285), (20, 20, 20), 3)
        cv2.line(image, (405, 435), (555, 435), (20, 20, 20), 3)
        cv2.line(image, (405, 285), (405, 435), (20, 20, 20), 3)
        cv2.line(image, (555, 285), (555, 435), (20, 20, 20), 3)
        cv2.circle(image, (485, 370), 42, (35, 35, 35), -1)
        cv2.ellipse(image, (467, 326), (25, 55), -20, 0, 360,
                    (35, 35, 35), -1)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        square = detect_center_square(gray)
        self.assertIsNotNone(square)
        circle_centre, radius = detect_base_circle(gray, *square)
        self.assertIsNotNone(circle_centre)
        np.testing.assert_allclose(circle_centre, [485, 370], atol=7.0)
        self.assertAlmostEqual(radius, 42.0, delta=6.0)


if __name__ == "__main__":
    unittest.main()
