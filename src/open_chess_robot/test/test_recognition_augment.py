#!/usr/bin/env python3
"""Tests for the recognition retry augmentations.

These guard the "non-color" contract: the piece classifier also decides piece
color, so every augmentation must preserve the per-pixel channel ordering
(a white piece must not be pushed toward black). No ROS/torch.
"""

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ocr_runtime.recognition_augment import (
    AUGMENTATIONS,
    apply_augmentation,
)


def _sample_image():
    # A small image whose channels are strictly ordered R > G > B everywhere,
    # plus some structure so contrast/unsharp have something to act on.
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[..., 0] = 180  # R
    img[..., 1] = 120  # G
    img[..., 2] = 60   # B
    img[2:6, 2:6, :] += 20
    return img


class ApplyAugmentationTests(unittest.TestCase):
    def test_none_and_empty_are_identity(self):
        img = _sample_image()
        np.testing.assert_array_equal(apply_augmentation(img, "none"), img)
        np.testing.assert_array_equal(apply_augmentation(img, ""), img)

    def test_unknown_augmentation_raises(self):
        with self.assertRaises(KeyError):
            apply_augmentation(_sample_image(), "sepia")

    def test_all_preserve_shape_and_dtype(self):
        img = _sample_image()
        for name in AUGMENTATIONS:
            out = apply_augmentation(img, name)
            self.assertEqual(out.shape, img.shape, name)
            self.assertEqual(out.dtype, np.uint8, name)

    def test_brighten_and_darken_shift_luminance(self):
        img = _sample_image()
        self.assertGreater(apply_augmentation(img, "brighten").mean(), img.mean())
        self.assertLess(apply_augmentation(img, "darken").mean(), img.mean())

    def test_channel_ordering_preserved(self):
        # The non-color guarantee: R >= G >= B must hold after every transform,
        # so white-vs-black piece luminance ordering survives.
        img = _sample_image()
        for name in AUGMENTATIONS:
            out = apply_augmentation(img, name).astype(int)
            self.assertTrue(np.all(out[..., 0] >= out[..., 1]), name)
            self.assertTrue(np.all(out[..., 1] >= out[..., 2]), name)


if __name__ == "__main__":
    unittest.main()
