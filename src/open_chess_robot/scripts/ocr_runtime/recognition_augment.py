"""Non-color photometric augmentations for the recognition retry fallback.

When the default recognition fails, the commander retries on a different image
source / augmentation (see the fallback plan in settings). These transforms
perturb edges and texture enough to give the classifier a second chance at a
borderline call, WITHOUT changing color: the piece classifier also decides piece
*color* (white vs black), so hue/saturation/grayscale changes are forbidden -
they could turn a white piece "black". Every transform here scales all channels
equally (multiplicative brightness/contrast) or sharpens, preserving the
white-vs-black luminance ordering.

Apply these to the warped board crop just before the classifier, never to the
raw frame (marker/corner detection must run on the clean image).
"""

import cv2
import numpy as np


def identity(img):
    return img


def contrast(img, alpha=1.3):
    """Multiplicative contrast about mid-gray; preserves channel ratios."""
    out = (img.astype(np.float32) - 128.0) * alpha + 128.0
    return np.clip(out, 0, 255).astype(np.uint8)


def brighten(img, alpha=1.2):
    """Multiplicative brightness; preserves white-vs-black ordering."""
    return np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def darken(img, alpha=0.8):
    return np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def unsharp(img, amount=1.0, sigma=1.0):
    """Unsharp mask: emphasize edges without touching color balance."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0.0)


AUGMENTATIONS = {
    "none": identity,
    "contrast": contrast,
    "brighten": brighten,
    "darken": darken,
    "unsharp": unsharp,
}


def apply_augmentation(img, name):
    """Apply the named augmentation; ``"none"``/empty returns the image as-is."""
    if not name or name == "none":
        return img
    try:
        transform = AUGMENTATIONS[name]
    except KeyError:
        raise KeyError(f"unknown augmentation '{name}'")
    return transform(img)
