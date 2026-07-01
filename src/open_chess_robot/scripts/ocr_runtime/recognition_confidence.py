"""Aggregate per-square recognition model confidence into a board-level summary.

This module intentionally depends only on numpy (and the standard library) so it
can be unit tested without loading the torch recognition models or python-chess.
Square indices follow the python-chess convention ``sq = rank * 8 + file`` with
file 0 = A and rank 0 = 1, so index 0 is A1 and index 63 is H8.
"""

import numpy as np

_FILES = "ABCDEFGH"


def square_name(square):
    """Return the uppercase board-square name (e.g. ``"E2"``) for an index."""
    return f"{_FILES[square % 8]}{square // 8 + 1}"


def summarize_confidence(occupancy_confidence, piece_confidence, threshold,
                         squares=None):
    """Summarize per-square model confidence.

    Args:
        occupancy_confidence: per-square softmax probability of the predicted
            occupancy class (empty/occupied), aligned to ``squares``.
        piece_confidence: per-square softmax probability of the predicted piece
            class; values <= 0 mark squares classified as empty.
        threshold: squares whose confidence is below this value are reported as
            ambiguous.
        squares: square indices aligned to the confidence arrays. Defaults to
            ``range(len(occupancy_confidence))``.

    Returns:
        (overall_confidence, ambiguous_square_names). ``overall_confidence`` is
        the mean per-square confidence, where an occupied square uses the weaker
        of its occupancy and piece decisions. Returns ``(-1.0, [])`` when the
        inputs are missing or mismatched.
    """
    occ = np.asarray(occupancy_confidence, dtype=float).reshape(-1)
    pie = np.asarray(piece_confidence, dtype=float).reshape(-1)
    n = occ.shape[0]
    if n == 0 or pie.shape[0] != n:
        return -1.0, []

    if squares is None:
        squares = range(n)
    squares = list(squares)

    per_square = occ.copy()
    occupied = pie > 0
    per_square[occupied] = np.minimum(occ[occupied], pie[occupied])

    ambiguous = [
        square_name(squares[i])
        for i in range(n)
        if per_square[i] < threshold
    ]
    overall = float(per_square.mean())
    return overall, ambiguous
