from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


CAPTURE_BOX_SQUARE = "H4"
CAPTURE_BOX_Y_OFFSET = -0.1
EN_PASSANT_CAPTURE_BOX_Y_OFFSET = -0.2

PROMOTION_PIECES = {
    "q": "queen",
    "r": "rook",
    "b": "bishop",
    "n": "knight",
}


@dataclass(frozen=True)
class EncodedMove:
    raw: str
    uci: str
    start: str
    end: str
    is_hop: bool
    is_capture: bool
    is_castling: bool
    is_en_passant: bool
    is_promotion: bool
    promotion_piece: Optional[str] = None


@dataclass(frozen=True)
class CaptureBoxDrop:
    square: str = CAPTURE_BOX_SQUARE
    x_offset: float = 0.0
    y_offset: float = CAPTURE_BOX_Y_OFFSET


def parse_encoded_move(raw_move: str) -> EncodedMove:
    """Parse the string command published on /chess_move.

    The wire format is UCI plus five flags:
    <uci><hop><capture><castling><en-passant><promotion>.
    Promotion UCI contains the promoted piece letter, e.g. e7e8q00001.
    """
    raw_move = raw_move.strip()
    if len(raw_move) < 9:
        raise ValueError(f"encoded move is too short: {raw_move!r}")

    uci = raw_move[:-5].lower()
    flags = raw_move[-5:]
    if len(uci) not in (4, 5):
        raise ValueError(f"encoded move must contain 4 or 5 UCI chars: {raw_move!r}")
    if any(flag not in "01" for flag in flags):
        raise ValueError(f"encoded move flags must be 0 or 1: {raw_move!r}")

    is_promotion = flags[4] == "1"
    promotion_piece = uci[4] if len(uci) == 5 else None
    if is_promotion and promotion_piece not in PROMOTION_PIECES:
        raise ValueError(
            "promotion moves must include one of q, r, b, n as the fifth UCI char"
        )
    if promotion_piece is not None and not is_promotion:
        raise ValueError("promotion piece is present but promotion flag is not set")

    return EncodedMove(
        raw=raw_move,
        uci=uci,
        start=uci[:2].upper(),
        end=uci[2:4].upper(),
        promotion_piece=promotion_piece,
        is_hop=flags[0] == "1",
        is_capture=flags[1] == "1",
        is_castling=flags[2] == "1",
        is_en_passant=flags[3] == "1",
        is_promotion=is_promotion,
    )


def normal_capture_drop() -> CaptureBoxDrop:
    return CaptureBoxDrop(y_offset=CAPTURE_BOX_Y_OFFSET)


def en_passant_capture_drop() -> CaptureBoxDrop:
    return CaptureBoxDrop(y_offset=EN_PASSANT_CAPTURE_BOX_Y_OFFSET)


def castling_rook_move(start: str, end: str) -> tuple[str, str]:
    start = start.upper()
    end = end.upper()
    if len(start) != 2 or len(end) != 2 or start[1] != end[1]:
        raise ValueError(f"{start} to {end} is not valid castling")

    if ord(start[0]) > ord(end[0]):
        return "A" + start[1], chr(ord(end[0]) + 1) + start[1]
    return "H" + start[1], chr(ord(end[0]) - 1) + start[1]


def en_passant_capture_square(start: str, end: str) -> str:
    start = start.upper()
    end = end.upper()
    if len(start) != 2 or len(end) != 2:
        raise ValueError(f"{start} to {end} is not a valid en-passant move")
    return end[0] + start[1]


def promotion_color(end_square: str) -> str:
    rank = end_square.upper()[1]
    if rank == "8":
        return "white"
    if rank == "1":
        return "black"
    return "unknown-color"


def promotion_piece_name(piece: str) -> str:
    try:
        return PROMOTION_PIECES[piece.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported promotion piece: {piece!r}") from exc


def promotion_prompt(decoded_move: EncodedMove) -> str:
    if not decoded_move.is_promotion or decoded_move.promotion_piece is None:
        raise ValueError("promotion prompt requires a promotion move")
    color = promotion_color(decoded_move.end).upper()
    piece = promotion_piece_name(decoded_move.promotion_piece).upper()
    return (
        f"PROMOTION: Replace pawn on {decoded_move.end} with "
        f"{color} {piece}, then confirm."
    )
