from dataclasses import dataclass
from enum import Enum


class InitialBoardDecision(str, Enum):
    INITIAL_BOARD_OK = "initial_board_ok"
    REVIEW_CUSTOM_BOARD = "recognized_custom_board"
    RETRY_RECOGNITION = "invalid_or_ambiguous"


@dataclass(frozen=True)
class InitialBoardCheck:
    decision: InitialBoardDecision
    status_name: str
    fen: str


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
VALID_STATUS_NAMES = {"VALID", "OPPOSITE_CHECK"}


def board_status_name(board) -> str:
    status = board.status()
    name = getattr(status, "name", None)
    if name:
        return name
    if status == 0:
        return "VALID"
    return str(status)


def classify_initial_board(board) -> InitialBoardCheck:
    status_name = board_status_name(board)
    fen = board.fen()

    if fen == STARTING_FEN:
        decision = InitialBoardDecision.INITIAL_BOARD_OK
    elif status_name in VALID_STATUS_NAMES:
        decision = InitialBoardDecision.REVIEW_CUSTOM_BOARD
    else:
        decision = InitialBoardDecision.RETRY_RECOGNITION

    return InitialBoardCheck(
        decision=decision,
        status_name=status_name,
        fen=fen,
    )
