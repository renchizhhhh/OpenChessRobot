"""Pure (ROS/OpenAI/TTS-free) core of the LLM chess-commentary feature.

The live feature pre-generates spoken commentary for a human's likely replies
*before* they move, then speaks the cached line for the move they actually play:

  1. The robot is about to move -> the post-move FEN is sent to the manager.
  2. ``predict_human_moves`` asks the engine for the human's candidate replies.
  3. ``build_move_analysis_cache`` fans those out to the LLM in parallel and
     collects a ``{move: analysis}`` dict (the LLM echoes each move as the key,
     per the multi_pv system message).
  4. The human moves; the commander publishes ``fen:...,move:...``; the manager
     pulls the move out with ``extract_move`` and looks it up in the cache.

Everything here is dependency-injected (the engine and the LLM request function
are passed in), so it runs without ROS, OpenAI, Stockfish, or audio - which is
what makes the standalone fake-game test possible. The
``llm_commentary_manager.py`` node is a thin wrapper that supplies the real
engine, LLM call, and TTS.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Iterator

import chess

logger = logging.getLogger(__name__)

_PIECE_NAMES = {
    "P": "white pawn", "N": "white knight", "B": "white bishop",
    "R": "white rook", "Q": "white queen", "K": "white king",
    "p": "black pawn", "n": "black knight", "b": "black bishop",
    "r": "black rook", "q": "black queen", "k": "black king",
}

_MOVE_RE = re.compile(r"move:(\w+)")
_FEN_RE = re.compile(r"fen:(.+),move:")


def decode_fen_to_piece(fen):
    """Render a FEN as a human-readable square->piece listing for the LLM prompt.

    e.g. "e1: white king; e8: black king; Empty squares: a1, a2, ...".
    """
    board = chess.Board(fen=fen)
    occupied = []
    empty = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        name = chess.square_name(square)
        if piece:
            occupied.append(f"{name}: {_PIECE_NAMES[piece.symbol()]}")
        else:
            empty.append(name)
    return "; ".join(occupied) + f"; Empty squares: {', '.join(empty)}"


def predict_human_moves(engine, fen, number):
    """Return ``{move: pv_info}`` for the engine's top ``number`` replies to ``fen``.

    ``move`` is the engine's principal-variation first move (the key the cache and
    the human's recognized move are matched on). The volatile ``score`` field is
    dropped (it is not part of the prompt) without mutating the engine's state.
    """
    moves = {}
    engine.multipv(fen, number)
    for pv in engine.multipv_info:
        move = pv["pv"][0]
        moves[move] = {key: value for key, value in pv.items() if key != "score"}
    return moves


def format_move_query(pieces, pv_item):
    """Build the user prompt for one candidate move's principal variation."""
    return f"The current chess board: {pieces} \nThe moves: {pv_item}"


def format_played_move(pieces, move):
    """Build the fallback user prompt for a single move that was just played.

    The fallback path has no engine analysis (no pv/wdl), only the board and the
    move, so this is the single-move counterpart to :func:`format_move_query`.
    """
    return f"The current chess board: {pieces} \nThe move just played: {move}"


def parse_json_block(text):
    """Parse an LLM reply into a dict, tolerating ```json ... ``` code fences.

    Raises ``json.JSONDecodeError`` if no JSON object can be recovered, so a bad
    reply fails its own task instead of silently poisoning the cache.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Drop the opening fence (``` or ```json) and the closing fence.
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    return json.loads(cleaned)


def build_move_analysis_cache(config, pieces, pvs, request_fn,
                              max_workers=None, timeout=None):
    """Fan candidate moves out to the LLM and merge the replies into one cache.

    Args:
        config: LLM config passed straight through to ``request_fn``.
        pieces: board description from :func:`decode_fen_to_piece`.
        pvs: ``{move: pv_info}`` from :func:`predict_human_moves`.
        request_fn: ``request_fn(config, user_input) -> str`` returning the raw
            LLM reply (a JSON block keyed by move). Injected so the node wires in
            the real OpenAI call and the test wires in a fake.
        max_workers: thread-pool size; defaults to one thread per candidate.
        timeout: overall seconds to wait for replies; partial results on timeout.

    Returns:
        ``{move: analysis_text}``. Candidates whose call fails or whose reply is
        not valid JSON are skipped (logged), so one bad move cannot drop the rest.
    """
    cache = {}
    if not pvs:
        return cache

    def task(move, pv_info):
        return parse_json_block(request_fn(config, format_move_query(pieces, {move: pv_info})))

    with ThreadPoolExecutor(max_workers=max_workers or len(pvs)) as executor:
        futures = {executor.submit(task, move, info): move for move, info in pvs.items()}
        try:
            for future in as_completed(futures, timeout=timeout):
                move = futures[future]
                try:
                    cache.update(future.result())
                except Exception as exc:  # bad LLM reply / parse error for one move
                    logger.warning("analysis for move %s failed: %s", move, exc)
        except TimeoutError:
            logger.warning("analysis timed out after %ss; cached %d/%d moves",
                           timeout, len(cache), len(pvs))
    return cache


def format_recognized_move(board, move):
    """Encode the played position + move for the /chess_chat lookup contract.

    Mirrors :func:`extract_move`; the manager parses the move back out of this.
    """
    return f"fen:{board.fen()},move:{move}"


def extract_move(payload):
    """Pull the ``move:<uci>`` token back out of a /chess_chat payload (or None)."""
    match = _MOVE_RE.search(payload)
    return match.group(1) if match else None


def extract_fen(payload):
    """Pull the FEN back out of a /chess_chat payload (or None).

    Mirrors :func:`extract_move`; the FEN carries no comma, so it runs up to the
    ``,move:`` separator that :func:`format_recognized_move` writes.
    """
    match = _FEN_RE.search(payload)
    return match.group(1) if match else None


def load_settings(yaml_path):
    """Load the single config/llm_commentary/llm_commentary.yaml settings file as a dict."""
    import yaml
    with open(yaml_path, "r") as handle:
        return yaml.safe_load(handle)


def read_prompt(prompt_path):
    """Read a system-prompt file, flattened to a single line (no newlines).

    Lines are joined with a space so multi-line prompts don't glue words
    together across line breaks; blank lines collapse away.
    """
    with open(prompt_path, "r", encoding="utf-8") as handle:
        lines = (line.strip() for line in handle.read().splitlines())
        return " ".join(line for line in lines if line)


class LLMClient(ABC):
    """Provider-agnostic LLM interface used by the commentary pipeline.

    ``config`` is a dict carrying ``system_message``, ``temperature``, and
    optional ``max_tokens`` (assembled by the node). The concrete
    model id belongs to the client. Implementations lazy-import their SDK so this
    module stays dependency-light.
    """

    @abstractmethod
    def complete(self, config, user_input) -> str:
        """Return the full reply text for one request (blocking, with retries)."""

    @abstractmethod
    def stream(self, config, user_input) -> Iterator[str]:
        """Yield reply text deltas as they arrive."""


class TtsSink(ABC):
    """Text-to-speech output. ``speak`` blocks until the clause has been spoken."""

    @abstractmethod
    def speak(self, text: str) -> None:
        ...


def speak_stream(llm, config, user_input, sink, boundaries=(",", ".")):
    """Stream an LLM reply and speak it clause by clause as it arrives.

    Buffers deltas until a boundary token (a lone "," or ".") then flushes the
    clause to ``sink``. The single replacement for the previously duplicated
    OpenAI/HoloLens streaming loops. Returns the full spoken text (for logging).
    """
    buffer = ""
    spoken = ""
    for delta in llm.stream(config, user_input):
        if not delta:
            continue
        buffer += delta
        if delta.strip() in boundaries:
            sink.speak(buffer)
            spoken += buffer
            buffer = ""
    if buffer.strip():
        sink.speak(buffer)
        spoken += buffer
    return spoken
