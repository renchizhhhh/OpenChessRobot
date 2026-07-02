#!/usr/bin/env python3
"""Standalone test for the LLM chess-commentary core.

Exercises the full pre-cache -> lookup pipeline on fake games with a fake engine
and a fake LLM, so it needs no ROS, OpenAI key, Stockfish, or audio.
"""

import json
import re
import sys
import unittest
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ocr_runtime.llm_commentary import (
    decode_fen_to_piece,
    predict_human_moves,
    format_move_query,
    parse_json_block,
    build_move_analysis_cache,
    format_recognized_move,
    format_played_move,
    extract_move,
    extract_fen,
    speak_stream,
    load_settings,
    read_prompt,
    LLMClient,
    TtsSink,
)
from ocr_runtime.llm_clients import LLM_CLIENTS, OpenAIClient, GeminiClient, ClaudeClient
from ocr_runtime.tts_sinks import TTS_SINKS, ElevenLabsSink


class FakeEngine:
    """Stands in for ChessEngineWrapper: top-N legal moves as multipv_info."""

    def __init__(self):
        self.multipv_info = []
        self.calls = []

    def multipv(self, fen, num):
        self.calls.append((fen, num))
        board = chess.Board(fen=fen)
        moves = [m.uci() for m in board.legal_moves][:num]
        # Mimic the real per-move info dict, including the volatile "score".
        self.multipv_info = [
            {"pv": [uci], "wdl": [500, 400, 100], "score": -i}
            for i, uci in enumerate(moves)
        ]
        return moves


def fake_llm(config, user_input, fence=False, bad=False):
    """Echo the queried move as the JSON key, like the multi_pv system message.

    The move to analyze is the single key of the ``{move: pv}`` dict that
    ``format_move_query`` embeds in the prompt.
    """
    move = re.search(r"The moves: \{'([^']+)'", user_input).group(1)
    if bad:
        return "not json at all"
    body = json.dumps({move: f"Analysis of {move}."})
    return f"```json\n{body}\n```" if fence else body


# Fake games as SAN move lists (replayed to drive the pipeline).
FAKE_GAMES = {
    "scholars": ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"],
    "italian": ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d4", "exd4"],
}


class DecodeFenTests(unittest.TestCase):
    def test_starting_position(self):
        text = decode_fen_to_piece(chess.STARTING_FEN)
        self.assertIn("e1: white king", text)
        self.assertIn("d8: black queen", text)
        # The 32 middle squares are empty at the start.
        empty = text.split("Empty squares:")[1]
        self.assertIn("e4", empty)
        self.assertNotIn("e1", empty)


class PredictHumanMovesTests(unittest.TestCase):
    def test_keys_are_moves_and_score_dropped(self):
        engine = FakeEngine()
        board = chess.Board()
        pvs = predict_human_moves(engine, board.fen(), 99)
        legal = {m.uci() for m in board.legal_moves}
        self.assertEqual(set(pvs), legal)
        for info in pvs.values():
            self.assertNotIn("score", info)   # volatile field stripped
            self.assertIn("wdl", info)         # everything else preserved

    def test_respects_count_limit(self):
        engine = FakeEngine()
        pvs = predict_human_moves(engine, chess.Board().fen(), 5)
        self.assertEqual(len(pvs), 5)
        self.assertEqual(engine.calls[-1], (chess.Board().fen(), 5))


class ParseJsonBlockTests(unittest.TestCase):
    def test_plain_fenced_and_whitespace(self):
        self.assertEqual(parse_json_block('{"e2e4": "ok"}'), {"e2e4": "ok"})
        self.assertEqual(
            parse_json_block('```json\n{"e2e4": "ok"}\n```'), {"e2e4": "ok"})
        self.assertEqual(
            parse_json_block('```\n{"e2e4": "ok"}\n```'), {"e2e4": "ok"})
        self.assertEqual(parse_json_block('   {"a": 1}   '), {"a": 1})

    def test_invalid_raises(self):
        with self.assertRaises(json.JSONDecodeError):
            parse_json_block("not json at all")


class BuildCacheTests(unittest.TestCase):
    def test_full_cache_covers_all_candidates(self):
        engine = FakeEngine()
        fen = chess.Board().fen()
        pieces = decode_fen_to_piece(fen)
        pvs = predict_human_moves(engine, fen, 99)
        cache = build_move_analysis_cache(None, pieces, pvs, fake_llm)
        self.assertEqual(set(cache), set(pvs))
        a_move = next(iter(pvs))
        self.assertEqual(cache[a_move], f"Analysis of {a_move}.")

    def test_fenced_replies_are_parsed(self):
        engine = FakeEngine()
        fen = chess.Board().fen()
        pvs = predict_human_moves(engine, fen, 3)
        cache = build_move_analysis_cache(
            None, decode_fen_to_piece(fen), pvs,
            lambda c, u: fake_llm(c, u, fence=True))
        self.assertEqual(set(cache), set(pvs))

    def test_one_bad_reply_does_not_drop_the_rest(self):
        engine = FakeEngine()
        fen = chess.Board().fen()
        pvs = predict_human_moves(engine, fen, 4)
        victim = next(iter(pvs))

        def flaky(config, user_input):
            move = re.search(r"The moves: \{'([^']+)'", user_input).group(1)
            return fake_llm(config, user_input, bad=(move == victim))

        cache = build_move_analysis_cache(None, decode_fen_to_piece(fen), pvs, flaky)
        self.assertNotIn(victim, cache)
        self.assertEqual(set(cache), set(pvs) - {victim})

    def test_empty_pvs(self):
        self.assertEqual(build_move_analysis_cache(None, "", {}, fake_llm), {})


class RecognizedMoveContractTests(unittest.TestCase):
    def test_roundtrip(self):
        board = chess.Board()
        payload = format_recognized_move(board, "e7e5")
        self.assertEqual(extract_move(payload), "e7e5")
        # The FEN (with its spaces) round-trips too, up to the ,move: separator.
        self.assertEqual(extract_fen(payload), board.fen())

    def test_extract_move_none_when_absent(self):
        self.assertIsNone(extract_move("fen:somefen,nothinghere"))

    def test_extract_fen_none_when_absent(self):
        self.assertIsNone(extract_fen("no fen here,move:e2e4"))

    def test_format_played_move_mentions_board_and_move(self):
        pieces = decode_fen_to_piece(chess.STARTING_FEN)
        request = format_played_move(pieces, "e2e4")
        self.assertIn("The move just played: e2e4", request)
        self.assertIn("e1: white king", request)


class FakeLLM(LLMClient):
    """Yields a fixed reply as space-split tokens, with a comma/period after each."""

    def __init__(self, reply):
        self.reply = reply

    def complete(self, config, user_input):
        return self.reply

    def stream(self, config, user_input):
        words = self.reply.split(" ")
        for i, word in enumerate(words):
            yield word
            yield "." if i == len(words) - 1 else ","


class FakeSink(TtsSink):
    def __init__(self):
        self.spoken = []

    def speak(self, text):
        self.spoken.append(text)


class SpeakStreamTests(unittest.TestCase):
    def test_clause_splitting_and_full_text(self):
        llm = FakeLLM("good solid developing move")
        sink = FakeSink()
        spoken = speak_stream(llm, {}, "anything", sink)
        # One clause flushed per boundary token.
        self.assertEqual(sink.spoken, ["good,", "solid,", "developing,", "move."])
        self.assertEqual(spoken, "good,solid,developing,move.")

    def test_trailing_clause_without_boundary_is_flushed(self):
        class TailLLM(LLMClient):
            def complete(self, config, user_input):
                return "tail"

            def stream(self, config, user_input):
                yield "no boundary here"

        sink = FakeSink()
        speak_stream(TailLLM(), {}, "x", sink)
        self.assertEqual(sink.spoken, ["no boundary here"])


class ConfigLoaderTests(unittest.TestCase):
    def test_read_prompt_flattens_lines(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            sysmsg = Path(tmp) / "sys.txt"
            sysmsg.write_text("line one\n\nline two\n")
            # Lines join with a space and blank lines collapse away.
            self.assertEqual(read_prompt(str(sysmsg)), "line one line two")

    def test_load_settings_parses_yaml(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "llm_commentary.yaml"
            cfg.write_text(
                "llm:\n  provider: openai\n  temperature: 0.7\n"
                "pipeline:\n  num_multipv: 5\n")
            settings = load_settings(str(cfg))
        self.assertEqual(settings["llm"]["provider"], "openai")
        self.assertEqual(settings["pipeline"]["num_multipv"], 5)

    def test_shipped_config_is_valid(self):
        """The real config/llm_commentary/llm_commentary.yaml loads consistently."""
        settings = load_settings(
            str(ROOT / "config" / "llm_commentary" / "llm_commentary.yaml"))
        self.assertIn(settings["llm"]["provider"], LLM_CLIENTS)
        self.assertIn(settings["tts"]["sink"], TTS_SINKS)
        self.assertIn(settings["llm"]["provider"], settings["llm"]["models"])
        prompts = ROOT / "config" / "llm_commentary" / "prompts"
        self.assertTrue((prompts / settings["prompts"]["multipv"]).exists())
        self.assertTrue((prompts / settings["prompts"]["single"]).exists())


class AdapterRegistryTests(unittest.TestCase):
    """Adapters register and construct without importing their (optional) SDK."""

    def test_registries(self):
        self.assertEqual(set(LLM_CLIENTS), {"openai", "gemini", "claude"})
        self.assertEqual(set(TTS_SINKS), {"elevenlabs"})

    def test_construct_without_sdk(self):
        # Construction must not import openai / google / anthropic / elevenlabs (lazy).
        self.assertEqual(OpenAIClient(model="m").model, "m")
        self.assertEqual(GeminiClient(model="g").model, "g")
        self.assertEqual(ClaudeClient(model="c").model, "c")
        self.assertEqual(ElevenLabsSink(voice_id="v").voice_id, "v")

    def test_implement_interfaces(self):
        self.assertIsInstance(OpenAIClient(), LLMClient)
        self.assertIsInstance(GeminiClient(), LLMClient)
        self.assertIsInstance(ClaudeClient(), LLMClient)
        self.assertIsInstance(ElevenLabsSink(), TtsSink)


class FakeGameEndToEndTests(unittest.TestCase):
    """The whole point: every move a human actually plays was pre-cached."""

    def _run_game(self, sans):
        engine = FakeEngine()
        board = chess.Board()
        checked = 0
        for san in sans:
            move = board.parse_san(san)
            if board.turn == chess.BLACK:
                # Black is the human: before the move, the manager would have
                # cached analysis for all of black's replies to this position.
                fen = board.fen()
                pvs = predict_human_moves(engine, fen, 99)
                cache = build_move_analysis_cache(
                    None, decode_fen_to_piece(fen), pvs, fake_llm)
                # The commander publishes the played move; the manager looks it up.
                played = extract_move(format_recognized_move(board, move.uci()))
                self.assertIn(played, cache)
                self.assertEqual(cache[played], f"Analysis of {played}.")
                checked += 1
            board.push(move)
        return checked

    def test_fake_games(self):
        for name, sans in FAKE_GAMES.items():
            with self.subTest(game=name):
                self.assertGreater(self._run_game(sans), 0)


if __name__ == "__main__":
    unittest.main()
