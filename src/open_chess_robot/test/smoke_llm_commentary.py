#!/usr/bin/env python3
"""Offline smoke test for the LLM-commentary adapters (no ROS, no robot).

Exercises the two genuinely-new external edges - the LLM provider and the
ElevenLabs TTS - against the REAL SDKs, config/llm_commentary/llm_commentary.yaml
and your API keys. Stockfish is not needed (candidate moves are hand-crafted).

Unlike test_llm_commentary.py (fully mocked, run by pytest/catkin), this hits
live services and costs API calls, so it is a manual tool, not a CI test.

Three independent stages, any subset of which runs for any subset of models:
  multipv  the pre-cache path: fan candidate replies out to the LLM in parallel
           and collect the {move: analysis} cache (uses the multipv prompt).
  single   the fallback path: stream a single-move analysis (uses the single
           prompt); reports time-to-first-token, the latency a human perceives.
  tts      speak a line through ElevenLabs (this model's output if available).

Every step is timed and collected into a summary table printed at the end.

Usage:
    python smoke_llm_commentary.py                      # config's provider, all stages
    python smoke_llm_commentary.py openai gemini        # both models, all stages
    python smoke_llm_commentary.py all                  # every registered model
    python smoke_llm_commentary.py openai --stages multipv single   # no audio
    python smoke_llm_commentary.py --stages tts         # audio only, model-independent
"""
import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from ocr_runtime.llm_commentary import (
    decode_fen_to_piece, build_move_analysis_cache, format_played_move,
    load_settings, read_prompt,
)
from ocr_runtime.llm_clients import LLM_CLIENTS
from ocr_runtime.tts_sinks import TTS_SINKS

SETTINGS = load_settings(REPO / "config" / "llm_commentary" / "llm_commentary.yaml")
PROMPTS = REPO / "config" / "llm_commentary" / "prompts"

STAGES = ("multipv", "single", "tts")
DEFAULT_LINE = "Nice move. The bishop steps forward, eyeing the center."

# A simple opening position (white just played e4); black to move.
FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
# Hand-crafted candidate human replies (what predict_human_moves would return).
PVS = {
    "c7c5": {"pv": ["c7c5"], "wdl": [400, 450, 150]},
    "e7e5": {"pv": ["e7e5"], "wdl": [430, 420, 150]},
    "g8f6": {"pv": ["g8f6"], "wdl": [380, 460, 160]},
}


class Timer:
    """Collects (label, seconds) for every step and prints a summary table."""

    def __init__(self):
        self.records = []

    def record(self, label, dt):
        self.records.append((label, dt))
        print(f"    [t] {label}: {dt:.2f}s")

    @contextmanager
    def step(self, label):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(label, time.perf_counter() - t0)

    def summary(self):
        if not self.records:
            return
        print("\n=== timing summary ===")
        width = max(len(label) for label, _ in self.records)
        for label, dt in self.records:
            print(f"  {label:<{width}}  {dt:6.2f}s")
        print(f"  {'TOTAL':<{width}}  {sum(d for _, d in self.records):6.2f}s")


def base_config(prompt_file):
    return {"temperature": SETTINGS["llm"].get("temperature"),
            "max_tokens": SETTINGS["llm"].get("max_tokens"),
            "system_message": read_prompt(PROMPTS / prompt_file)}


def make_llm(provider):
    opts = SETTINGS["llm"].get(provider) or {}
    return LLM_CLIENTS[provider](model=SETTINGS["llm"]["models"][provider], **opts)


def run_multipv(provider, client, pieces, timer):
    """Pre-cache path: parallel complete() over candidate replies."""
    print(f"  - multipv cache on {len(PVS)} candidate moves (parallel) ...")
    with timer.step(f"{provider} · multipv cache ({len(PVS)} moves)"):
        cache = build_move_analysis_cache(
            base_config(SETTINGS["prompts"]["multipv"]), pieces, PVS, client.complete)
    print(f"    cached {len(cache)}/{len(PVS)} moves:")
    for move, text in cache.items():
        print(f"      {move}: {text[:90]}")
    return cache


def run_single(provider, client, timer):
    """Fallback path: stream single-move analysis, timing first-token latency."""
    print("  - single-move stream ...")
    config = base_config(SETTINGS["prompts"]["single"])
    # Mirror the fallback path: decode the FEN and send pieces + played move.
    request = format_played_move(decode_fen_to_piece(FEN), "c7c5")
    t0 = time.perf_counter()
    chunks, first_dt = [], None
    for chunk in client.stream(config, request):
        if first_dt is None:
            first_dt = time.perf_counter() - t0
        chunks.append(chunk)
    total_dt = time.perf_counter() - t0
    timer.record(f"{provider} · single first-token", first_dt or 0.0)
    timer.record(f"{provider} · single full stream", total_dt)
    print(f"    {sum(len(c) for c in chunks)} chars, {len(chunks)} deltas: "
          f"{''.join(chunks)[:120]!r}")


def run_tts(label, text, timer):
    sink_name = SETTINGS["tts"]["sink"]
    print(f"  - {sink_name} speaking: {text[:80]!r}")
    with timer.step(f"{label} · tts speak"):
        TTS_SINKS[sink_name](**SETTINGS["tts"].get(sink_name, {})).speak(text)
    print("    playback returned (did you hear it?)")


def run_model(provider, stages, timer):
    print(f"\n[model] {provider} ({SETTINGS['llm']['models'][provider]}) "
          f"| stages: {', '.join(s for s in STAGES if s in stages)}")
    client = make_llm(provider)
    pieces = decode_fen_to_piece(FEN)

    cache = {}
    if "multipv" in stages:
        cache = run_multipv(provider, client, pieces, timer)
    if "single" in stages:
        run_single(provider, client, timer)
    if "tts" in stages:
        text = next(iter(cache.values())) if cache else DEFAULT_LINE
        run_tts(provider, text, timer)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*",
                        help="model names (e.g. openai gemini), or 'all'; "
                             "default: the provider in llm_commentary.yaml")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES),
                        help="which stages to run (default: all)")
    args = parser.parse_args(argv)

    if any(m == "all" for m in args.models):
        args.models = list(LLM_CLIENTS)
    elif not args.models:
        args.models = [SETTINGS["llm"]["provider"]]

    unknown = [m for m in args.models if m not in LLM_CLIENTS]
    if unknown:
        parser.error(f"unknown model(s) {unknown}; choose from {sorted(LLM_CLIENTS)}")
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    timer = Timer()

    # tts-only (no LLM stage) is model-independent: run it once.
    if args.stages == ["tts"]:
        run_tts("default", DEFAULT_LINE, timer)
    else:
        for provider in args.models:
            run_model(provider, args.stages, timer)

    timer.summary()


if __name__ == "__main__":
    main()
