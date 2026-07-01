#!/usr/bin/env python3
"""ROS node: speaks LLM chess commentary.

Thin wrapper around the pure ``ocr_runtime.llm_commentary`` pipeline;
supplies the Stockfish engine, the ROS plumbing, and the config-selected adapters.

  /chess_board_after_robot_move (FEN) -> cache_respond_callback: pre-cache
        {move: analysis} for the human's likely replies.
  /chess_chat ("fen:...,move:...")    -> local_analyze_callback: speak the
        cached line for the move played, else stream a single-move fallback.

Config lives in config/llm_commentary/llm_commentary.yaml; API keys come from the environment.
"""
import os

import rospy
from std_msgs.msg import String

from ocr_runtime.logger import setup_logger
from engine.wrapper import ChessEngineWrapper
from ocr_runtime.paths import resource_path, user_data_path
from ocr_runtime.llm_commentary import (
    decode_fen_to_piece,
    predict_human_moves,
    build_move_analysis_cache,
    extract_move,
    extract_fen,
    format_played_move,
    speak_stream,
    load_settings,
    read_prompt,
)
from ocr_runtime.llm_clients import LLM_CLIENTS
from ocr_runtime.tts_sinks import TTS_SINKS


class CommentaryManager:
    def __init__(self, config0: dict, config1: dict, llm, sink, pipeline: dict) -> None:
        self.config0 = config0  # multi-pv (pre-cache) system message
        self.config1 = config1  # single-move (fallback) system message
        self.llm = llm
        self.sink = sink
        self.logger = setup_logger(
            "llm_commentary", user_data_path("logs", "commentary.txt"))

        self.analyze_engine = ChessEngineWrapper(mode="stockfish16", depth=11)
        self.cache_database = dict()
        self.num_multipv = pipeline.get("num_multipv", 50)
        self.thread_timeout = pipeline.get("thread_timeout", 5)
        self.thread_multiplier = pipeline.get("thread_multiplier", 2)
        rospy.set_param("is_speaking", False)

    def speak_cached_analysis(self, analysis: str):
        rospy.set_param("is_speaking", True)
        try:
            self.sink.speak(analysis)
        finally:
            rospy.set_param("is_speaking", False)

    def speak_stream_analysis(self, config, user_input):
        rospy.set_param("is_speaking", True)
        try:
            spoken = speak_stream(self.llm, config, user_input, self.sink)
            self.logger.info(f"LLM output: {spoken}")
        finally:
            rospy.set_param("is_speaking", False)


def cache_respond_callback(msg, manager: CommentaryManager):
    """Pre-generate commentary for the human's likely replies to the new board."""
    board_fen = msg.data
    pieces = decode_fen_to_piece(board_fen)
    pvs = predict_human_moves(manager.analyze_engine, board_fen, manager.num_multipv)
    manager.cache_database = build_move_analysis_cache(
        manager.config0, pieces, pvs, manager.llm.complete,
        max_workers=os.cpu_count() * manager.thread_multiplier,
        timeout=manager.thread_timeout,
    )
    rospy.loginfo(
        f"cached analysis for {len(manager.cache_database)} candidate moves")


def local_analyze_callback(msg, manager: CommentaryManager):
    """Speak the cached line for the played move, or stream a fallback analysis.

    Args:
        msg: zipped board+move, e.g. "fen:2r3k1/...,move:c1f4".
    """
    move = extract_move(msg.data)
    if move in manager.cache_database:
        manager.speak_cached_analysis(manager.cache_database[move])
    else:
        rospy.loginfo(f"no cached analysis for move {move}; streaming fallback")
        pieces = decode_fen_to_piece(extract_fen(msg.data))
        request = format_played_move(pieces, move)
        manager.speak_stream_analysis(manager.config1, request)


def build_manager():
    """Assemble the manager from the single config/llm_commentary/llm_commentary.yaml file."""
    settings = load_settings(
        resource_path("config", "llm_commentary", "llm_commentary.yaml"))
    llm_cfg, tts_cfg = settings["llm"], settings["tts"]

    provider = llm_cfg["provider"]
    if provider not in LLM_CLIENTS:
        raise ValueError(f"unknown llm provider '{provider}'; "
                         f"choose from {sorted(LLM_CLIENTS)}")
    # Optional provider-specific block (e.g. llm.gemini.thinking_budget).
    provider_opts = llm_cfg.get(provider) or {}
    llm = LLM_CLIENTS[provider](model=llm_cfg["models"][provider], **provider_opts)

    sink_name = tts_cfg["sink"]
    if sink_name not in TTS_SINKS:
        raise ValueError(f"unknown tts sink '{sink_name}'; "
                         f"choose from {sorted(TTS_SINKS)}")
    sink = TTS_SINKS[sink_name](**tts_cfg.get(sink_name, {}))

    base = {"temperature": llm_cfg.get("temperature"),
            "max_tokens": llm_cfg.get("max_tokens")}
    prompts_dir = resource_path("config", "llm_commentary", "prompts")
    config0 = {**base, "system_message": read_prompt(
        prompts_dir / settings["prompts"]["multipv"])}
    config1 = {**base, "system_message": read_prompt(
        prompts_dir / settings["prompts"]["single"])}

    return CommentaryManager(config0, config1, llm, sink, settings["pipeline"])


if __name__ == "__main__":
    try:
        rospy.init_node("llm_commentary_manager", anonymous=True, log_level=rospy.WARN)
        manager = build_manager()
        rospy.loginfo("node initialized")
        rospy.Subscriber("/chess_chat", String, local_analyze_callback, manager)
        rospy.Subscriber(
            "/chess_board_after_robot_move", String, cache_respond_callback, manager)
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
