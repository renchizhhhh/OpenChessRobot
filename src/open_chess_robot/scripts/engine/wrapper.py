"""
    This module initially implements the Stockfish class by Ilya Zhelyabuzhsky and 
    then extened to fit other engine like Lc0. 

    :copyright: (c) blackholebug (r.z.).
    :license: MIT, see LICENSE for more details.
"""

import subprocess
from typing import Any, List, Optional
import copy
from os import path
from dataclasses import dataclass
import re
import chess
import typing
from pathlib import Path
import math

def cp_to_winrate(cp):
    return 100 / (1 + math.exp(-0.00368208 * cp))

class ChessEngineWrapper(object):
    def __init__(self, mode:str, path=str(Path.resolve(Path(__file__)).parent ), depth=20, elo=2200) -> None:
        self.engine = self._init_engine(mode, path, depth, elo)
        self.move = ''
        self.ponder = ''
        self.info_dict: dict = dict()
        self.multipv_info: typing.List[dict] = list()

    def _init_engine(self, mode: str, path: str, depth: int, elo: int):
        if mode == "stockfish15":
            path += "/stockfish15"
            return UCIEngine(
                path=path,
                depth=depth,
                mode=mode,
            )
        elif mode == "stockfish-uci":
            path += "/stockfish"
            return UCIEngine(
                path=path,
                depth=depth,
                mode=mode,
                parameters={"UCI_LimitStrength": "true", "UCI_Elo": elo,},
            )
        elif mode == "stockfish16":
            path += "/stockfish16"
            return UCIEngine(
                path=path,
                depth=depth,
                mode=mode,
            )
        elif mode == "maia":
            path += "/maia/lc0"
            return UCIEngine(
                path=path,
                mode=mode,
                depth=1,
            )

    def shutdown(self):
        # deprecated
        # if type(self.engine) == chess.engine.SimpleEngine:
        #     self.engine.quit()
        if type(self.engine) == UCIEngine:
            self.engine.__del__()
       
    def next_move(self, board: chess.Board):
        # deprecated
        # if type(self.engine) == chess.engine.SimpleEngine:
        #     result = self.engine.play(board, chess.engine.Limit(time=0.1, nodes=1))
        #     return result.move.uci()
        if type(self.engine) == UCIEngine:
            self.engine.set_fen_position(board.fen())
            self.move, self.ponder = self.engine.get_best_move(wtime=None)
            self.info_dict = self.parse_into_dict(self.engine.info)
            return self.move
        
    def evaluate_move(self, board: chess.Board, move: str, depth: int=None):
        """evaluate a given move on the given board. This function didn't update the 
        information of the Engine class or the board class. The evaluation is accurate 
        only with full level stockfish.

        Args:
            board (chess.Board): a chess board object
            move (str): a given uci move

        Returns:
            Dict: the dict containing the last line of stiockfish info under 3 keywords: score, mate and pv.
        """
        assert self.engine.mode in ["stockfish15", "stockfish16"], "Evaluation only works with default stockfish15"
        self.engine.set_fen_position(board.fen())
        info = self.engine.evaluate_a_move(move, depth)
        info = self.parse_into_dict(info)
        return info

    def get_score_from_info(self, info: dict) -> typing.Tuple[int, bool]:
        """extract score from the info dict. If a mate is found, it will return the mate counts.

        Args:
            info (dict): info dict

        Returns:
            typing.Tuple[int, bool]: cp score or mate count, found mate or not
        """
        score = info.get('score')
        if score is not None:
            return score, False
        else:
            score = info.get('mate')
            return score, True

    def multipv(self, board: chess.Board, num: int) -> typing.List[str]:
        """produce multiple candidate moves

        Args:
            board (chess.Board): current chess board
            num (int): number of moves to find

        Raises:
            NotImplementedError: right now only works for the customized UCIEngine

        Returns:
            list(str): multiple best moves in string
        """
        if type(self.engine) == UCIEngine:
            self.engine.set_fen_position(board.fen())
            moves = self.engine.get_top_moves(num)
            self.multipv_info = self.parse_multipv_into_dict(self.engine.multipv_info)
            return moves
        else:
            raise NotImplementedError("multipv is not implemented for the current chess engine.")

    def parse_into_dict(self, info_string):
        info_dict = {}
        score_match = re.search(r'score cp (\S+)', info_string)
        if score_match:
            info_dict['score'] = int(score_match.group(1))
        mate_match = re.search(r'score mate (\S+)', info_string)
        if mate_match:
            info_dict['mate'] = int(mate_match.group(1))
        pv_match = re.search(r' pv (.+)$', info_string)
        if pv_match:
            move_sequence = pv_match.group(1)
            info_dict['pv'] = move_sequence.split()
        return info_dict

    def parse_multipv_into_dict(self, info_string_list):
        multipv_info = []
        for line in info_string_list:
            multipv_info.append(self.parse_into_dict(line))
        return multipv_info
    
    def ucipv_to_algebraic(self, board: chess.Board, moves: typing.List[str]):
        new_moves = []
        for move in moves:
            new_moves.append(chess.Move.from_uci(move))
        pv = board.variation_san(new_moves)
        return pv

    def classify_opponent_move(self, previous_score, current_score):
        threshold = cp_to_winrate(current_score) - cp_to_winrate(previous_score)
        # refer to https://www.reddit.com/r/chess/comments/3ctsyk/definitions_of_inaccuracies_mistakes_blunders/
        # however, change a little bit based on several games on lichess
        if 20 <= threshold:
            return "blunder"
        elif 10 <= threshold < 20:
            return "mistake"
        elif 5 <= threshold < 10:
            return "inaccuracy"
        elif -10 <= threshold < -5:
            return "fair"
        elif -20 <= threshold < -10:
            return "good"
        elif threshold < -20:
            return "killer"
        elif 0 < threshold < 5:
            return "negative"
        elif -5 <= threshold < 0:
            return "possitive"
        else:
            return ""

    def get_info(self, key):
        return self.info_dict.get(key)

    def get_multipv(self):
        return self.multipv_info


class UCIEngine:
    """Integrates the UCI chess engine with Python."""

    _del_counter = 0
    # Used in test_models: will count how many times the del function is called.

    def __init__(
        self, path: str = "engine", mode = 'stockfish15', depth: int = 15, parameters: dict = None
    ) -> None:
        self.mode = mode
        if 'stockfish' in mode:
            self._DEFAULT_PARAMS = {
                "Debug Log File": "",
                "Contempt": 0,
                "Min Split Depth": 0,
                "Threads": 10,
                "Ponder": "false",
                "Hash": 6000,
                "MultiPV": 1,
                "Skill Level": 20,
                "Move Overhead": 10,
                "Slow Mover": 100,
                "UCI_Chess960": "false",
                "UCI_LimitStrength": "false",
                "UCI_Elo": 1350,
            }
        elif mode == 'maia':
            self._DEFAULT_PARAMS = {
                "MultiPV": 1,
                "Threads": 10,
                "UCI_Chess960": "false",
            }
        else:
            self._DEFAULT_PARAMS = {
                "Threads": 1,
            }
        self._path = path
        self._uciengine = subprocess.Popen(
            self._path,
            universal_newlines=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        self._has_quit_command_been_sent = False

        # self._stockfish_major_version: int = int(
        #     self._read_line().split(" ")[1].split(".")[0].replace("-", "")
        # )

        self._put("uci")

        self.depth = str(depth)
        self.info: str = ""
        self.multipv_info: list = []

        self._parameters: dict = {}
        self.update_engine_parameters(self._DEFAULT_PARAMS)
        self.update_engine_parameters(parameters)

        self._prepare_for_new_position(True)

    def get_parameters(self) -> dict:
        """Returns current board position.

        Returns:
            Dictionary of current UCIEngine engine's parameters.
        """
        return self._parameters

    def update_engine_parameters(self, new_param_valuesP: Optional[dict]) -> None:
        """Updates the stockfish parameters.

        Args:
            new_param_values:
                Contains (key, value) pairs which will be used to update
                the _parameters dictionary.

        Returns:
            None
        """
        if not new_param_valuesP:
            return

        new_param_values = copy.deepcopy(new_param_valuesP)

        if len(self._parameters) > 0:
            for key in new_param_values:
                if key not in self._parameters:
                    raise ValueError(f"'{key}' is not a key that exists.")

        if ("Skill Level" in new_param_values) != (
            "UCI_Elo" in new_param_values
        ) and "UCI_LimitStrength" not in new_param_values:
            # This means the user wants to update the Skill Level or UCI_Elo (only one,
            # not both), and that they didn't specify a new value for UCI_LimitStrength.
            # So, update UCI_LimitStrength, in case it's not the right value currently.
            if "Skill Level" in new_param_values:
                new_param_values.update({"UCI_LimitStrength": "false"})
            elif "UCI_Elo" in new_param_values:
                new_param_values.update({"UCI_LimitStrength": "true"})

        if "Threads" in new_param_values and "Hash" in new_param_values:
            # Recommended to set the hash param after threads.
            threads_value = new_param_values["Threads"]
            del new_param_values["Threads"]
            hash_value = None
            if "Hash" in new_param_values:
                hash_value = new_param_values["Hash"]
                del new_param_values["Hash"]
            else:
                hash_value = self._parameters["Hash"]
            new_param_values["Threads"] = threads_value
            new_param_values["Hash"] = hash_value

        for name, value in new_param_values.items():
            self._set_option(name, value, True)
        self.set_fen_position(self.get_fen_position(), False)
        # Getting SF to set the position again, since UCI option(s) have been updated.

    def reset_engine_parameters(self) -> None:
        """Resets the stockfish parameters.

        Returns:
            None
        """
        self.update_engine_parameters(self._DEFAULT_PARAMS)

    def _prepare_for_new_position(self, send_ucinewgame_token: bool = True) -> None:
        if send_ucinewgame_token:
            self._put("ucinewgame")
        self._is_ready()
        self.info = ""

    def _put(self, command: str) -> None:
        if not self._uciengine.stdin:
            raise BrokenPipeError()
        if self._uciengine.poll() is None and not self._has_quit_command_been_sent:
            self._uciengine.stdin.write(f"{command}\n")
            self._uciengine.stdin.flush()
            if command == "quit":
                self._has_quit_command_been_sent = True

    def _read_line(self) -> str:
        if not self._uciengine.stdout:
            raise BrokenPipeError()
        if self._uciengine.poll() is not None:
            raise Exception("The UCIEngine process has crashed")
        return self._uciengine.stdout.readline().strip()

    def _set_option(
        self, name: str, value: Any, update_parameters_attribute: bool = True
    ) -> None:
        self._put(f"setoption name {name} value {value}")
        if update_parameters_attribute:
            self._parameters.update({name: value})
        self._is_ready()

    def _is_ready(self) -> None:
        self._put("isready")
        while self._read_line() != "readyok":
            pass

    def _go(self) -> None:
        if "maia" in self.mode:
            self._put(f"go nodes 1")
        else:
            self._put(f"go depth {self.depth}")

    def _go_time(self, time: int) -> None:
        self._put(f"go movetime {time}")

    def _go_searchmove(self, move, depth) -> None:
        self._put(f"go depth {depth} searchmoves {move}")

    def _go_remaining_time(self, wtime: Optional[int], btime: Optional[int]) -> None:
        cmd = "go"
        if wtime is not None:
            cmd += f" wtime {wtime}"
        if btime is not None:
            cmd += f" btime {btime}"
        self._put(cmd)

    def set_fen_position(
        self, fen_position: str, send_ucinewgame_token: bool = True
    ) -> None:
        """Sets current board position in Forsyth–Edwards notation (FEN).

        Args:
            fen_position:
              FEN string of board position.

            send_ucinewgame_token:
              Whether to send the "ucinewgame" token to the UCIEngine engine.
              The most prominent effect this will have is clearing UCIEngine's transposition table,
              which should be done if the new position is unrelated to the current position.

        Returns:
            None
        """
        self._prepare_for_new_position(send_ucinewgame_token)
        self._put(f"position fen {fen_position}")

    def set_position(self, moves: Optional[List[str]] = None) -> None:
        """Sets current board position.

        Args:
            moves:
              A list of moves to set this position on the board.
              Must be in full algebraic notation.
              example: ['e2e4', 'e7e5']
        """
        self.set_fen_position(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", True
        )
        self.make_moves_from_current_position(moves)

    def make_moves_from_current_position(self, moves: Optional[List[str]]) -> None:
        """Sets a new position by playing the moves from the current position.

        Args:
            moves:
              A list of moves to play in the current position, in order to reach a new position.
              Must be in full algebraic notation.
              Example: ["g4d7", "a8b8", "f1d1"]
        """
        if not moves:
            return
        self._prepare_for_new_position(False)
        for move in moves:
            if not self.is_move_correct(move):
                raise ValueError(f"Cannot make move: {move}")
            self._put(f"position fen {self.get_fen_position()} moves {move}")

    def get_board_visual(self, perspective_white: bool = True) -> str:
        """Returns a visual representation of the current board position.

        Args:
            perspective_white:
              A bool that indicates whether the board should be displayed from the
              perspective of white (True: white, False: black)

        Returns:
            String of visual representation of the chessboard with its pieces in current position.
        """
        self._put("d")
        board_rep_lines = []
        count_lines = 0
        while count_lines < 17:
            board_str = self._read_line()
            if "+" in board_str or "|" in board_str:
                count_lines += 1
                if perspective_white:
                    board_rep_lines.append(f"{board_str}")
                else:
                    # If the board is to be shown from black's point of view, all lines are
                    # inverted horizontally and at the end the order of the lines is reversed.
                    board_part = board_str[:33]
                    # To keep the displayed numbers on the right side,
                    # only the string representing the board is flipped.
                    number_part = board_str[33:] if len(board_str) > 33 else ""
                    board_rep_lines.append(f"{board_part[::-1]}{number_part}")
        if not perspective_white:
            board_rep_lines = board_rep_lines[::-1]
        board_str = self._read_line()
        if "a   b   c" in board_str:
            # Engine being used is recent enough to have coordinates, so add them:
            if perspective_white:
                board_rep_lines.append(f"  {board_str}")
            else:
                board_rep_lines.append(f"  {board_str[::-1]}")
        while "Checkers" not in self._read_line():
            # Gets rid of the remaining lines in _uciengine.stdout.
            # "Checkers" is in the last line outputted by UCIEngine for the "d" command.
            pass
        board_rep = "\n".join(board_rep_lines) + "\n"
        return board_rep

    def get_fen_position(self) -> str:
        """Returns current board position in Forsyth–Edwards notation (FEN).

        Returns:
            String with current position in Forsyth–Edwards notation (FEN)
        """
        if  'stockfish' in self.mode:
            self._put("d")
            while True:
                text = self._read_line()
                splitted_text = text.split(" ")
                if splitted_text[0] == "Fen:":
                    while "Checkers" not in self._read_line():
                        pass
                    return " ".join(splitted_text[1:])
        elif 'maia'in self.mode:
            self._put("fen")
            text = self._read_line()
            return " ".join(text)

    def set_skill_level(self, skill_level: int = 20) -> None:
        """Sets current skill level of stockfish engine.

        Args:
            skill_level:
              Skill Level option between 0 (weakest level) and 20 (full strength)

        Returns:
            None
        """
        self.update_engine_parameters(
            {"UCI_LimitStrength": "false", "Skill Level": skill_level}
        )

    def set_elo_rating(self, elo_rating: int = 1350) -> None:
        """Sets current elo rating of stockfish engine, ignoring skill level.

        Args:
            elo_rating: Aim for an engine strength of the given Elo

        Returns:
            None
        """
        self.update_engine_parameters(
            {"UCI_LimitStrength": "true", "UCI_Elo": elo_rating}
        )

    def evaluate_a_move(self, move: str = None, depth: int = None) -> Optional[str]:
        """Evaluate a move on the current board. A correct fen must be set before calling
        this function. If depth is not set, then it will search at stockfish depth-1. 

        Returns:
            A string of last line of info. 
        """
        if "stockfish" in self.mode:
            if depth:
                self._go_searchmove(move, depth)
            else:
                self._go_searchmove(move, depth=int(self.depth)-1)
        elif "maia" in self.mode:
            raise NotImplementedError("Maia doesn't support the move evaluation")
        return self._extract_last_info_from_popen_process()

    def get_best_move(self, wtime: int = None, btime: int = None) -> Optional[str]:
        """Returns best move with current position on the board.
        wtime and btime arguments influence the search only if provided.

        Returns:
            A string of move in algebraic notation or None, if it's a mate now.
        """
        if "stockfish" in self.mode:
            if wtime is not None or btime is not None:
                self._go_remaining_time(wtime, btime)
            else:
                self._go()
            print(f"running stockfish at {self.depth}")
        elif "maia" in self.mode:
            self._go()
            print("running maia")
        return self._parse_output_from_popen_process()

    def get_best_move_time(self, time: int = 1000) -> Optional[str]:
        """Returns best move with current position on the board after a determined time

        Args:
            time:
              Time for stockfish to determine best move in milliseconds (int)

        Returns:
            A string of move in algebraic notation or None, if it's a mate now.
        """
        self._go_time(time)
        return self._parse_output_from_popen_process()

    def _extract_last_info_from_popen_process(self) -> Optional[str]:
        """Don't update the info attribute of the object but only the last line of info

        Returns:
            Optional[str]: the last info line above the best move line
        """
        last_text: str = ""
        while True:
            text = self._read_line()
            splitted_text = text.split(" ")
            if splitted_text[0] == "bestmove":
                return last_text
            last_text = text

    def _parse_output_from_popen_process(self) -> Optional[List[str]]:
        # Precondition - a "go" command must have been sent to SF before calling this function.
        # This function needs existing output to read from the SF popen process.
        last_text: str = ""
        while True:
            text = self._read_line()
            # print(f'current line: {text}')
            splitted_text = text.split(" ")
            if splitted_text[0] == "bestmove":
                self.info = last_text
                return None if splitted_text[1] == "(none)" else [splitted_text[1], splitted_text[3]]
            last_text = text

    @staticmethod
    def _is_fen_syntax_valid(fen: str) -> bool:
        # Code for this function taken from: https://gist.github.com/Dani4kor/e1e8b439115878f8c6dcf127a4ed5d3e
        # Some small changes have been made to the code.

        regexMatch = re.match(
            r"\s*^(((?:[rnbqkpRNBQKP1-8]+\/){7})[rnbqkpRNBQKP1-8]+)\s([b|w])\s(-|[K|Q|k|q]{1,4})\s(-|[a-h][1-8])\s(\d+\s\d+)$",
            fen,
        )
        if not regexMatch:
            return False
        regexList = regexMatch.groups()
        if len(regexList[0].split("/")) != 8:
            return False  # 8 rows not present.
        for fenPart in regexList[0].split("/"):
            field_sum = 0
            previous_was_digit = False
            for c in fenPart:
                if c in ["1", "2", "3", "4", "5", "6", "7", "8"]:
                    if previous_was_digit:
                        return False  # Two digits next to each other.
                    field_sum += int(c)
                    previous_was_digit = True
                elif c.lower() in ["p", "n", "b", "r", "q", "k"]:
                    field_sum += 1
                    previous_was_digit = False
                else:
                    return False  # Invalid character.
            if field_sum != 8:
                return False  # One of the rows doesn't have 8 columns.
        return True

    def is_fen_valid(self, fen: str) -> bool:
        if not UCIEngine._is_fen_syntax_valid(fen):
            return False
        temp_sf = UCIEngine(path=self._path, parameters={"Hash": 1})
        # Using a new temporary SF instance, in case the fen is an illegal position that causes
        # the SF process to crash.
        best_move = None
        temp_sf.set_fen_position(fen, False)
        try:
            temp_sf._put("go depth 10")
            best_move, _ = temp_sf._parse_output_from_popen_process()
        except UCIException:
            # If a UCIException is thrown, then it happened in read_line() since the SF process crashed.
            # This is likely due to the position being illegal, so set the var to false:
            return False
        else:
            return best_move is not None
        finally:
            temp_sf.__del__()
            # Calling this function before returning from either the except or else block above.
            # The __del__ function should generally be called implicitly by python when this
            # temp_sf object goes out of scope, but calling it explicitly guarantees this will happen.

    def is_move_correct(self, move_value: str) -> bool:
        """Checks new move.

        Args:
            move_value:
              New move value in algebraic notation.

        Returns:
            True, if new move is correct, else False.
        """
        old_self_info = self.info
        self._put(f"go depth 1 searchmoves {move_value}")
        is_move_correct = self._parse_output_from_popen_process() is not None
        self.info = old_self_info
        return is_move_correct

    def get_wdl_stats(self) -> Optional[List]:
        """Returns UCIEngine's win/draw/loss stats for the side to move.

        Returns:
            A list of three integers, unless the game is over (in which case,
            None is returned).
        """

        if not self.does_current_engine_version_have_wdl_option():
            raise RuntimeError(
                "Your version of UCIEngine isn't recent enough to have the UCI_ShowWDL option."
            )
        self._go()
        lines = []
        while True:
            text = self._read_line()
            splitted_text = text.split(" ")
            lines.append(splitted_text)
            if splitted_text[0] == "bestmove":
                break
        for current_line in reversed(lines):
            if current_line[0] == "bestmove" and current_line[1] == "(none)":
                return None
            elif "multipv" in current_line:
                index_of_multipv = current_line.index("multipv")
                if current_line[index_of_multipv + 1] == "1" and "wdl" in current_line:
                    index_of_wdl = current_line.index("wdl")
                    wdl_stats = []
                    for i in range(1, 4):
                        wdl_stats.append(int(current_line[index_of_wdl + i]))
                    return wdl_stats
        raise RuntimeError("Reached the end of the get_wdl_stats function.")

    def get_evaluation(self) -> dict:
        """Evaluates current position

        Returns:
            A dictionary of the current advantage with "type" as "cp" (centipawns) or "mate" (checkmate in)
        """

        evaluation = dict()
        fen_position = self.get_fen_position()
        compare = 1 if "w" in fen_position else -1
        # UCIEngine shows advantage relative to current player. This function will instead
        # use positive to represent advantage white, and negative for advantage black.
        self._put(f"position {fen_position}")
        self._go()
        while True:
            text = self._read_line()
            splitted_text = text.split(" ")
            if splitted_text[0] == "info":
                for n in range(len(splitted_text)):
                    if splitted_text[n] == "score":
                        evaluation = {
                            "type": splitted_text[n + 1],
                            "value": int(splitted_text[n + 2]) * compare,
                        }
            elif splitted_text[0] == "bestmove":
                return evaluation

    def get_top_moves(self, num_top_moves: int = 5) -> List[dict]:
        """Returns info on the top moves in the position.

        Args:
            num_top_moves:
                The number of moves to return info on, assuming there are at least
                those many legal moves.

        Returns:
            A list of dictionaries. In each dictionary, there are keys for Move, Centipawn, and Mate;
            the corresponding value for either the Centipawn or Mate key will be None.
            If there are no moves in the position, an empty list is returned.
        """

        if num_top_moves <= 0:
            raise ValueError("num_top_moves is not a positive number.")
        old_MultiPV_value = self._parameters["MultiPV"]
        if num_top_moves != self._parameters["MultiPV"]:
            self._set_option("MultiPV", num_top_moves)
            self._parameters.update({"MultiPV": num_top_moves})
        self._go()
        lines = []
        while True:
            text = self._read_line()
            splitted_text = text.split(" ")
            lines.append(splitted_text)
            if splitted_text[0] == "bestmove":
                break
        top_moves: List[str] = []
        multipv_info = List[str] = []
        for current_line in reversed(lines):
            if current_line[0] == "bestmove":
                if current_line[1] == "(none)":
                    top_moves = []
                    break
            elif (
                ("multipv" in current_line)
                and ("depth" in current_line)
                and current_line[current_line.index("depth") + 1] == self.depth
            ):
                multiPV_number = int(current_line[current_line.index("multipv") + 1])
                if multiPV_number <= num_top_moves:
                    has_centipawn_value = "cp" in current_line
                    has_mate_value = "mate" in current_line
                    if has_centipawn_value == has_mate_value:
                        raise RuntimeError(
                            "Having a centipawn value and mate value should be mutually exclusive."
                        )
                    top_moves.insert(
                        0,
                        current_line[current_line.index("pv") + 1]
                    )
                    multipv_info.insert(
                        0,
                        current_line
                    )
            else:
                break
        self.multipv_info = multipv_info
        if old_MultiPV_value != self._parameters["MultiPV"]:
            self._set_option("MultiPV", old_MultiPV_value)
            self._parameters.update({"MultiPV": old_MultiPV_value})
        return top_moves

    @dataclass
    class BenchmarkParameters:
        ttSize: int = 16
        threads: int = 1
        limit: int = 13
        fenFile: str = "default"
        limitType: str = "depth"
        evalType: str = "mixed"

        def __post_init__(self):
            self.ttSize = self.ttSize if self.ttSize in range(1, 128001) else 16
            self.threads = self.threads if self.threads in range(1, 513) else 1
            self.limit = self.limit if self.limit in range(1, 10001) else 13
            self.fenFile = (
                self.fenFile
                if self.fenFile.endswith(".fen") and path.isfile(self.fenFile)
                else "default"
            )
            self.limitType = (
                self.limitType
                if self.limitType in ["depth", "perft", "nodes", "movetime"]
                else "depth"
            )
            self.evalType = (
                self.evalType
                if self.evalType in ["mixed", "classical", "NNUE"]
                else "mixed"
            )

    def benchmark(self, params: BenchmarkParameters) -> str:
        """Benchmark will run the bench command with BenchmarkParameters.
        It is an Additional custom non-UCI command, mainly for debugging.
        Do not use this command during a search!
        """
        if type(params) != self.BenchmarkParameters:
            params = self.BenchmarkParameters()

        self._put(
            f"bench {params.ttSize} {params.threads} {params.limit} {params.fenFile} {params.limitType} {params.evalType}"
        )
        while True:
            text = self._read_line()
            splitted_text = text.split(" ")
            if splitted_text[0] == "Nodes/second":
                return text

    def set_depth(self, depth_value: int = 2) -> None:
        """Sets current depth of stockfish engine.

        Args:
            depth_value: Depth option higher than 1
        """
        self.depth = str(depth_value)

    def __del__(self) -> None:
        UCIEngine._del_counter += 1
        if self._uciengine.poll() is None:
            self._put("quit")
            while self._uciengine.poll() is None:
                pass
