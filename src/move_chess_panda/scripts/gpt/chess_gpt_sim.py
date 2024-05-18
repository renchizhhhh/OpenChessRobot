from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from engine.wrapper import ChessEngineWrapper, cp_to_winrate
import chess
import chess.pgn
import numpy as np
from pathlib import Path
import typing
import time
import json
import queue
import openai
import openai.error
from tqdm import tqdm


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.6f} seconds.")
        return result

    return wrapper


def create_messages(config, user_msg=""):
    """combine the system message from config and the generated user message
    to create the input for GPT

    Args:
        config (dict): config file including the system message
        user_msg (str, optional): _description_. Defaults to "".

    Returns:
        _type_: _description_
    """
    messages = [{"role": "user", "content": ""}]
    message = config.get("system_message")
    input = f"```{user_msg}```"
    if user_msg:
        message += "\n" + input
    messages[0]["content"] = message
    return messages


def create_llm(config, messages, stream=True):
    return openai.ChatCompletion.create(
        model=config.get("model"),
        messages=messages,
        temperature=config.get("temperature"),
        max_tokens=config.get("max_tokens"),
        stream=stream,
    )

def create_config(config_path, system_msg_path):
    with open(config_path, "r") as file:
        config = json.load(file)
    openai.organization = ""
    openai.api_key = config["key"]
    # config["max_tokens"] = 500
    config["system_message"] = load_system_message(system_msg_path)
    return config


def save_responce(config, messages):
    max_retry = 10
    pause_duration = 0.5
    for _ in range(max_retry):
        try:
            response = create_llm(config, messages, stream=False)
            generated_text = response.choices[0].message.content
            token_expenses = response['usage']['total_tokens']
            
            return generated_text
        except Exception as e:
            print(f"failed with error: {e}. Will try again after {pause_duration}s")
            time.sleep(pause_duration)
    return f"API Failed after {max_retry} trials"


def load_system_message(path):
    with open(path, "r", encoding="utf-8") as file:
        doc = file.read().splitlines()
    system_msg = ""
    for line in doc:
        system_msg += line
    return system_msg


def load_chess_games(folder=""):
    """load games into arrays of moves

    Args:
        folder (str, optional): path for the csv file parsed from pgn. Defaults to "".

    Returns:
        games: arrays of games [[move]]
    """
    games = list()
    folder = Path(folder)
    games_list = folder.glob("*.csv")
    for file in np.sort(list(games_list)):
        games.append(np.genfromtxt(Path(folder).joinpath(file), delimiter=",", dtype=str))
    return games


def zip_board_info(board: chess.Board, move: chess.Move=None):
    """wrap the board and move information into dict

    Args:
        board (chess.Board): the board to analyze
        multipv (int): number of best moves to prob

    Returns:
        String: the encoded chess fen
    """
    msg_dict = dict()
    last_board = board.copy()
    if move:
        last_move = last_board.san(move)
    else:
        try:
            last_move = last_board.san(last_board.pop())
        except IndexError:
            print("Opps! The move stack is empty!")
    last_fen = last_board.fen().replace(" ", "_")
    history = chess.Board().variation_san(last_board.move_stack)
    msg_dict["fen"] = last_fen
    msg_dict["move"] = last_move
    msg_dict["history"] = history
    return msg_dict


# @timeit
def prepare_eva_msg(chess_eng: ChessEngineWrapper, board: chess.Board, move: str, turn: int):
    msg_dict = dict()
    info = chess_eng.evaluate_move(board, move)
    centipawn = info.get("score")
    centipawn *= turn
    win = cp_to_winrate(centipawn)
    pv = info.get("pv")
    moves = [chess.Move.from_uci(move) for move in pv]
    pv_san = board.variation_san(moves)
    msg_dict["centipawn"] = centipawn
    msg_dict["win"] = win
    msg_dict["pv"] = pv_san
    return msg_dict


@timeit
def generate_game_full_info(game: typing.List[str], idx: int, log_path: str):
    """read a game and generate corresponding evaluations using stockfish.
    Then save the generated evaluations into a text file for GPT analysis.
    """
    sf = ChessEngineWrapper("stockfish", depth=20)
    board = chess.Board()
    input_msgs = list()
    turn = 1
    file_path = Path(log_path).joinpath(f"info_game{idx}.json")
    for i, move in enumerate(game[:10]):
        eva_msg_dict = prepare_eva_msg(sf, board, move, turn)
        board.push_uci(move)
        turn *= -1
        board_msg_dict = zip_board_info(board)
        board_msg_dict.update(eva_msg_dict)
        board_msg_dict["move_count"] = str(i + 1)
        board_msg_dict["question"] = "can you analyse the game?"
        # combine the taliored message and the system message
        input_msgs.append(board_msg_dict)
    with open(file_path, "w") as file:
        json.dump(input_msgs, file, indent=4)
    sf.shutdown()


@timeit
def generate_game_selective_info(game: typing.List[str], game_idx: int, move_samples: typing.List[str], log_path: str):
    """read a game and generate corresponding evaluations using stockfish.
    Then save the generated evaluations into a text file for GPT analysis.
    """
    sf = ChessEngineWrapper("stockfish", depth=20)
    board = chess.Board()
    input_msgs = list()
    turn = 1
    file_path = Path(log_path).joinpath(f"info_selective_game{game_idx}.json")
    for i, move in enumerate(tqdm(game)):
        if move in move_samples:
            eva_msg_dict = prepare_eva_msg(sf, board, move, turn)
            board_msg_dict = zip_board_info(board, chess.Move.from_uci(move))
            board_msg_dict.update(eva_msg_dict)
            board_msg_dict["move_count"] = str(i + 1)
            board_msg_dict["question"] = "can you analyse the game?"
            input_msgs.append(board_msg_dict)
            move_samples.remove(move)
        if len(move_samples) == 0:
            break
        board.push_uci(move)
        turn *= -1
    with open(file_path, "w") as file:
        json.dump(input_msgs, file, indent=4)
    sf.shutdown()


@timeit
def generate_first_moves_info(log_path):
    """read a game and generate corresponding evaluations using stockfish.
    Then save the generated evaluations into a text file for GPT analysis.
    """
    sf = ChessEngineWrapper("stockfish", depth=20)
    board = chess.Board()
    input_msgs = list()
    turn = 1
    file_path = Path(log_path).joinpath(f"info_first_move.json")
    for i, move in enumerate(board.legal_moves):
        # eva_msg = prepare_eva_msg(sf, board, move, turn)
        eva_msg_dict = prepare_eva_msg(sf, board, move, turn, depth=40)
        board_msg_dict = zip_board_info(board, move)
        board_msg_dict.update(eva_msg_dict)
        board_msg_dict["move_count"] = str(i + 1)
        board_msg_dict["question"] = "can you analyse the game?"
        # combine the taliored message and the system message
        # input_msg = create_messages(config, user_msg)
        input_msgs.append(board_msg_dict)
    with open(file_path, "w") as file:
        json.dump(input_msgs, file, indent=4)
    sf.shutdown()


@timeit
def gpt_request(config, idx, game_info_path, log_path):
    # A mock function for gpt request
    file_path = Path(log_path).joinpath(f"gpt_game{idx}.json")
    outputs = list()
    with open(game_info_path, "r") as file:
        game_info = json.load(file)
    for move_info in game_info[:3]:
        out_dict = dict()
        input_msg = create_messages(config, str(move_info))
        output = save_responce(config, input_msg)
        out_dict["output"] = output
        out_dict["input"] = move_info
        outputs.append(out_dict)
    with open(file_path, "w") as file:
        json.dump(outputs, file, indent=4)


def json_logger(data, log_path, file_name):
    file_path = Path(log_path).joinpath(f"{file_name}.json")
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


@timeit
def worker(q: queue.Queue, config: dict, log_path: str):
    while True:
        move_info = q.get()
        if move_info is None:
            break
        out_dict = {}
        input_msg = create_messages(config, str(move_info))
        # print("send request")
        output = save_responce(config, input_msg)
        # print("receive response")
        out_dict["output"] = output
        out_dict["input"] = move_info
        json_logger(out_dict, log_path, move_info["move_count"] + move_info["move"])
        q.task_done()
