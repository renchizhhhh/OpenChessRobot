from chess_gpt_sim import load_chess_games, generate_first_moves_info, generate_game_selective_info
import random
from pathlib import Path
import json

save_folder = "/home/charles/panda/catkin_ws/src/move_chess_panda/scripts/logs/GptSim"
# generate the moves for the first 
# generate_first_moves_info(log_path = save_folder)


# generate random moves from existing games
games = load_chess_games("/home/charles/panda/catkin_ws/src/move_chess_panda/scripts/data/games/Adams")
game_samples = random.sample(games, 10)

for i, game in enumerate(game_samples):
    move_index = random.sample(range(1, 10), 2)
    move_samples = [list(game)[m] for m in move_index] 
    generate_game_selective_info(game, i, move_samples, save_folder)
