from chess_gpt_sim import *
import threading

# path for the generated game info and outputs 
input_folder = Path("/home/charles/panda/catkin_ws/src/move_chess_panda/scripts/logs/GptSim/to_GPT")
output_folder = Path("/home/charles/panda/catkin_ws/src/move_chess_panda/scripts/logs/GptSim/from_GPT")

# create config
config_path = Path(__file__).parent.joinpath("config.json")
system_msg_folder = Path(__file__).parent.joinpath("system_messages")
configs = {}
for system_msg_path in system_msg_folder.glob("*.txt"):
    configs[system_msg_path.stem] = create_config(config_path, system_msg_path)
print(f"find {len(configs)} configurations")

for game_info_path in input_folder.glob("*.json"):
    with open(game_info_path, "r") as file:
        game_info = json.load(file)
    output_path_p = output_folder.joinpath(game_info_path.stem)
    print(f"processing {game_info_path.name} informations")
    # iterate each config
    for name, config in configs.items():
        output_path = output_path_p.joinpath(name)
        if output_path.exists():
            print("the output folder exists. data can be overwritten.")
        else:
            output_path.mkdir(parents=True)
            print(f"createing the output folder for config {name}")

        # create queue
        q = queue.Queue()
        for move_info in game_info:
            q.put(move_info)

        num_threads = 10
        threads = []
        for i in range(num_threads):
            threads.append(threading.Thread(target=worker, args=(q, config, output_path)))

        for thread in threads:
            thread.start()

        q.join()

        for i in range(num_threads):
            q.put(None)
        for thread in threads:
            thread.join()
