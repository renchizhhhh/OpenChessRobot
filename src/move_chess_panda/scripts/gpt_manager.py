#!/home/charles/panda/panda_env310/bin/python3.10

import openai
import pygame
from io import BytesIO
from pathlib import Path

import rospy
from std_msgs.msg import String, Bool
from utili.logger import setup_logger
from gpt import create_config, create_llm, create_messages

from matcha.cli import *
import torchaudio

import re
import json

class GPTAssistant:
    def __init__(self, config: dict, device: str, model_name: str = 'gpt-4') -> None:
        self.model_name = model_name
        self.config = config
        self.stop = True
        openai.api_key = config.get("key")
        self.logger = setup_logger("gpt_logger", "/home/charles/panda/catkin_ws/src/move_chess_panda/scripts/logs/gpt.txt")
        if device == 'local':
            self.mTTS_settings = self._init_mTTS()
        if device == 'holo':
            self.gpt_pub = rospy.Publisher("/chat_to_hololens", String, queue_size=50, latch=False)
    
    def _init_mTTS(self):
        config = dict()
        config['model'] = "matcha_ljspeech"
        config['checkpoint_path'] = None
        config['vocoder'] = None
        config['text'] = "This is a test"
        config['file'] = None
        config['spk'] = None
        config['temperature'] = 0.66
        config['speaking_rate'] = 1.0
        config['steps'] = 10
        config['denoiser_strength'] = 0.00025
        config['output_folder'] = os.getcwd()
        paths = dict()
        paths["matcha"] = get_user_data_dir() / f"{config['model']}.ckpt"
        paths["vocoder"] = get_user_data_dir() / "hifigan_T2_v1"
        device = torch.device("cuda")
        model = load_matcha(config['model'], paths["matcha"], device)
        vocoder, denoiser = load_vocoder("hifigan_T2_v1", paths["vocoder"], device)
        spk = torch.tensor([config['spk']], device=device, dtype=torch.long) if config['spk'] is not None else None
        return config, device, model, vocoder, denoiser, spk

    @torch.inference_mode()
    def mTTS(self, text, settings):
        config, device, model, vocoder, denoiser, spk = settings
        text = text.strip()
        text_processed = process_text(0, text, device)

        output = model.synthesise(
            text_processed["x"],
            text_processed["x_lengths"],
            n_timesteps=config['steps'],
            temperature=config['temperature'],
            spks=spk,
            length_scale=config['speaking_rate'],
        )
        output = to_waveform(output["mel"], vocoder, denoiser)
        return output.unsqueeze(0)

    def _audio_init(self):
        if pygame.mixer.get_init() is None:
            pygame.mixer.init()
            rospy.loginfo("audio init")
        self.stop = False

    def _audio_spin(self):
        if pygame.mixer.get_init() is None:
            raise Exception(f"audio not initialized!")
        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy() and not rospy.is_shutdown():
            pass
        pygame.mixer.quit()
        rospy.loginfo("audio quit")
        self.stop = True

    def _audio_break(self):
        if pygame.mixer.get_init() is None:
            raise Exception(f"audio not initialized!")
        pygame.mixer.quit()
        rospy.logwarn("audio stopped middle way")

    def _audio_speak(self, text):
        audio_file = BytesIO()
        audio_data = self.mTTS(text, self.mTTS_settings)
        audio_sr = int(22050)
        torchaudio.save(audio_file, audio_data, audio_sr, format="wav")
        # Play the audio file using Pygame
        audio_file.seek(0)
        pygame.mixer.music.load(audio_file, "wav")
        pygame.mixer.music.play(1)

    def pub_GPT_stream(self, user_request=""):
        """
        A chat completion function using the specified OpenAI model. This function
        will publish the GPT feedback to HoloLens for the verbal interaction. 
        Reference: https://platform.openai.com/docs/api-reference/chat

        Args:
            user_request: the prompt to send to the OpenAI server.
        """
        sentense = ""
        if not user_request:
            raise Exception("The prompt cannot be empty!")
        user_request = create_messages(self.config, user_request)
        output = ""
        for chunk in create_llm(self.config, user_request):
            content = chunk["choices"][0].get("delta", {}).get("content")
            if content is not None:
                sentense += content
                if content.strip() in [",","."]:
                    self.gpt_pub.publish(sentense)
                    rospy.sleep(0.2)
                    print(f"Publish to Hololens: {sentense}", end='\n')
                    output += sentense
                    sentense = ""
        self.logger.info(f"GPT output: {output}")

    def speak_complete_stream(self, user_request=""):
        """
        A chat completion function using the specified OpenAI model. This function
        will use the local audio device to speak the feedback. 
        Reference: https://platform.openai.com/docs/api-reference/chat

        Args:
            user_request: the prompt to send to the OpenAI server.
        """
        sentense = ""
        if not user_request:
            raise Exception("The user_request message is empty!")
        self._audio_init()
        # self._audio_speak("Starting analyzing, please wait.")
        user_request = create_messages(self.config, user_request)
        output = ""
        rospy.set_param('is_speaking', True)
        for chunk in create_llm(self.config, user_request):
            content = chunk["choices"][0].get("delta", {}).get("content")
            if content is not None:
                sentense += content
                if content.strip() in [",","."]:
                    while pygame.mixer.music.get_busy() and not rospy.is_shutdown():
                        rospy.sleep(0.1)
                    self._audio_speak(sentense)
                    print(f"speak {sentense}", end='\n')
                    output += sentense
                    sentense = ""
        self.logger.info(f"GPT output: {output}")
        if not self.stop:
            self._audio_spin()
        rospy.set_param('is_speaking', False)


def analyze_callback(msg, assistant: GPTAssistant):
    """

    Args:
        msg: the prompt to send to the OpenAI server.
        assistant: the GPT manager 
    """
    move_input = msg.data
    move = re.search(r"'move':'(\w+)'", move_input)
    cached = ['Nh3', 'Nf3', 'Nc3', 'Na3', 'h3', 'g3', 'f3', 'e3', 'd3', 'c3',
                'b3', 'a3', 'h4', 'g4', 'f4', 'e4', 'd4', 'c4', 'b4', 'a4']
    if move in cached:
        cached_path = f"/home/charles/panda/catkin_ws/src/move_chess_panda/scripts/data/cache_gpt/open/{move}.json"
        with open(cached_path, "r") as file:
            move_analysis = json.load(file)
            move_analysis.get('output')
        assistant._audio_speak(move_analysis)
    else:
        print("No match found!")
        # TODO: update the conversation to general topics
        move_input += "'question': 'can you analyse the game?'"
        rospy.loginfo("Message received: /chess_fen_and_moves")
    #   user_msg = "{'question':'can you analyse?',\
    #   'fen':'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1', 'move':'b5', \
    #   'history': '1. e4'}"
        assistant.speak_complete_stream(move_input)

def test_analyze_callback(msg, assistant: GPTAssistant):
    user_request = msg.data
    assistant.pub_GPT_stream(user_request)

if __name__ == "__main__":
    try:
        config_path = Path(__file__).parent.joinpath("gpt/config.json")
        system_msg_path = Path(__file__).parent.joinpath("gpt/system_messages/open_1_eg.txt")
        config = create_config(config_path, system_msg_path)
        my_gpt = GPTAssistant(device='local', config=config)

        rospy.init_node("gpt_manager", anonymous=True, log_level=rospy.WARN)
        rospy.loginfo("node initialized")
        # rospy.Subscriber("/chess_chat", String, analyze_callback, my_gpt)
        rospy.Subscriber("/chess_chat", String, test_analyze_callback, my_gpt)
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
