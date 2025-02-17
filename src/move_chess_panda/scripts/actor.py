#!/home/charles/panda/panda_env310/bin/python3.10
import os
import pygame
from io import BytesIO
import rospy
from std_msgs.msg import String, Bool
from matcha.cli import *
import torchaudio
import re

from elevenlabs import stream
from elevenlabs.client import ElevenLabs

key=os.getenv("ELEVENLAB_API_KEY")
client = ElevenLabs(api_key=key,)

text_to_speak = """I determine the exact 3D position of the board first by detecting the four ArUco markers placed on it 
and calculate the chessboard’s 3D grid. 
Then once I identify which square each piece is on through my occupancy and piece classification, 
I map that square to its corresponding 3D coordinates on the board"""

opening_to_speak = """Great I am playing e4 which is one of the most popular and aggressive openings for White, 
controlling the center and opening lines for my queen and bishop. 
Now if Black plays c5 they are introducing the Sicilian Defense.
Instead of directly placing a pawn in the center (like with e5), Black plays c5 to exert influence on the d4 square."""

movie_to_speak = """“This is the famous position from 2001: A Space Odyssey, where Frank Poole faced against the HAL 9000.
In the movie Poole as White captured on a6 with the queen and walked right into a forced mate. 
But if we analyze carefully White might try something safer—like shifting the queen to b7 instead of immediately grabbing the pawn. 
Or pushing a rook to e1. What would you like me to try? """

# text_to_speak = """I determine the exact 3D position of the board first by detecting the four ArUco markers placed on it. 
# These markers let me calculate the chessboard’s position in three-dimensional space."""

class GPTAudio:
    def __init__(self,) -> None:
        self.stop = True  # still necessary?

        self.mTTS_settings = self._init_mTTS()
        rospy.set_param("is_speaking", False)
        self.poses_sent = []
        self.change_pos_pub = rospy.Publisher("/change_pose", String, queue_size=10)
        self.chess_move_pub = rospy.Publisher("/chess_move", String, queue_size=10)

    def _init_mTTS(self):
        config = dict()
        config["model"] = "matcha_ljspeech"
        config["checkpoint_path"] = None
        config["vocoder"] = None
        config["text"] = "This is a test"
        config["file"] = None
        config["spk"] = None
        config["temperature"] = 0.66
        config["speaking_rate"] = 1.0
        config["steps"] = 10
        config["denoiser_strength"] = 0.00025
        config["output_folder"] = os.getcwd()
        paths = dict()
        paths["matcha"] = get_user_data_dir() / f"{config['model']}.ckpt"
        paths["vocoder"] = get_user_data_dir() / "hifigan_T2_v1"
        device = torch.device("cuda")
        model = load_matcha(config["model"], paths["matcha"], device)
        vocoder, denoiser = load_vocoder("hifigan_T2_v1", paths["vocoder"], device)
        spk = (
            torch.tensor([config["spk"]], device=device, dtype=torch.long)
            if config["spk"] is not None
            else None
        )
        return config, device, model, vocoder, denoiser, spk

    @torch.inference_mode()
    def mTTS(self, text, settings):
        config, device, model, vocoder, denoiser, spk = settings
        text = text.strip()
        text_processed = process_text(0, text, device)

        output = model.synthesise(
            text_processed["x"],
            text_processed["x_lengths"],
            n_timesteps=config["steps"],
            temperature=config["temperature"],
            spks=spk,
            length_scale=config["speaking_rate"],
        )
        output = to_waveform(output["mel"], vocoder, denoiser)
        return output.unsqueeze(0)

    def _audio_init(self):
        if pygame.mixer.get_init() is None:
            pygame.mixer.init()
            rospy.loginfo("audio init")
        self.stop = False

    def _audio_spin(self, last_sentence=False):
        if pygame.mixer.get_init() is None:
            raise Exception(f"audio not initialized!")
        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy() and not rospy.is_shutdown():
            pass
        if last_sentence:
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

    def elevenlab_speak_stream(self, text):
        audio_stream = client.text_to_speech.convert_as_stream(
            text=text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_multilingual_v2"
        )
        stream(audio_stream)

    def speak_sentence_one(self, sentence=text_to_speak, speech_aloud=True):
        # self._audio_init()
        rospy.set_param("is_speaking", True)
        segments = re.split(r'[,.]', sentence)
        for segment in segments:
            if segment and not segment.isspace():
                segment = segment.strip()
                while rospy.get_param("is_moving"):
                    rospy.sleep(0.1)
                if "markers" in segment and "markers" not in self.poses_sent:
                    print("send markers")
                    self.change_pos_pub.publish("markers")
                    self.poses_sent.append("markers")
                elif "occupancy" in segment and "occupancy" not in self.poses_sent:
                    print("send occupancy")
                    self.change_pos_pub.publish("squareE4")
                    self.poses_sent.append("occupancy")
                elif "coordinates" in segment and "coordinates" not in self.poses_sent:
                    print("send coordinates")
                    self.change_pos_pub.publish("rotate")
                    self.poses_sent.append("coordinates")
                if speech_aloud:
                    # self._audio_speak(segment)
                    # self._audio_spin()
                    self.elevenlab_speak_stream(segment)
        # self._audio_spin(last_sentence=True)
        rospy.set_param("is_speaking", False)

    def speak_sentence_two(self, sentence=text_to_speak, speech_aloud=True):
        # self._audio_init()
        rospy.set_param("is_speaking", True)
        segments = re.split(r'[,.]', sentence)
        for segment in segments:
            if segment and not segment.isspace():
                segment = segment.strip()
                while rospy.get_param("is_moving"):
                    rospy.sleep(0.1)
                if "e4" in segment and "e4" not in self.poses_sent:
                    print("send e4")
                    self.chess_move_pub.publish("e2e400000")
                    self.poses_sent.append("e4")
                elif "c5" in segment and "c5" not in self.poses_sent:
                    print("send c5")
                    self.chess_move_pub.publish("c7c500000")
                    self.poses_sent.append("c5")
                elif "e5" in segment and "e5" not in self.poses_sent:
                    print(f"send e5")
                    # self.change_pos_pub.publish("squareE5")
                    self.poses_sent.append("e5")
                elif "d4" in segment and "d4" not in self.poses_sent:
                    print("send d4")
                    self.change_pos_pub.publish("squareD4")
                    self.poses_sent.append("d4")
                if speech_aloud:
                    # self._audio_speak(segment)
                    # self._audio_spin()
                    self.elevenlab_speak_stream(segment)
        # self._audio_spin(last_sentence=True)
        rospy.set_param("is_speaking", False)
        self.change_pos_pub.publish("low")

    def speak_sentence_three(self, sentence=text_to_speak, speech_aloud=True):
        # self._audio_init()
        rospy.set_param("is_speaking", True)
        segments = re.split(r'[,.]', sentence)
        self.change_pos_pub.publish("stare")
        for segment in segments:
            if segment and not segment.isspace():
                segment = segment.strip()
                while rospy.get_param("is_moving"):
                    rospy.sleep(0.1)
                if "a6" in segment and "a6" not in self.poses_sent:
                    print("send a6")
                    self.change_pos_pub.publish("squareA6")
                    self.poses_sent.append("a6")
                elif "b7" in segment and "b7" not in self.poses_sent:
                    print("send b7")
                    self.change_pos_pub.publish("undoa8b700000")
                    self.poses_sent.append("b7")
                elif "e1" in segment and "e1" not in self.poses_sent:
                    print("send e1")
                    self.change_pos_pub.publish("undof1e100000")
                    self.poses_sent.append("e1")
                print(f"is moving? {rospy.get_param('is_moving')}")
                if speech_aloud:
                    # self._audio_speak(segment)
                    # self._audio_spin()
                    self.elevenlab_speak_stream(segment)
        # self._audio_spin(last_sentence=True)
        rospy.set_param("is_speaking", False)
        self.change_pos_pub.publish("low")

if __name__ == "__main__":
    try:
        my_gpt = GPTAudio()
        rospy.init_node("actor", anonymous=True, log_level=rospy.WARN)
        my_gpt.speak_sentence_one(text_to_speak, speech_aloud=True)
        # my_gpt.speak_sentence_two(opening_to_speak)
        # my_gpt.speak_sentence_three(movie_to_speak)
        # rospy.spin()

    except rospy.ROSInterruptException:
        pass
