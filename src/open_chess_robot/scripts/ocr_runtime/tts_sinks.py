"""Concrete TTS adapters for the LLM chess-commentary feature.

Each sink implements ``ocr_runtime.llm_commentary.TtsSink`` and
lazy-imports its SDK. Only ElevenLabs (online) is provided for now; local neural
TTS was dropped because its install footprint (torch/torchaudio/checkpoints) far
outweighs the benefit of a hosted voice. Add a local sink here later if needed.
"""

import logging
import os

from ocr_runtime.llm_commentary import TtsSink

logger = logging.getLogger(__name__)


class ElevenLabsSink(TtsSink):
    """Speak text via the ElevenLabs streaming API; blocks until playback ends."""

    def __init__(self, voice_id="JBFqnCBsd6RMkjVDRZzb",
                 model_id="eleven_multilingual_v2", api_key=None):
        self.voice_id = voice_id
        self.model_id = model_id
        # Accept either spelling of the env var (the legacy code used ELEVENLAB_).
        self._api_key = (api_key or os.environ.get("ELEVENLABS_API_KEY")
                         or os.environ.get("ELEVENLAB_API_KEY"))
        self._client = None
        self._stream = None

    def _ensure_client(self):
        if self._client is None:
            from elevenlabs import stream
            from elevenlabs.client import ElevenLabs
            self._client = ElevenLabs(api_key=self._api_key)
            self._stream = stream
        return self._client

    def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        client = self._ensure_client()
        audio_stream = client.text_to_speech.convert_as_stream(
            text=text, voice_id=self.voice_id, model_id=self.model_id)
        self._stream(audio_stream)  # blocking playback


#: Sink id -> sink class, for selection by rosparam in the node.
TTS_SINKS = {
    "elevenlabs": ElevenLabsSink,
}
