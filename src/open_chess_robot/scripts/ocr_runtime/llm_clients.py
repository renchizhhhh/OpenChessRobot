"""Concrete LLM adapters for the LLM chess-commentary feature.

Each adapter implements ``ocr_runtime.llm_commentary.LLMClient`` and
lazy-imports its SDK, so importing this module costs nothing and only the
selected provider's package (and API key) is required at runtime.

Adding a provider = one subclass implementing ``complete`` + ``stream``.
"""

import logging
import os
import time

from ocr_runtime.llm_commentary import LLMClient

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """OpenAI chat-completions adapter (default model gpt-4o).

    Consolidates the previous ``save_responce`` retry loop and ``create_llm``
    streaming, with a single normalized delta extraction (the old code read
    chunks two different ways). The OpenAI client is built lazily, not at import.
    """

    def __init__(self, model="gpt-4o", api_key=None, max_retry=10, pause=0.5):
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._max_retry = max_retry
        self._pause = pause
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _messages(self, config, user_input):
        # Mirrors the legacy create_messages: system message + fenced user input
        # folded into a single user turn.
        content = config.get("system_message", "")
        if user_input:
            content += f"\n```{user_input}```"
        return [{"role": "user", "content": content}]

    def _create(self, config, user_input, stream):
        return self._ensure_client().chat.completions.create(
            model=self.model,
            messages=self._messages(config, user_input),
            temperature=config.get("temperature"),
            max_tokens=config.get("max_tokens"),
            stream=stream,
        )

    def complete(self, config, user_input):
        for attempt in range(self._max_retry):
            try:
                response = self._create(config, user_input, stream=False)
                return response.choices[0].message.content
            except Exception as exc:
                logger.warning("OpenAI complete failed (try %d): %s", attempt + 1, exc)
                time.sleep(self._pause)
        raise RuntimeError(f"OpenAI request failed after {self._max_retry} retries")

    def stream(self, config, user_input):
        for chunk in self._create(config, user_input, stream=True):
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class GeminiClient(LLMClient):
    """Google Gemini adapter using the current ``google-genai`` SDK.

    (The older ``google-generativeai`` package is deprecated.) Default model
    gemini-2.5-flash; override in config/llm_commentary/llm_commentary.yaml.
    """

    def __init__(self, model="gemini-2.5-flash", api_key=None, max_retry=10, pause=0.5,
                 thinking_budget=None):
        self.model = model
        self._api_key = (api_key or os.environ.get("GEMINI_API_KEY")
                         or os.environ.get("GOOGLE_API_KEY"))
        self._max_retry = max_retry
        self._pause = pause
        # None -> leave the model's default thinking on; 0 disables it (2.5-flash),
        # trading a little depth for much lower latency on short commentary tasks.
        self._thinking_budget = thinking_budget
        self._client = None
        self._types = None

    def _ensure_client(self):
        if self._client is None:
            from google import genai
            from google.genai import types
            self._client = genai.Client(api_key=self._api_key)
            self._types = types
        return self._client

    def _gen_config(self, config):
        fields = {"system_instruction": config.get("system_message"),
                  "temperature": config.get("temperature"),
                  "max_output_tokens": config.get("max_tokens")}
        kwargs = {key: value for key, value in fields.items() if value is not None}
        if self._thinking_budget is not None:
            kwargs["thinking_config"] = self._types.ThinkingConfig(
                thinking_budget=self._thinking_budget)
        return self._types.GenerateContentConfig(**kwargs)

    def _contents(self, user_input):
        return f"```{user_input}```" if user_input else ""

    def complete(self, config, user_input):
        client = self._ensure_client()
        gen_config = self._gen_config(config)
        for attempt in range(self._max_retry):
            try:
                response = client.models.generate_content(
                    model=self.model, contents=self._contents(user_input),
                    config=gen_config)
                return response.text
            except Exception as exc:
                logger.warning("Gemini complete failed (try %d): %s", attempt + 1, exc)
                time.sleep(self._pause)
        raise RuntimeError(f"Gemini request failed after {self._max_retry} retries")

    def stream(self, config, user_input):
        client = self._ensure_client()
        gen_config = self._gen_config(config)
        for chunk in client.models.generate_content_stream(
                model=self.model, contents=self._contents(user_input),
                config=gen_config):
            if getattr(chunk, "text", None):
                yield chunk.text


#: Provider id -> client class, for selection by rosparam in the node.
LLM_CLIENTS = {
    "openai": OpenAIClient,
    "gemini": GeminiClient,
}
