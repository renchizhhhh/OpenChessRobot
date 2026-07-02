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
        self._send_temperature = model.lower().startswith(("gpt-3.5", "gpt-4"))

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
        # Only pass optional params when set. Current models use
        # max_completion_tokens (not the deprecated max_tokens); temperature is
        # dropped once the API reports a custom value as unsupported.
        kwargs = {
            "model": self.model,
            "messages": self._messages(config, user_input),
            "stream": stream,
        }
        if self._send_temperature and config.get("temperature") is not None:
            kwargs["temperature"] = config["temperature"]
        if config.get("max_tokens") is not None:
            kwargs["max_completion_tokens"] = config["max_tokens"]
        return self._ensure_client().chat.completions.create(**kwargs)

    @staticmethod
    def _is_temperature_error(exc):
        msg = str(exc).lower()
        return "temperature" in msg and ("unsupported" in msg or "does not support" in msg)

    def complete(self, config, user_input):
        for attempt in range(self._max_retry):
            try:
                response = self._create(config, user_input, stream=False)
                return response.choices[0].message.content
            except Exception as exc:
                if self._send_temperature and self._is_temperature_error(exc):
                    logger.warning("OpenAI rejected a custom temperature; retrying without it")
                    self._send_temperature = False
                    continue
                logger.warning("OpenAI complete failed (try %d): %s", attempt + 1, exc)
                time.sleep(self._pause)
        raise RuntimeError(f"OpenAI request failed after {self._max_retry} retries")

    def stream(self, config, user_input):
        try:
            chunks = self._create(config, user_input, stream=True)
        except Exception as exc:
            if not (self._send_temperature and self._is_temperature_error(exc)):
                raise
            logger.warning("OpenAI rejected a custom temperature; retrying without it")
            self._send_temperature = False
            chunks = self._create(config, user_input, stream=True)
        for chunk in chunks:
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
        # Custom temperature is honoured by the gemini-1.5/2.0/2.5 families;
        # newer models only accept the default, so gate on the model name.
        self._send_temperature = any(
            tag in model.lower() for tag in ("gemini-1.5", "gemini-2.0", "gemini-2.5"))

    def _ensure_client(self):
        if self._client is None:
            from google import genai
            from google.genai import types
            self._client = genai.Client(api_key=self._api_key)
            self._types = types
        return self._client

    def _gen_config(self, config):
        fields = {"system_instruction": config.get("system_message"),
                  "max_output_tokens": config.get("max_tokens")}
        if self._send_temperature:
            fields["temperature"] = config.get("temperature")
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


class ClaudeClient(LLMClient):
    """Anthropic Claude adapter using the ``anthropic`` SDK.

    Default model claude-sonnet-5; override in config/llm_commentary/llm_commentary.yaml.
    max_tokens is required by the Messages API. Thinking is disabled so the whole
    token budget goes to the spoken reply and latency stays low (Sonnet 5 runs
    adaptive thinking by default, which would otherwise consume the budget).
    """

    def __init__(self, model="claude-sonnet-5", api_key=None, max_retry=10, pause=0.5):
        self.model = model
        self._api_key = (api_key or os.environ.get("CLAUDE_API_KEY")
                         or os.environ.get("ANTHROPIC_API_KEY"))
        self._max_retry = max_retry
        self._pause = pause
        self._client = None
        # Custom temperature is honoured by the pre-5 families (sonnet-4.x,
        # opus-4.5/4.6, haiku, claude-3); the sonnet-5 / opus-4.7+ tier rejects a
        # non-default value, so gate on the model name.
        m = model.lower()
        self._send_temperature = any(
            tag in m for tag in ("sonnet-4", "opus-4-5", "opus-4-6", "haiku", "claude-3"))

    def _ensure_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _params(self, config, user_input):
        params = {
            "model": self.model,
            "max_tokens": config.get("max_tokens") or 1024,
            "system": config.get("system_message", ""),
            "messages": [{"role": "user",
                          "content": f"```{user_input}```" if user_input else "."}],
            "thinking": {"type": "disabled"},
        }
        if self._send_temperature and config.get("temperature") is not None:
            params["temperature"] = config["temperature"]
        return params

    def complete(self, config, user_input):
        client = self._ensure_client()
        params = self._params(config, user_input)
        for attempt in range(self._max_retry):
            try:
                response = client.messages.create(**params)
                return next((b.text for b in response.content if b.type == "text"), "")
            except Exception as exc:
                logger.warning("Claude complete failed (try %d): %s", attempt + 1, exc)
                time.sleep(self._pause)
        raise RuntimeError(f"Claude request failed after {self._max_retry} retries")

    def stream(self, config, user_input):
        client = self._ensure_client()
        with client.messages.stream(**self._params(config, user_input)) as stream:
            for text in stream.text_stream:
                yield text


#: Provider id -> client class, for selection by rosparam in the node.
LLM_CLIENTS = {
    "openai": OpenAIClient,
    "gemini": GeminiClient,
    "claude": ClaudeClient,
}
