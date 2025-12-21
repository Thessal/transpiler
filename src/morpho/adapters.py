import abc
import os
import ollama
import json
import os
from typing import Optional, Type, TypeVar
from typing import Dict, Type, Any, Optional
from pydantic import BaseModel


class LLMAdapter(abc.ABC):
    @abc.abstractmethod
    def generate_strategies(self, system_context: str, user_prompt: str, schema: Type[BaseModel]) -> BaseModel:
        pass


class OllamaAdapter(LLMAdapter):
    def __init__(self, host: str, model: str):
        self.client = ollama.Client(host=host)
        self.model = model

    def generate_strategies(self, system_context: str, user_prompt: str, schema: Type[BaseModel]) -> BaseModel:
        response = self.client.chat(
            model=self.model,
            messages=[
                {'role': 'system', 'content': system_context},
                {'role': 'user', 'content': user_prompt},
            ],
            format=schema.model_json_schema(),
            options={'temperature': 0.7, 'num_ctx': 8192}
        )
        return schema.model_validate_json(response['message']['content'])


class GeminiAdapter(LLMAdapter):
    def __init__(self, api_key_env: str, model: str):
        from google import genai
        from google.genai import types

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} not set.")

        # New Client Initialization
        self.client = genai.Client(api_key=api_key)
        self.types = types
        self.model = model

    def generate_strategies(self, system_context: str, user_prompt: str, schema: Type[BaseModel]) -> BaseModel:
        full_prompt = f"{system_context}\n\n{user_prompt}"

        # New Generation Pattern using 'types.GenerateContentConfig'
        response = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config=self.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,  # Pass the Pydantic class directly
                temperature=0.7,
            )
        )

        # The new SDK might return a parsed object if configured,
        # but usually returns .text that we validate.
        return schema.model_validate_json(response.text)


class OpenAIAdapter(LLMAdapter):
    """
    - You can set `temperature` and `max_output_tokens`.
      :contentReference[oaicite:2]{index=2}
    """

    def __init__(
        self,
        api_key_env: str,
        model: str,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        timeout_s: float = 120.0,
    ):
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} not set.")

        self.client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout_s)

        # for model in self.client.list():
        #     print(model.id)

    def generate_strategies(
        self,
        system_context: str,
        user_prompt: str,
        schema: Type[BaseModel],
    ) -> BaseModel:
        messages = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.responses.parse(
            model=self.model,
            input=messages,
            text_format=schema,
            # temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise ValueError(
                "OpenAI returned no parsed output (possible refusal or empty response)."
            )

        return parsed


def get_adapter(config: Dict) -> LLMAdapter:
    provider = config.get("provider", "ollama").lower()

    if provider == "ollama":
        cfg = config.get("ollama", {})
        return OllamaAdapter(host=cfg.get("host"), model=cfg.get("model"))

    elif provider == "gemini":
        cfg = config.get("gemini", {})
        return GeminiAdapter(api_key_env=cfg.get("api_key_env"), model=cfg.get("model"))

    elif provider == "openai":
        cfg = config.get("openai", {})
        return OpenAIAdapter(api_key_env=cfg.get("api_key_env"), model=cfg.get("model"))

    else:
        raise ValueError(f"Unknown provider: {provider}")
