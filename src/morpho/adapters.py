import abc
import os
import ollama
from google import genai
from google.genai import types
from typing import Dict, Type, Any
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
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} not set.")
        
        # New Client Initialization
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate_strategies(self, system_context: str, user_prompt: str, schema: Type[BaseModel]) -> BaseModel:
        full_prompt = f"{system_context}\n\n{user_prompt}"
        
        # New Generation Pattern using 'types.GenerateContentConfig'
        response = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema, # Pass the Pydantic class directly
                temperature=0.7,
            )
        )
        
        # The new SDK might return a parsed object if configured, 
        # but usually returns .text that we validate.
        return schema.model_validate_json(response.text)

def get_adapter(config: Dict) -> LLMAdapter:
    provider = config.get("provider", "ollama").lower()
    print(config)
    
    if provider == "ollama":
        cfg = config.get("ollama", {})
        return OllamaAdapter(host=cfg.get("host"), model=cfg.get("model"))
    
    elif provider == "gemini":
        cfg = config.get("gemini", {})
        print(cfg)
        return GeminiAdapter(api_key_env=cfg.get("api_key_env"), model=cfg.get("model"))
    
    else:
        raise ValueError(f"Unknown provider: {provider}")