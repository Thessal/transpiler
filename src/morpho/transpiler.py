from typing import List, Dict
from pydantic import BaseModel, Field, ConfigDict
from morpho.util import load_json
from morpho.adapters import get_adapter


# Data Schemas


class StrategyImplementation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., description="Short name of the strategy variant.")
    description: str = Field(..., description="Logic and mathematical basis.")
    code: str = Field(..., description="Executable implementation code.")


class StrategyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategies: List[StrategyImplementation] = Field(
        ..., description="List of generated implementations.")


# Workflow

class Transpiler:
    def __init__(self, config: Dict, prompt_path: str):
        # Load template
        self.template_config = load_json(prompt_path)
        self.system_context = self.template_config["system_context"]
        self.user_prompt = self.template_config["user_prompt"]

        # Initialize Generator
        self.generator_config = config
        self.generator = get_adapter(config, role="generator")
        print(
            f"[Init] Transpiler initialized using {self.generator_config["generator"].get('provider')}")

    def _inject(self, system_context_args, user_prompt_args):
        system_context = self.system_context
        user_prompt = self.user_prompt
        for k, v in system_context_args.items():
            marker = "{{"+k+"}}"
            assert marker in system_context
            system_context = system_context.replace(marker, v)
        for k, v in user_prompt_args.items():
            marker = "{{"+k+"}}"
            assert marker in user_prompt
            user_prompt = user_prompt.replace(marker, v)
        return system_context, user_prompt

    def generate(self, system_context_args, user_prompt_args, syntax_score=False, sematic_score=False):
        try:
            system_context, user_prompt = self._inject(
                system_context_args, user_prompt_args)
        except Exception as e:
            raise Exception("\n".join([
                f"Failed to build prompt : {repr(e)}\n" +
                f"[system context]\n",
                f"{self.system_context}",
                f"[user prompt]\n",
                f"{self.user_prompt}",
                f"[args]\n",
                f"{system_context_args}, {user_prompt_args}",
            ]))

        # Generate
        # print("[Exec] Querying AI Model...")
        response_obj = self.generator.generate_strategies(
            system_context, user_prompt, StrategyResponse)

        return response_obj
