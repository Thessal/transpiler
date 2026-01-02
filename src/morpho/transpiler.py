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
        self.system_context = "\n".join(self.template_config["system_context"])
        self.system_context_args = self.template_config["system_context_args"]
        self.user_prompt = "\n".join(self.template_config["user_prompt"])
        self.user_prompt_args = self.template_config["user_prompt_args"]

        # Initialize Generator
        self.generator_config = config
        self.generator = get_adapter(config, role="generator")
        print(
            f"[Init] Transpiler initialized using {self.generator_config["generator"].get('provider')}")

    def inject(self, system_context_args, user_prompt_args):
        assert self.system_context_args.keys() == system_context_args.keys()
        assert self.user_prompt_args.keys() == user_prompt_args.keys()
        system_context = self.system_context
        user_prompt = self.user_prompt
        if system_context_args:
            system_context = system_context.format(**system_context_args)
        if user_prompt_args:
            user_prompt = user_prompt.format(**user_prompt_args)
        return system_context, user_prompt

    def generate(self, system_context_args, user_prompt_args):
        system_context, user_prompt = self.inject(
            system_context_args, user_prompt_args)

        # Generate
        print("[Exec] Querying AI Model...")
        response_obj = self.generator.generate_strategies(
            system_context, user_prompt, StrategyResponse)

        # Save Artifacts
        print(
            f"[Save] Processing {len(response_obj.strategies)} generated strategies...")
        # self._save_artifacts(response_obj.strategies, {
        #     "source_files": {
        #         "idea": idea_file,
        #         "spec": spec_file,
        #         "docs": doc_files,
        #         "funcs": func_files
        #     },
        #     "prompts": {
        #         "system": system_context,
        #         "user": user_prompt
        #     }
        # })

#     def _save_artifacts(self, strategies: List[StrategyImplementation], metadata_context: Dict):
#         for strat in strategies:
#             # Generate Hash from the Code
#             code_hash = compute_hash(strat.code)

#             # File Paths
#             bf_path = self.output_dir / f"{code_hash}.bf"
#             json_path = self.output_dir / f"{code_hash}.json"

#             # 1. Write Code File (.bf)
#             # Only write if it doesn't exist to prevent overwrites (or remove check to overwrite)
#             if not bf_path.exists():
#                 bf_path.write_text(strat.code, encoding='utf-8')

#             # 2. Write Metadata File (.json)
#             metadata = {
#                 "hash": code_hash,
#                 "name": strat.name,
#                 "description": strat.description,
#                 "generated_at": str(Path().resolve()),  # or timestamp
#                 "inputs": metadata_context["source_files"],
#                 # We optionally include prompts here for reproducibility
#                 # "prompts": metadata_context["prompts"]
#             }

#             with open(json_path, 'w', encoding='utf-8') as f:
#                 json.dump(metadata, f, indent=2)

#         print(f"[Done] Artifacts saved to {self.output_dir}")
