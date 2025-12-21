import json
from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel, Field, ConfigDict
from .adapters import get_adapter
from .util import load_json, read_file, compute_hash

# --- Data Schemas ---


class StrategyImplementation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., description="Short name of the strategy variant.")
    description: str = Field(..., description="Logic and mathematical basis.")
    code: str = Field(..., description="Executable implementation code.")


class StrategyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategies: List[StrategyImplementation] = Field(
        ..., description="List of generated implementations.")


class StrategyWorkflow:
    def __init__(self, config_path: str = "config.json"):
        self.config = load_json(config_path)

        # Setup directories
        dirs = self.config.get("directories", {})
        self.input_dir = Path(dirs.get("input", "./inputs"))
        self.output_dir = Path(dirs.get("output", "./outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Adapter
        self.adapter = get_adapter(self.config["generator"])
        print(
            f"[Init] Workflow initialized using {self.config["generator"].get('provider')}")

    def run(self, idea_file: str, spec_file: str, doc_files: List[str], func_files: List[str]):
        """
        Main execution method.
        """
        # 1. Load Inputs
        print("[Load] Reading input files...")
        idea_text = read_file(self.input_dir / idea_file)
        spec_text = read_file(self.input_dir / idea_file)

        docs_text = "\n".join(
            [f"--- DOC: {f} ---\n{read_file(self.input_dir / f)}" for f in doc_files])
        funcs_text = "\n".join(
            [f"--- FUNC: {f} ---\n{read_file(self.input_dir / f)}" for f in func_files])

        # 2. Build Contexts
        system_context = (
            "You are an expert quantitative researcher.\n"
            f"### LANGUAGE SPEC ###\n{spec_text}\n"
            f"### EXISTING CODE ###\n{funcs_text}\n"
            f"### REFERENCE DOCS ###\n{docs_text}\n"
        )

        user_prompt = (
            f"### STRATEGY IDEA ###\n{idea_text}\n\n"
            "Generate a JSON response containing a list of `strategies`.\n"
            "Provide **at least 10 distinct implementations**.\n"
            "Vary parameters, execution logic, and edge case handling.\n"
            "When you write the code, refer the language spec and and the references."
        )

        # 3. Generate
        print("[Exec] Querying AI Model...")
        response_obj = self.adapter.generate_strategies(
            system_context, user_prompt, StrategyResponse)

        # 4. Save Artifacts
        print(
            f"[Save] Processing {len(response_obj.strategies)} generated strategies...")
        self._save_artifacts(response_obj.strategies, {
            "source_files": {
                "idea": idea_file,
                "spec": spec_file,
                "docs": doc_files,
                "funcs": func_files
            },
            "prompts": {
                "system": system_context,
                "user": user_prompt
            }
        })

    def _save_artifacts(self, strategies: List[StrategyImplementation], metadata_context: Dict):
        for strat in strategies:
            # Generate Hash from the Code
            code_hash = compute_hash(strat.code)

            # File Paths
            bf_path = self.output_dir / f"{code_hash}.bf"
            json_path = self.output_dir / f"{code_hash}.json"

            # 1. Write Code File (.bf)
            # Only write if it doesn't exist to prevent overwrites (or remove check to overwrite)
            if not bf_path.exists():
                bf_path.write_text(strat.code, encoding='utf-8')

            # 2. Write Metadata File (.json)
            metadata = {
                "hash": code_hash,
                "name": strat.name,
                "description": strat.description,
                "generated_at": str(Path().resolve()),  # or timestamp
                "inputs": metadata_context["source_files"],
                # We optionally include prompts here for reproducibility
                # "prompts": metadata_context["prompts"]
            }

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

        print(f"[Done] Artifacts saved to {self.output_dir}")
