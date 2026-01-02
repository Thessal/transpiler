from typing import Any, List
import pytest
from morpho import load_json, Transpiler


def transpiler_transpile(transpiler:Transpiler)->List[Any]:
    _ = transpiler.generate(system_context_args={}, user_prompt_args={"idea_text":"Implement an oscillator for daily trading"})
    # TODO
    result = []
    return result

def transpiler_verify(transpiler:Transpiler):
    # TODO
    pass

def transpiler_save(transpiler:Transpiler):
    # TODO
    pass

def test_transpiler():
    config = load_json("./tests/demo_config.json")
    transpiler = Transpiler(config, prompt_path="./tests/test_prompts/prompt.json")
    result = transpiler_transpile(transpiler)
    transpiler_verify(transpiler)
    transpiler_save(transpiler)

if __name__ == "__main__":
    pytest.main()