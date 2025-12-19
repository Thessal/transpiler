from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Union, Set

# --- 1. Type Definitions ---

class Type:
    SIGNAL = "Signal"
    INT = "Integer"
    FLOAT = "Float"

@dataclass
class Signal:
    """Mock Data Container"""
    data: Any
    def __repr__(self): return f"<Sig {self.data}>"

@dataclass
class OperatorDef:
    """
    Defines the 'Checklist' for an operator.
    signature: Maps argument name to expected Type.
    """
    name: str
    signature: Dict[str, str]  # e.g., {'input': Type.SIGNAL, 'window': Type.INT}
    kernel: Callable

# --- 2. AST / Runtime Nodes ---

@dataclass
class AppliedOperator:
    """
    Represents an Operator State.
    bindings: The arguments we have collected so far.
    """
    definition: OperatorDef
    bindings: Dict[str, Any] = field(default_factory=dict)

    def get_missing_keys(self) -> Set[str]:
        required = set(self.definition.signature.keys())
        current = set(self.bindings.keys())
        return required - current

    def is_saturated(self) -> bool:
        return len(self.get_missing_keys()) == 0

    def __repr__(self):
        # Visualizes state: SMA(window=20, input=?)
        items = []
        for key in self.definition.signature:
            if key in self.bindings:
                val = self.bindings[key]
                items.append(f"{key}={val}")
            else:
                items.append(f"{key}=?")
        return f"{self.definition.name}({', '.join(items)})"

# --- 3. Named Reduction Engine ---

class Reducer:
    def __init__(self):
        self.ops = {}
        self._register_basics()

    def _register_basics(self):
        # Define SMA: requires 'input' (Signal) and 'window' (Int)
        self.ops["SMA"] = OperatorDef(
            name="SMA",
            signature={"input": Type.SIGNAL, "window": Type.INT},
            kernel=self._sma_kernel
        )
        # Define DIFF: requires 'a' and 'b'
        self.ops["DIFF"] = OperatorDef(
            name="DIFF",
            signature={"a": Type.SIGNAL, "b": Type.SIGNAL},
            kernel=lambda a, b: Signal(a.data - b.data)
        )

    def _sma_kernel(self, input, window):
        # Note: Kernel receives kwargs extracted from the bindings
        import numpy as np
        # Simple suffix sum for demo
        res = np.convolve(input.data, np.ones(window)/window, mode='valid')
        return Signal(res)

    def get_op(self, name):
        return AppliedOperator(self.ops[name])

    def apply(self, node: AppliedOperator, **kwargs) -> Union[AppliedOperator, Signal]:
        """
        Applies named arguments to an operator.
        """
        # 1. Validation: Ensure we aren't passing unknown arguments
        valid_keys = node.definition.signature.keys()
        for k in kwargs:
            if k not in valid_keys:
                raise ValueError(f"Argument '{k}' is not valid for {node.definition.name}. Expected: {list(valid_keys)}")

        # 2. Merge Bindings (Create new state, immutability preferred)
        new_bindings = node.bindings.copy()
        new_bindings.update(kwargs)
        
        new_node = AppliedOperator(node.definition, new_bindings)

        # 3. Saturation Check
        if new_node.is_saturated():
            print(f"  [Reduction] {new_node.definition.name} Saturated! Executing...")
            # Unpack the dict into the kernel function
            return new_node.definition.kernel(**new_node.bindings)
        else:
            missing = new_node.get_missing_keys()
            print(f"  [Partial]   Updated {node.definition.name}. Still missing: {missing}")
            return new_node

# --- 4. Demonstration ---

import numpy as np

engine = Reducer()
data_src = Signal(np.array([10, 20, 30, 40, 50]))

print("--- Scenario 1: Config-First (Currying) ---")
# User defines the generic "Smooth" operator first
# Note: We provide 'window' first. The engine doesn't care about order.
op_smooth = engine.apply(engine.get_op("SMA"), window=3)

print(f"Current State: {op_smooth}") 
# Output: SMA(window=3, input=?)

print("\n--- Scenario 2: Data Injection ---")
# Now we provide the missing 'input'
result = engine.apply(op_smooth, input=data_src)

print(f"Result: {result}")

print("\n--- Scenario 3: Mixed Order One-Shot ---")
# We can provide arguments in ANY order.
# Here we provide 'input' before 'window', opposite of how we might think conventionally.
res2 = engine.apply(engine.get_op("SMA"), input=data_src, window=2)
print(f"Result (Window=2): {res2}")