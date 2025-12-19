import re
from typing import List, Dict, Union, Any, Optional
from typesystem import *

# --- Define the Type Environment (The Rules) ---
# Function signatures
# Mapping function names to their Operator signatures based on your input
STD_LIB = {
    "data": Operator(Atomic("String"), Generic("Signal", Atomic("Float"))),
    
    "ts_mean": Operator(
        DictType({
            "signal": Either([Generic("Signal", Atomic("Float")), Generic("Signal", Atomic("Int"))]),
            "period": Atomic("Int") # Usually period is int, input said Float but standard is int
        }), 
        Generic("Signal", Atomic("Float"))
    ),
    
    "divide": Operator(
        DictType({
            "dividend": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
            "divisor":  Either([Generic("Signal", Atomic("Float")), Atomic("Float")])
        }),
        Generic("Signal", Atomic("Float"))
    ),
    
    "multiply": Operator(
        TupleType([Generic("Signal", Atomic("Float")), Generic("Signal", Atomic("Float"))]),
        Generic("Signal", Atomic("Float")) 
    ),
    
    "int": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
            "round": Atomic("Bool")
        }),
        Generic("Signal", Atomic("Int"))
    )
}

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Any, Callable, Dict, List, Optional, Tuple, Sequence
# import re
# import numpy as np
# from dataclasses import dataclass, field
# from typing import Dict, Any, Callable, Union, Set

# # --- 3. Named Reduction Engine ---

# class Reducer:
#     def __init__(self):
#         self.ops = {}
#         self._register_basics()

#     def _register_basics(self):
#         # Define SMA: requires 'input' (Signal) and 'window' (Int)
#         self.ops["SMA"] = OperatorDef(
#             name="SMA",
#             signature={"input": Type.SIGNAL, "window": Type.INT},
#             kernel=self._sma_kernel
#         )
#         # Define DIFF: requires 'a' and 'b'
#         self.ops["DIFF"] = OperatorDef(
#             name="DIFF",
#             signature={"a": Type.SIGNAL, "b": Type.SIGNAL},
#             kernel=lambda a, b: Signal(a.data - b.data)
#         )

#     def _sma_kernel(self, input, window):
#         # Note: Kernel receives kwargs extracted from the bindings
#         import numpy as np
#         # Simple suffix sum for demo
#         res = np.convolve(input.data, np.ones(window)/window, mode='valid')
#         return Signal(res)

#     def get_op(self, name):
#         return AppliedOperator(self.ops[name])

#     def apply(self, node: AppliedOperator, **kwargs) -> Union[AppliedOperator, Signal]:
#         """
#         Applies named arguments to an operator.
#         """
#         # 1. Validation: Ensure we aren't passing unknown arguments
#         valid_keys = node.definition.signature.keys()
#         for k in kwargs:
#             if k not in valid_keys:
#                 raise ValueError(f"Argument '{k}' is not valid for {node.definition.name}. Expected: {list(valid_keys)}")

#         # 2. Merge Bindings (Create new state, immutability preferred)
#         new_bindings = node.bindings.copy()
#         new_bindings.update(kwargs)
        
#         new_node = AppliedOperator(node.definition, new_bindings)

#         # 3. Saturation Check
#         if new_node.is_saturated():
#             print(f"  [Reduction] {new_node.definition.name} Saturated! Executing...")
#             # Unpack the dict into the kernel function
#             return new_node.definition.kernel(**new_node.bindings)
#         else:
#             missing = new_node.get_missing_keys()
#             print(f"  [Partial]   Updated {node.definition.name}. Still missing: {missing}")
#             return new_node
        



# # =========================
# # Evaluation
# # =========================
# def ensure_signal(v: Any, shape: Tuple[int, int]) -> np.ndarray:
#     if isinstance(v, np.ndarray):
#         return v
#     return np.full(shape, float(v), dtype=float)


# def eval_delay(k: int, x: np.ndarray) -> np.ndarray:
#     out = np.roll(x, shift=k, axis=0)
#     out[:k, :] = np.nan
#     return out


# def eval_rolling_mean(w: int, x: np.ndarray) -> np.ndarray:
#     T, N = x.shape
#     xv = np.where(np.isnan(x), 0.0, x)
#     xc = np.where(np.isnan(x), 0.0, 1.0)
#     csum_v = np.cumsum(xv, axis=0)
#     csum_c = np.cumsum(xc, axis=0)

#     out = np.full_like(x, np.nan, dtype=float)
#     for t in range(T):
#         t0 = t - w + 1
#         if t0 <= 0:
#             sv = csum_v[t]
#             sc = csum_c[t]
#         else:
#             sv = csum_v[t] - csum_v[t0 - 1]
#             sc = csum_c[t] - csum_c[t0 - 1]
#         out[t] = np.where(sc > 0, sv / sc, np.nan)
#     return out


# def eval_cs_rank(x: np.ndarray) -> np.ndarray:
#     T, N = x.shape
#     out = np.full_like(x, np.nan, dtype=float)
#     for t in range(T):
#         row = x[t]
#         mask = ~np.isnan(row)
#         vals = row[mask]
#         if vals.size == 0:
#             continue
#         order = np.argsort(vals, kind="mergesort")
#         ranks = np.empty_like(order, dtype=float)
#         ranks[order] = np.linspace(0.0, 1.0, num=vals.size, endpoint=True)
#         out[t, mask] = ranks
#     return out


# @dataclass
# class FnValue:
#     name: str
#     op_type: OpType
#     captured: Tuple[Any, ...]
#     ev: "Evaluator"

#     def __call__(self, *more: Any) -> Any:
#         args = self.captured + more
#         expected = list(tuple_items(self.op_type.in_type))

#         if len(args) < len(expected):
#             rem_in = mk_remaining_input([self.ev.runtime_type(a) for a in args], expected)
#             return FnValue(self.name, OpType(rem_in, self.op_type.out_type), args, self.ev)

#         if len(args) > len(expected):
#             raise TypeError(f"Too many args for {self.name}")

#         return self.ev.eval_builtin(self.name, args)


# class Evaluator:
#     def __init__(self, inputs: Dict[str, np.ndarray]):
#         self.inputs = inputs
#         any_arr = next(iter(inputs.values()))
#         self.shape = any_arr.shape
#         self.cache: Dict[int, Any] = {}

#     def runtime_type(self, v: Any) -> Type:
#         if isinstance(v, FnValue):
#             return v.op_type
#         if isinstance(v, np.ndarray):
#             return SIGR
#         if isinstance(v, int):
#             return IntParamType()
#         if isinstance(v, float):
#             return FloatParamType()
#         if isinstance(v, tuple):
#             return TupleType(tuple(self.runtime_type(x) for x in v))
#         raise TypeError(type(v))

#     def eval(self, n: Node) -> Any:
#         k = id(n)
#         if k in self.cache:
#             return self.cache[k]

#         if n.op == "lit":
#             self.cache[k] = n.payload
#             return n.payload

#         if n.op == "opconst":
#             op_t = n.typ
#             assert isinstance(op_t, OpType)
#             v = FnValue(n.payload, op_t, tuple(), self)
#             self.cache[k] = v
#             return v

#         if n.op == "input":
#             v = self.inputs[n.payload]
#             self.cache[k] = v
#             return v

#         if n.op == "tuple":
#             v = tuple(self.eval(a) for a in n.args)
#             self.cache[k] = v
#             return v

#         if n.op == "apply":
#             fn = self.eval(n.args[0])
#             args = [self.eval(a) for a in n.args[1:]]
#             if len(args) == 1 and isinstance(args[0], tuple):
#                 args = list(args[0])
#             v = fn(*args)  # FnValue or python callable
#             self.cache[k] = v
#             return v

#         if n.op in {"add", "sub", "mul", "div", "pow"}:
#             a0 = self.eval(n.args[0])
#             a1 = self.eval(n.args[1])
#             if is_signal(n.typ):
#                 x = ensure_signal(a0, self.shape)
#                 y = ensure_signal(a1, self.shape)
#                 if n.op == "add": v = x + y
#                 elif n.op == "sub": v = x - y
#                 elif n.op == "mul": v = x * y
#                 elif n.op == "div": v = x / y
#                 else: v = np.power(x, y)
#             else:
#                 x = float(a0) if is_floatparam(n.typ) else int(a0)
#                 y = float(a1) if is_floatparam(n.typ) else int(a1)
#                 if n.op == "add": v = x + y
#                 elif n.op == "sub": v = x - y
#                 elif n.op == "mul": v = x * y
#                 elif n.op == "div": v = x / y
#                 else: v = x ** y
#             self.cache[k] = v
#             return v

#         if n.op in {"delay", "rolling_mean", "cs_rank", "compose"}:
#             args = [self.eval(a) for a in n.args]
#             v = self.eval_builtin(n.op, args)
#             self.cache[k] = v
#             return v

#         raise RuntimeError(f"Unknown op {n.op}")

#     def eval_builtin(self, name: str, args: Sequence[Any]) -> Any:
#         if name == "delay":
#             return eval_delay(int(args[0]), ensure_signal(args[1], self.shape))
#         if name == "rolling_mean":
#             return eval_rolling_mean(int(args[0]), ensure_signal(args[1], self.shape))
#         if name == "cs_rank":
#             return eval_cs_rank(ensure_signal(args[0], self.shape))
#         if name == "compose":
#             f, g = args
#             if not isinstance(f, FnValue) or not isinstance(g, FnValue):
#                 raise TypeError("compose expects operator values")
#             return lambda x: g(f(x))
#         raise RuntimeError(name)
