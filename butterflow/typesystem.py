import re
from typing import List, Dict, Union, Any, Optional

# ==========================================
# PHASE 1: Type System & Schema Definition
# ==========================================

class Type:
    def matches(self, other: 'Type') -> bool:
        raise NotImplementedError

class AnyType(Type):
    def matches(self, other): return True
    def __repr__(self): return "Any"

class Atomic(Type):
    def __init__(self, name): self.name = name
    def matches(self, other): return isinstance(other, Atomic) and self.name == other.name
    def __repr__(self): return self.name

class Generic(Type):
    def __init__(self, base, inner):
        self.base = base
        self.inner = inner
    def matches(self, other):
        return isinstance(other, Generic) and self.base == other.base and self.inner.matches(other.inner)
    def __repr__(self): return f"{self.base}<{self.inner}>"

class Either(Type):
    def __init__(self, options: List[Type]): self.options = options
    def matches(self, other):
        return any(opt.matches(other) for opt in self.options)
    def __repr__(self): return f"Either({self.options})"

class DictType(Type):
    def __init__(self, fields: Dict[str, Type]): self.fields = fields
    def matches(self, other):
        if not isinstance(other, DictType): return False
        # Check if all required fields in self exist in other and match types
        for k, t in self.fields.items():
            if k not in other.fields: return False
            if not t.matches(other.fields[k]): return False
        return True
    def __repr__(self): return f"Dict({self.fields})"

class TupleType(Type):
    def __init__(self, items: List[Type]): self.items = items
    def matches(self, other):
        if not isinstance(other, TupleType) or len(self.items) != len(other.items): return False
        return all(s.matches(o) for s, o in zip(self.items, other.items))
    def __repr__(self): return f"Tuple({self.items})"

# The "Operator" type represents a function signature: Operator<Args, ReturnType>
class Operator(Type):
    def __init__(self, args: Type, ret: Type):
        self.args = args # Can be DictType or TupleType
        self.ret = ret
    def matches(self, other):
        # An Operator matches another if their return types match 
        # (Covariance usually, but strict equality for now)
        if not isinstance(other, Operator): return False
        return self.ret.matches(other.ret)
    def __repr__(self): return f"Operator<{self.args}, {self.ret}>"


# from dataclasses import dataclass, field
# from typing import Dict, Any, Callable, Union, Set, Tuple

# # --- 1. Type Definitions ---

# class Type: pass

# @dataclass(frozen=True)
# class ScalarType(Type):
#     name: str  # "Real" | "Int" | "Bool" | "String"

# @dataclass(frozen=True)
# class SignalType:
#     """Mock Data Container"""
#     name: str
#     data: Any
#     def __repr__(self): return f"<Sig {self.data}>"

# @dataclass(frozen=True)
# class OperatorDef:
#     """
#     Defines the 'Checklist' for an operator.
#     signature: Maps argument name to expected Type.
#     """
#     name: str
#     signature: Dict[str, str]  # e.g., {'input': Type.SIGNAL, 'window': Type.INT}
#     kernel: Callable

# @dataclass(frozen=True)
# class TupleType(Type):
#     items: Dict[str, str]  # e.g., {'input': Type.SIGNAL, 'window': Type.INT}


# @dataclass(frozen=True)
# class OpType(Type):
#     in_type: Type
#     out_type: Type



# # =========================
# # Types
# # =========================
# class Type: ...

# @dataclass(frozen=True)
# class TupleType(Type):
#     items: Tuple[Type, ...]

# @dataclass(frozen=True)
# class ScalarType(Type):
#     name: str  # "Real" | "Int" | "Bool"


# REAL = ScalarType("Real")
# INT  = ScalarType("Int")
# BOOL = ScalarType("Bool")


# @dataclass(frozen=True)
# class SignalType(Type):
#     scalar: ScalarType = REAL


# @dataclass(frozen=True)
# class IntParamType(Type): ...


# @dataclass(frozen=True)
# class FloatParamType(Type): ...


# @dataclass(frozen=True)
# class TupleType(Type):
#     items: Tuple[Type, ...]


# @dataclass(frozen=True)
# class OpType(Type):
#     in_type: Type
#     out_type: Type

# @dataclass(frozen=True)
# class SignalType(Type):
#     scalar: ScalarType = REAL


# @dataclass(frozen=True)
# class IntParamType(Type): ...


# @dataclass(frozen=True)
# class FloatParamType(Type): ...


# @dataclass(frozen=True)
# class TupleType(Type):
#     items: Tuple[Type, ...]


# @dataclass(frozen=True)
# class OpType(Type):
#     in_type: Type
#     out_type: Type
