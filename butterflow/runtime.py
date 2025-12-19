from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Sequence
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Union, Set

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
    



# =========================
# DAG
# =========================
@dataclass(frozen=True)
class Node:
    op: str
    args: Tuple["Node", ...]
    typ: Type
    payload: Any = None


class DagBuilder:
    def __init__(self):
        self.memo: Dict[Tuple[Any, ...], Node] = {}

    def mk(self, op: str, args: Tuple[Node, ...], typ: Type, payload: Any = None) -> Node:
        key = (op, typ, payload, tuple(id(a) for a in args))
        if key in self.memo:
            return self.memo[key]
        n = Node(op, args, typ, payload)
        self.memo[key] = n
        return n


def lower(e: Expr, types: Dict[int, Type], nodes: Dict[str, Node], mk: Callable[..., Node]) -> Node:
    t = types[id(e)]

    if isinstance(e, EVar):
        return nodes[e.name]
    if isinstance(e, ELitInt):
        return mk("lit", tuple(), IntParamType(), payload=e.value)
    if isinstance(e, ELitFloat):
        return mk("lit", tuple(), FloatParamType(), payload=e.value)
    if isinstance(e, ELitString):
        return mk("lit", tuple(), FloatParamType(), payload=e.value)

    if isinstance(e, ETuple):
        kids = tuple(lower(x, types, nodes, mk) for x in e.items)
        return mk("tuple", kids, t, payload=None)

    if isinstance(e, EUnary):
        inner = lower(e.inner, types, nodes, mk)
        if e.op == "+":
            return inner
        if e.op == "-":
            zero = mk("lit", tuple(), FloatParamType(), payload=0.0)
            return mk("sub", (zero, inner), t, payload=None)

    if isinstance(e, EBinOp):
        l = lower(e.left, types, nodes, mk)
        r = lower(e.right, types, nodes, mk)
        opmap = {"+": "add", "-": "sub", "*": "mul", "/": "div", "^": "pow"}
        return mk(opmap[e.op], (l, r), t, payload=None)

    if isinstance(e, EApp):
        fn = lower(e.fn, types, nodes, mk)
        args = tuple(lower(a, types, nodes, mk) for a in e.args)
        return mk("apply", (fn,) + args, t, payload=None)

    raise TypeError(e)


# =========================
# Reduction: flatten apply + reduce fully-matched operators
# =========================
class Reducer:
    def __init__(self, mk: Callable[..., Node]):
        self.mk = mk

    def reduce(self, n: Node, max_iters: int = 100) -> Node:
        cur = n
        for _ in range(max_iters):
            ch, cur = self._once(cur)
            if not ch:
                return cur
        return cur

    def _once(self, n: Node) -> Tuple[bool, Node]:
        changed = False
        new_args = []
        for a in n.args:
            ch, na = self._once(a)
            changed = changed or ch
            new_args.append(na)
        if changed:
            n = self.mk(n.op, tuple(new_args), n.typ, payload=n.payload)

        # (1) normalize: apply(apply(f,a), b) -> apply(f,a,b)
        if n.op == "apply" and n.args and n.args[0].op == "apply":
            inner = n.args[0]
            flat = (inner.args[0],) + inner.args[1:] + n.args[1:]
            return True, self.mk("apply", flat, n.typ, payload=None)

        # (2) reduce fully matched builtins: apply(opconst(name), ...) -> name(...)
        if n.op == "apply" and n.args and n.args[0].op == "opconst":
            op_node = n.args[0]
            op_name = op_node.payload
            op_t = op_node.typ
            assert isinstance(op_t, OpType)

            expected = list(tuple_items(op_t.in_type))
            provided = list(n.args[1:])

            # tuple-as-single-arg: op((a,b)) expands
            if len(provided) == 1 and type_eq(provided[0].typ, op_t.in_type) and provided[0].op == "tuple":
                provided = list(provided[0].args)

            if len(provided) == len(expected):
                return True, self.mk(op_name, tuple(provided), op_t.out_type, payload=None)

            # partial apply: adjust node type to the curried operator type
            if len(provided) < len(expected):
                rem_in = mk_remaining_input([p.typ for p in provided], expected)
                new_t = OpType(rem_in, op_t.out_type)
                if not type_eq(new_t, n.typ):
                    return True, self.mk("apply", n.args, new_t, payload=None)

        # (3) reduce compose-at-application: apply(compose(f,g), x) -> apply(g, apply(f,x))
        if n.op == "apply" and len(n.args) == 2:
            fn, x = n.args
            if fn.op == "compose" and len(fn.args) == 2:
                f, g = fn.args
                inner = self.mk("apply", (f, x), g.typ.in_type if isinstance(g.typ, OpType) else g.typ, payload=None)
                outer = self.mk("apply", (g, inner), n.typ, payload=None)
                return True, outer

        return False, n
    



# =========================
# Compile pipeline
# =========================
def compile_program(src: str) -> Tuple[List[Binding], Dict[str, Node], DagBuilder]:
    binds = parse(src)
    for b in binds:
        if b.name in RESERVED:
            raise TypeError(f"'{b.name}' is reserved (builtin)")

    # env types: builtins + user declared
    declared = dict(BUILTIN_TYPES)
    for b in binds:
        declared[b.name] = b.typ

    tc = TypeChecker(declared)
    dag = DagBuilder()
    nodes: Dict[str, Node] = {}

    # builtins as opconst nodes (except input special-form)
    for name, t in BUILTIN_TYPES.items():
        if name == "input":
            continue
        nodes[name] = dag.mk("opconst", tuple(), t, payload=name)

    # lower bindings in order
    for b in binds:
        types = infer_types_for_binding(tc, b.expr, declared=b.typ)
        # special-case input("col") => Node("input", payload=col)
        if isinstance(b.expr, EApp) and isinstance(b.expr.fn, EVar) and b.expr.fn.name == "input":
            col = b.expr.args[0].value  # ELitString
            n = dag.mk("input", tuple(), b.typ, payload=col)
        else:
            n = lower(b.expr, types, nodes, dag.mk)
            if not type_eq(n.typ, b.typ):
                n = dag.mk(n.op, n.args, b.typ, payload=n.payload)
        nodes[b.name] = n

    return binds, nodes, dag
