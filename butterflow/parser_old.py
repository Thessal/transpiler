from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Sequence
import re
import numpy as np

# from types import 

# =========================
# Lexer
# =========================
@dataclass(frozen=True)
class Tok:
    kind: str
    text: str
    pos: int


TOKEN_SPEC = [
    ("WS",      r"[ \t\r\n]+"),
    ("ASSIGN",  r"="),
    ("COLON",   r":"),
    ("LT",      r"<"),
    ("GT",      r">"),
    ("LPAREN",  r"\("),
    ("RPAREN",  r"\)"),
    ("COMMA",   r","),
    ("OP",      r"[\+\-\*/\^]"),
    ("FLOAT",   r"[-]?\d+\.\d+"),
    ("INT",     r"[-]?\d+"),
    ("STRING",  r'"([^"\\]|\\.)*"'),
    ("IDENT",   r"[A-Za-z_][A-Za-z0-9_]*"),
]

TOKEN_RE = re.compile("|".join(f"(?P<{k}>{pat})" for k, pat in TOKEN_SPEC))


def lex(src: str) -> List[Tok]:
    out: List[Tok] = []
    i = 0
    while i < len(src):
        m = TOKEN_RE.match(src, i)
        if not m:
            raise SyntaxError(f"Unexpected character at {i}: {src[i:i+20]!r}")
        kind = m.lastgroup
        text = m.group()
        if kind != "WS":
            out.append(Tok(kind, text, i))
        i = m.end()
    out.append(Tok("EOF", "", len(src)))
    return out


# =========================
# AST
# =========================
class Expr: ...


@dataclass
class EVar(Expr):
    name: str


@dataclass
class ELitInt(Expr):
    value: int


@dataclass
class ELitFloat(Expr):
    value: float


@dataclass
class ELitString(Expr):
    value: str


@dataclass
class ETuple(Expr):
    items: List[Expr]


@dataclass
class EApp(Expr):
    fn: Expr
    args: List[Expr]  # supports chaining: f(a)(b) parses as EApp(EApp(f,[a]), [b])


@dataclass
class EBinOp(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass
class EUnary(Expr):
    op: str
    inner: Expr


@dataclass
class Binding:
    name: str
    typ: Type
    expr: Expr


# =========================
# Parser (postfix application)
# =========================
class Parser:
    def __init__(self, toks: List[Tok]):
        self.toks = toks
        self.i = 0

    def peek(self) -> Tok:
        return self.toks[self.i]

    def eat(self, kind: str) -> Tok:
        t = self.peek()
        if t.kind != kind:
            raise SyntaxError(f"Expected {kind} at {t.pos}, got {t.kind} ({t.text})")
        self.i += 1
        return t

    def accept(self, kind: str) -> Optional[Tok]:
        if self.peek().kind == kind:
            self.i += 1
            return self.toks[self.i - 1]
        return None

    def parse_program(self) -> List[Binding]:
        binds: List[Binding] = []
        while self.peek().kind != "EOF":
            binds.append(self.parse_binding())
        self.eat("EOF")
        return binds

    def parse_binding(self) -> Binding:
        name = self.eat("IDENT").text
        self.eat("COLON")
        typ = self.parse_type()
        self.eat("ASSIGN")
        expr = self.parse_term()
        return Binding(name, typ, expr)

    def parse_type(self) -> Type:
        t = self.eat("IDENT").text
        if t == "Signal":
            if self.accept("LT"):
                scalar = self.eat("IDENT").text
                self.eat("GT")
                return SignalType(ScalarType(scalar))
            return SignalType(REAL)
        if t == "IntParam":
            return IntParamType()
        if t == "FloatParam":
            return FloatParamType()
        if t == "Tuple":
            self.eat("LT")
            items = [self.parse_type()]
            while self.accept("COMMA"):
                items.append(self.parse_type())
            self.eat("GT")
            return TupleType(tuple(items))
        if t == "Op":
            self.eat("LT")
            in_t = self.parse_type()
            self.eat("COMMA")
            out_t = self.parse_type()
            self.eat("GT")
            return OpType(in_t, out_t)
        raise SyntaxError(f"Unknown type name: {t}")

    def parse_term(self) -> Expr:
        return self.parse_add()

    def parse_add(self) -> Expr:
        node = self.parse_mul()
        while self.peek().kind == "OP" and self.peek().text in {"+", "-"}:
            op = self.eat("OP").text
            rhs = self.parse_mul()
            node = EBinOp(op, node, rhs)
        return node

    def parse_mul(self) -> Expr:
        node = self.parse_pow()
        while self.peek().kind == "OP" and self.peek().text in {"*", "/"}:
            op = self.eat("OP").text
            rhs = self.parse_pow()
            node = EBinOp(op, node, rhs)
        return node

    def parse_pow(self) -> Expr:
        node = self.parse_unary()
        while self.peek().kind == "OP" and self.peek().text == "^":
            op = self.eat("OP").text
            rhs = self.parse_unary()
            node = EBinOp(op, node, rhs)
        return node

    def parse_unary(self) -> Expr:
        if self.peek().kind == "OP" and self.peek().text in {"+", "-"}:
            op = self.eat("OP").text
            inner = self.parse_unary()
            return EUnary(op, inner)
        return self.parse_postfix()

    def parse_postfix(self) -> Expr:
        node = self.parse_primary()
        while self.accept("LPAREN"):
            args: List[Expr] = []
            if self.peek().kind != "RPAREN":
                args.append(self.parse_term())
                while self.accept("COMMA"):
                    args.append(self.parse_term())
            self.eat("RPAREN")
            node = EApp(node, args)
        return node

    def parse_primary(self) -> Expr:
        t = self.peek()
        if t.kind == "INT":
            self.eat("INT")
            return ELitInt(int(t.text))
        if t.kind == "FLOAT":
            self.eat("FLOAT")
            return ELitFloat(float(t.text))
        if t.kind == "STRING":
            self.eat("STRING")
            raw = t.text[1:-1]
            return ELitString(bytes(raw, "utf-8").decode("unicode_escape"))
        if t.kind == "IDENT":
            return EVar(self.eat("IDENT").text)
        if self.accept("LPAREN"):
            first = self.parse_term()
            if self.accept("COMMA"):
                items = [first, self.parse_term()]
                while self.accept("COMMA"):
                    items.append(self.parse_term())
                self.eat("RPAREN")
                return ETuple(items)
            self.eat("RPAREN")
            return first
        raise SyntaxError(f"Unexpected token {t.kind}({t.text}) at {t.pos}")


def parse(src: str) -> List[Binding]:
    return Parser(lex(src)).parse_program()


# =========================
# Builtins (monomorphic demonstration)
# =========================
SIGR = SignalType(REAL)
OP_SIGSIG = OpType(SIGR, SIGR)

BUILTIN_TYPES: Dict[str, Type] = {
    "delay": OpType(TupleType((IntParamType(), SIGR)), SIGR),
    "rolling_mean": OpType(TupleType((IntParamType(), SIGR)), SIGR),
    "cs_rank": OpType(SIGR, SIGR),

    # higher-order: tuple of operators -> operator
    "compose": OpType(TupleType((OP_SIGSIG, OP_SIGSIG)), OP_SIGSIG),

    # special-form; typed via binding annotation
    "input": OpType(FloatParamType(), SIGR),  # placeholder
}
RESERVED = set(BUILTIN_TYPES.keys())


# =========================
# Type checker with currying
# =========================
class TypeChecker:
    def __init__(self, env_types: Dict[str, Type]):
        self.env_types = env_types

    def infer(self, e: Expr, binding_declared: Optional[Type] = None) -> Type:
        if isinstance(e, EVar):
            if e.name not in self.env_types:
                raise TypeError(f"Unknown variable: {e.name}")
            return self.env_types[e.name]

        if isinstance(e, ELitInt):
            return IntParamType()
        if isinstance(e, ELitFloat):
            return FloatParamType()
        if isinstance(e, ELitString):
            # only legal as input("col") argument; kept as placeholder type
            return FloatParamType()

        if isinstance(e, ETuple):
            return TupleType(tuple(self.infer(x, binding_declared) for x in e.items))

        if isinstance(e, EUnary):
            t = self.infer(e.inner, binding_declared)
            if e.op in {"+", "-"} and (is_param(t) or is_signal(t)):
                return t
            raise TypeError(f"Unary {e.op} not supported for {t}")

        if isinstance(e, EBinOp):
            lt = self.infer(e.left, binding_declared)
            rt = self.infer(e.right, binding_declared)
            if e.op in {"+", "-", "*", "/", "^"}:
                if is_signal(lt) or is_signal(rt):
                    return SIGR
                if is_param(lt) and is_param(rt):
                    return promote_param(lt, rt)
            raise TypeError(f"Unsupported binop {e.op} for {lt},{rt}")

        if isinstance(e, EApp):
            # input("col") special-form: output type is binding annotation
            if isinstance(e.fn, EVar) and e.fn.name == "input":
                if len(e.args) != 1 or not isinstance(e.args[0], ELitString):
                    raise TypeError('input("col") requires one string literal.')
                return binding_declared if (binding_declared and is_signal(binding_declared)) else SIGR

            fn_t = self.infer(e.fn, binding_declared)
            if not isinstance(fn_t, OpType):
                raise TypeError(f"Attempted to apply non-operator type: {fn_t}")

            arg_ts = [self.infer(a, binding_declared) for a in e.args]
            expected = list(tuple_items(fn_t.in_type))

            # tuple-as-single-arg: f((a,b)) where f expects Tuple[a,b]
            if len(arg_ts) == 1 and type_eq(arg_ts[0], fn_t.in_type):
                return fn_t.out_type

            if len(arg_ts) > len(expected):
                raise TypeError(f"Too many args: expected {len(expected)}, got {len(arg_ts)}")

            for i, at in enumerate(arg_ts):
                if not type_eq(at, expected[i]):
                    raise TypeError(f"Arg {i} mismatch: expected {expected[i]}, got {at}")

            if len(arg_ts) == len(expected):
                return fn_t.out_type

            rem_in = mk_remaining_input(arg_ts, expected)
            return OpType(rem_in, fn_t.out_type)

        raise TypeError(f"Unhandled expr: {e}")


def infer_types_for_binding(tc: TypeChecker, expr: Expr, declared: Type) -> Dict[int, Type]:
    out: Dict[int, Type] = {}

    def walk(e: Expr) -> None:
        out[id(e)] = tc.infer(e, binding_declared=declared)
        if isinstance(e, EUnary):
            walk(e.inner)
        elif isinstance(e, EBinOp):
            walk(e.left); walk(e.right)
        elif isinstance(e, ETuple):
            for x in e.items:
                walk(x)
        elif isinstance(e, EApp):
            walk(e.fn)
            for a in e.args:
                walk(a)

    walk(expr)
    return out




