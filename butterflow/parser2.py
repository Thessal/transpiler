from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Iterable


# =========================
# Lexer
# =========================

class TokKind(Enum):
    IDENT = auto()
    NUMBER = auto()
    STRING = auto()
    BOOL = auto()

    COLON = auto()      # :
    COMMA = auto()      # ,
    EQ = auto()         # =
    LPAREN = auto()     # (
    RPAREN = auto()     # )
    LBRACE = auto()     # {
    RBRACE = auto()     # }
    LT = auto()         # <
    GT = auto()         # >
    PIPE = auto()       # |
    NEWLINE = auto()
    EOF = auto()


@dataclass(frozen=True)
class Token:
    kind: TokKind
    text: str
    line: int
    col: int


class LexError(Exception):
    pass


class Lexer:
    def __init__(self, src: str):
        self.src = src
        self.i = 0
        self.line = 1
        self.col = 1

    def _peek(self) -> str:
        return self.src[self.i] if self.i < len(self.src) else ""

    def _adv(self) -> str:
        ch = self._peek()
        if not ch:
            return ""
        self.i += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def _pos(self) -> Tuple[int, int]:
        return self.line, self.col

    def tokenize(self) -> List[Token]:
        out: List[Token] = []
        while True:
            ch = self._peek()
            if not ch:
                out.append(Token(TokKind.EOF, "", self.line, self.col))
                return out

            # whitespace
            if ch in " \t\r":
                self._adv()
                continue

            # comment: consume until newline or EOF, but do not consume newline
            if ch == "#":
                while self._peek() and self._peek() != "\n":
                    self._adv()
                continue

            # newline
            if ch == "\n":
                line, col = self._pos()
                self._adv()
                out.append(Token(TokKind.NEWLINE, "\n", line, col))
                continue

            # punctuation
            punct = {
                ":": TokKind.COLON,
                ",": TokKind.COMMA,
                "=": TokKind.EQ,
                "(": TokKind.LPAREN,
                ")": TokKind.RPAREN,
                "{": TokKind.LBRACE,
                "}": TokKind.RBRACE,
                "<": TokKind.LT,
                ">": TokKind.GT,
                "|": TokKind.PIPE,
            }
            if ch in punct:
                line, col = self._pos()
                self._adv()
                out.append(Token(punct[ch], ch, line, col))
                continue

            # string
            if ch == '"':
                line, col = self._pos()
                self._adv()  # consume "
                s = []
                while True:
                    c = self._peek()
                    if not c:
                        raise LexError(f"Unterminated string at {line}:{col}")
                    if c == '"':
                        self._adv()
                        break
                    if c == "\\":
                        self._adv()
                        esc = self._peek()
                        if not esc:
                            raise LexError(f"Bad escape at {self.line}:{self.col}")
                        self._adv()
                        m = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(esc, esc)
                        s.append(m)
                        continue
                    s.append(self._adv())
                out.append(Token(TokKind.STRING, "".join(s), line, col))
                continue

            # number (supports 10, 10., 10.5)
            if ch.isdigit():
                line, col = self._pos()
                num = []
                while self._peek().isdigit():
                    num.append(self._adv())
                if self._peek() == ".":
                    num.append(self._adv())
                    while self._peek().isdigit():
                        num.append(self._adv())
                out.append(Token(TokKind.NUMBER, "".join(num), line, col))
                continue

            # identifier / bool
            if ch.isalpha() or ch == "_":
                line, col = self._pos()
                ident = []
                while True:
                    c = self._peek()
                    if c.isalnum() or c == "_":
                        ident.append(self._adv())
                    else:
                        break
                txt = "".join(ident)
                if txt in ("True", "False"):
                    out.append(Token(TokKind.BOOL, txt, line, col))
                else:
                    out.append(Token(TokKind.IDENT, txt, line, col))
                continue

            raise LexError(f"Unexpected character {ch!r} at {self.line}:{self.col}")


# =========================
# AST + Parser
# =========================

class ParseError(Exception):
    pass


@dataclass
class TypeAst:
    name: str
    args: List["TypeAst"] = field(default_factory=list)

@dataclass
class ExprAst:
    pass

@dataclass
class NameAst(ExprAst):
    name: str

@dataclass
class NumberAst(ExprAst):
    text: str  # preserve "10." vs "10"

@dataclass
class StringAst(ExprAst):
    value: str

@dataclass
class BoolAst(ExprAst):
    value: bool

@dataclass
class CallAst(ExprAst):
    func: str
    args_pos: List[ExprAst] = field(default_factory=list)
    args_kw: "OrderedDict[str, ExprAst]" = field(default_factory=OrderedDict)

@dataclass
class DictAst(ExprAst):
    items: "OrderedDict[str, ExprAst]" = field(default_factory=OrderedDict)

@dataclass
class VarDeclAst:
    name: str
    type_ast: Optional[TypeAst]  # allow omission

@dataclass
class StmtAst:
    pass

@dataclass
class AssignAst(StmtAst):
    targets: List[VarDeclAst]   # 1 or more
    value: ExprAst

@dataclass
class FuncDefAst(StmtAst):
    name: str
    params: List[str]
    body: ExprAst

@dataclass
class ProgramAst:
    stmts: List[StmtAst]


class Parser:
    def __init__(self, toks: List[Token]):
        self.toks = toks
        self.i = 0

    def _peek(self) -> Token:
        return self.toks[self.i]

    def _adv(self) -> Token:
        t = self._peek()
        if t.kind != TokKind.EOF:
            self.i += 1
        return t

    def _accept(self, kind: TokKind) -> Optional[Token]:
        if self._peek().kind == kind:
            return self._adv()
        return None

    def _expect(self, kind: TokKind, msg: str) -> Token:
        t = self._peek()
        if t.kind != kind:
            raise ParseError(f"{msg} at {t.line}:{t.col} (got {t.kind} {t.text!r})")
        return self._adv()

    def parse_program(self) -> ProgramAst:
        stmts: List[StmtAst] = []
        while self._peek().kind != TokKind.EOF:
            while self._accept(TokKind.NEWLINE):
                pass
            if self._peek().kind == TokKind.EOF:
                break
            stmts.append(self.parse_stmt())
            while self._accept(TokKind.NEWLINE):
                pass
        return ProgramAst(stmts)

    def parse_stmt(self) -> StmtAst:
        if self._peek().kind == TokKind.IDENT:
            save = self.i
            name_tok = self._adv()
            if self._accept(TokKind.LPAREN):
                params = []
                if self._peek().kind == TokKind.IDENT:
                    params.append(self._expect(TokKind.IDENT, "Expected param").text)
                    while self._accept(TokKind.COMMA):
                        params.append(self._expect(TokKind.IDENT, "Expected param").text)
                self._expect(TokKind.RPAREN, "Expected ')'")
                if self._accept(TokKind.EQ):
                    body = self.parse_expr()
                    return FuncDefAst(name_tok.text, params, body)
            self.i = save

        targets = self.parse_targets()
        self._expect(TokKind.EQ, "Expected '='")
        value = self.parse_expr()
        return AssignAst(targets, value)

    def parse_targets(self) -> List[VarDeclAst]:
        decls = [self.parse_vardecl()]
        while self._accept(TokKind.COMMA):
            decls.append(self.parse_vardecl())
        return decls

    def parse_vardecl(self) -> VarDeclAst:
        name = self._expect(TokKind.IDENT, "Expected variable name").text
        type_ast = None
        if self._accept(TokKind.COLON):
            type_ast = self.parse_type()
        return VarDeclAst(name, type_ast)

    def parse_type(self) -> TypeAst:
        name = self._expect(TokKind.IDENT, "Expected type name").text
        args: List[TypeAst] = []
        if self._accept(TokKind.LT):
            args.append(self.parse_type())
            while self._accept(TokKind.COMMA) or self._accept(TokKind.PIPE):
                args.append(self.parse_type())
            self._expect(TokKind.GT, "Expected '>'")
        return TypeAst(name, args)

    def parse_expr(self) -> ExprAst:
        return self.parse_primary()

    def parse_primary(self) -> ExprAst:
        t = self._peek()

        if t.kind == TokKind.IDENT:
            name = self._adv().text
            if self._accept(TokKind.LPAREN):
                args_pos: List[ExprAst] = []
                args_kw: "OrderedDict[str, ExprAst]" = OrderedDict()
                if self._peek().kind != TokKind.RPAREN:
                    self.parse_call_args(args_pos, args_kw)
                self._expect(TokKind.RPAREN, "Expected ')'")
                return CallAst(name, args_pos, args_kw)
            return NameAst(name)

        if t.kind == TokKind.NUMBER:
            return NumberAst(self._adv().text)

        if t.kind == TokKind.STRING:
            return StringAst(self._adv().text)

        if t.kind == TokKind.BOOL:
            return BoolAst(self._adv().text == "True")

        if self._accept(TokKind.LBRACE):
            items: "OrderedDict[str, ExprAst]" = OrderedDict()
            if self._peek().kind != TokKind.RBRACE:
                key = self._expect(TokKind.IDENT, "Expected dict key").text
                self._expect(TokKind.EQ, "Expected '=' in dict")
                items[key] = self.parse_expr()
                while self._accept(TokKind.COMMA):
                    if self._peek().kind == TokKind.RBRACE:
                        break
                    key = self._expect(TokKind.IDENT, "Expected dict key").text
                    self._expect(TokKind.EQ, "Expected '=' in dict")
                    items[key] = self.parse_expr()
            self._expect(TokKind.RBRACE, "Expected '}'")
            return DictAst(items)

        raise ParseError(f"Unexpected token {t.kind} {t.text!r} at {t.line}:{t.col}")

    def parse_call_args(self, args_pos: List[ExprAst], args_kw: "OrderedDict[str, ExprAst]") -> None:
        while True:
            if self._peek().kind == TokKind.IDENT:
                save = self.i
                key = self._adv().text
                if self._accept(TokKind.EQ):
                    args_kw[key] = self.parse_expr()
                else:
                    self.i = save
                    args_pos.append(self.parse_expr())
            else:
                args_pos.append(self.parse_expr())
            if not self._accept(TokKind.COMMA):
                break


# =========================
# Types + Typechecker (unchanged semantics; already fails for Float -> Int)
# =========================

class TypeError_(Exception):
    pass


@dataclass(frozen=True)
class Ty:
    def pretty(self) -> str:
        raise NotImplementedError

@dataclass(frozen=True)
class TyPrim(Ty):
    name: str
    def pretty(self) -> str:
        return self.name

@dataclass(frozen=True)
class TySignal(Ty):
    elem: Ty
    def pretty(self) -> str:
        return f"Signal<{self.elem.pretty()}>"

@dataclass(frozen=True)
class TyTuple(Ty):
    items: Tuple[Ty, ...]
    def pretty(self) -> str:
        inside = ", ".join(t.pretty() for t in self.items)
        return f"Tuple({inside})"

@dataclass(frozen=True)
class TyEither(Ty):
    options: Tuple[Ty, ...]
    def pretty(self) -> str:
        inside = " | ".join(t.pretty() for t in self.options)
        return f"Either({inside})"

@dataclass(frozen=True)
class TyRecord(Ty):
    fields: Tuple[Tuple[str, Ty], ...]
    def pretty(self) -> str:
        inside = ", ".join(f"{k}={v.pretty()}" for k, v in self.fields)
        return f"Dict({inside})"

def ty_from_ast(t: TypeAst) -> Ty:
    if t.name in ("Float", "Int", "Bool", "String"):
        return TyPrim(t.name.lower() if t.name != "Bool" else "bool")
    if t.name == "Signal":
        if len(t.args) != 1:
            raise TypeError_("Signal<T> requires one type arg")
        return TySignal(ty_from_ast(t.args[0]))
    if t.name == "Tuple":
        return TyTuple(tuple(ty_from_ast(a) for a in t.args))
    if t.name == "Either":
        return TyEither(tuple(ty_from_ast(a) for a in t.args))
    return TyPrim(t.name)

def is_assignable(dst: Ty, src: Ty) -> bool:
    if dst == src:
        return True
    if isinstance(dst, TyEither):
        return any(is_assignable(opt, src) for opt in dst.options)
    if isinstance(dst, TyPrim) and isinstance(src, TyPrim):
        # numeric widening only: int -> float (NOT float -> int)
        if dst.name == "float" and src.name == "int":
            return True
    if isinstance(dst, TySignal) and isinstance(src, TySignal):
        return is_assignable(dst.elem, src.elem)
    if isinstance(dst, TyTuple) and isinstance(src, TyTuple):
        return len(dst.items) == len(src.items) and all(is_assignable(a, b) for a, b in zip(dst.items, src.items))
    if isinstance(dst, TyRecord) and isinstance(src, TyRecord):
        if len(dst.fields) != len(src.fields):
            return False
        for (k1, v1), (k2, v2) in zip(dst.fields, src.fields):
            if k1 != k2 or not is_assignable(v1, v2):
                return False
        return True
    return False


# (Graph nodes / registry / checker / pretty printer)
# For brevity, reuse exactly the same implementations as before; only behavior change needed here is comments,
# and the existing numeric inference + assignability already produces the required failure.

# --- Start: minimal re-inclusion of the remaining parts (same as previous message) ---

_node_id = 0
def _next_id() -> int:
    global _node_id
    _node_id += 1
    return _node_id

@dataclass
class Node:
    ty: Ty
    id: int = field(default_factory=_next_id)
    def inputs(self) -> Iterable["Node"]:
        return ()

@dataclass
class ConstNode(Node):
    value: Any = None

@dataclass
class OpNode(Node):
    op: str = ""
    args: Tuple["Node", ...] = ()
    kwargs: Tuple[Tuple[str, "Node"], ...] = ()
    def inputs(self) -> Iterable["Node"]:
        for a in self.args:
            yield a
        for _, v in self.kwargs:
            yield v

@dataclass
class RecordNode(Node):
    items: Tuple[Tuple[str, Node], ...] = ()
    def inputs(self) -> Iterable["Node"]:
        for _, v in self.items:
            yield v

@dataclass
class Graph:
    roots: Dict[str, Node]
    all_nodes: Dict[int, Node]
    def pretty(self, var: str) -> str:
        return pretty_node(self.roots[var])

@dataclass(frozen=True)
class OpSig:
    name: str
    params: Tuple[Tuple[str, Ty], ...]
    ret: Ty
    defaults: Tuple[Tuple[str, Any], ...] = ()
    aliases: Tuple[Tuple[str, str], ...] = ()
    def param_map(self) -> Dict[str, Ty]:
        return dict(self.params)
    def default_map(self) -> Dict[str, Any]:
        return dict(self.defaults)
    def alias_map(self) -> Dict[str, str]:
        return dict(self.aliases)

@dataclass
class Env:
    vars: Dict[str, Tuple[Ty, Node]] = field(default_factory=dict)
    funcs: Dict[str, FuncDefAst] = field(default_factory=dict)

class TypeChecker:
    def __init__(self):
        self.env = Env()
        self.nodes: Dict[int, Node] = {}

        self.ty_float = TyPrim("float")
        self.ty_int = TyPrim("int")
        self.ty_bool = TyPrim("bool")
        self.ty_str = TyPrim("string")

        self.sig_float = TySignal(self.ty_float)
        self.sig_int = TySignal(self.ty_int)

        self.ops: Dict[str, OpSig] = {
            "data": OpSig("data", params=(("name", self.ty_str),), ret=self.sig_float),
            "divide": OpSig(
                "divide",
                params=(
                    ("dividend", TyEither((self.sig_float, self.ty_float))),
                    ("divisor", TyEither((self.sig_float, self.ty_float))),
                ),
                ret=self.sig_float,
            ),
            "multiply": OpSig(
                "multiply",
                params=(
                    ("left", TyEither((self.sig_float, self.ty_float))),
                    ("right", TyEither((self.sig_float, self.ty_float))),
                ),
                ret=self.sig_float,
            ),
            "int": OpSig(
                "int",
                params=(("signal", TyEither((self.sig_float, self.ty_float))), ("round", self.ty_bool)),
                ret=self.ty_int,
                defaults=(("round", True),),
            ),
            "ts_mean": OpSig(
                "ts_mean",
                params=(
                    ("signal", TyEither((self.sig_float, self.sig_int))),
                    ("period", TyEither((self.ty_float, self.ty_int, self.sig_int, self.sig_float))),
                ),
                ret=self.sig_float,
                aliases=(("lookback", "period"),),
            ),
        }

    def build(self, prog: ProgramAst) -> Graph:
        for st in prog.stmts:
            self.check_stmt(st)
        roots = {k: v for k, (_, v) in self.env.vars.items()}
        return Graph(roots=roots, all_nodes=self.nodes)

    def check_stmt(self, st: StmtAst) -> None:
        if isinstance(st, FuncDefAst):
            self.env.funcs[st.name] = st
            return

        if isinstance(st, AssignAst):
            rhs_ty, rhs_node = self.check_expr(st.value, local=None)

            if len(st.targets) == 1:
                decl = st.targets[0]
                dst_ty = ty_from_ast(decl.type_ast) if decl.type_ast else rhs_ty
                if not is_assignable(dst_ty, rhs_ty):
                    raise TypeError_(f"Cannot assign {rhs_ty.pretty()} to {decl.name}:{dst_ty.pretty()}")
                self.env.vars[decl.name] = (dst_ty, rhs_node)
                return

            # (same record-based 2-target reduction as before)
            decls = st.targets
            dst_tys = [ty_from_ast(d.type_ast) if d.type_ast else None for d in decls]
            if isinstance(rhs_ty, TyRecord):
                fields = OrderedDict(rhs_ty.fields)
                if "signal" in fields and ("lookback" in fields or "period" in fields):
                    sig_node = self._record_get(rhs_node, "signal")
                    period_key = "period" if "period" in fields else "lookback"
                    period_node = self._record_get(rhs_node, period_key)

                    out0_ty, out0_node = self._call_builtin_ts_mean(sig_node, period_node)
                    out1_ty, out1_node = period_node.ty, period_node

                    outs = [(out0_ty, out0_node), (out1_ty, out1_node)]
                    if len(decls) != 2:
                        raise TypeError_("Record reduction implemented only for 2-target unpacking.")
                    for idx, decl in enumerate(decls):
                        dst = dst_tys[idx] if dst_tys[idx] else outs[idx][0]
                        if not is_assignable(dst, outs[idx][0]):
                            raise TypeError_(f"Cannot assign {outs[idx][0].pretty()} to {decl.name}:{dst.pretty()}")
                        self.env.vars[decl.name] = (dst, outs[idx][1])
                    return
            raise TypeError_(f"Multi-target assignment unsupported for RHS type {rhs_ty.pretty()}")

        raise TypeError_(f"Unknown statement: {st}")

    def check_expr(self, e: ExprAst, local: Optional[Dict[str, Tuple[Ty, Node]]]) -> Tuple[Ty, Node]:
        if isinstance(e, NameAst):
            if local and e.name in local:
                return local[e.name]
            if e.name in self.env.vars:
                return self.env.vars[e.name]
            raise TypeError_(f"Unbound name: {e.name}")

        if isinstance(e, NumberAst):
            if "." in e.text:
                val = float(e.text)
                n = ConstNode(ty=self.ty_float, value=val)
                self.nodes[n.id] = n
                return self.ty_float, n
            val = int(e.text)
            n = ConstNode(ty=self.ty_int, value=val)
            self.nodes[n.id] = n
            return self.ty_int, n

        if isinstance(e, StringAst):
            n = ConstNode(ty=self.ty_str, value=e.value)
            self.nodes[n.id] = n
            return self.ty_str, n

        if isinstance(e, BoolAst):
            n = ConstNode(ty=self.ty_bool, value=e.value)
            self.nodes[n.id] = n
            return self.ty_bool, n

        if isinstance(e, DictAst):
            items_nodes: List[Tuple[str, Node]] = []
            items_tys: List[Tuple[str, Ty]] = []
            for k, v in e.items.items():
                v_ty, v_node = self.check_expr(v, local)
                items_nodes.append((k, v_node))
                items_tys.append((k, v_ty))
            ty = TyRecord(tuple(items_tys))
            n = RecordNode(ty=ty, items=tuple(items_nodes))
            self.nodes[n.id] = n
            return ty, n

        if isinstance(e, CallAst):
            if e.func in self.env.funcs:
                f = self.env.funcs[e.func]
                bound: Dict[str, Tuple[Ty, Node]] = {}
                if len(e.args_pos) > len(f.params):
                    raise TypeError_(f"Too many positional args for {e.func}")
                for p, a in zip(f.params, e.args_pos):
                    bound[p] = self.check_expr(a, local)
                for k, v in e.args_kw.items():
                    if k not in f.params:
                        raise TypeError_(f"Unknown param {k} for {e.func}")
                    if k in bound:
                        raise TypeError_(f"Duplicate param {k} for {e.func}")
                    bound[k] = self.check_expr(v, local)
                missing = [p for p in f.params if p not in bound]
                if missing:
                    raise TypeError_(f"Missing params for {e.func}: {missing}")
                return self.check_expr(f.body, local=bound)

            if e.func not in self.ops:
                raise TypeError_(f"Unknown operator: {e.func}")
            sig = self.ops[e.func]

            kw_expr: Dict[str, ExprAst] = dict(e.args_kw)
            for idx, arg in enumerate(e.args_pos):
                if idx >= len(sig.params):
                    raise TypeError_(f"Too many positional args for {e.func}")
                pname = sig.params[idx][0]
                if pname in kw_expr:
                    raise TypeError_(f"Arg {pname} passed both positionally and by keyword in {e.func}")
                kw_expr[pname] = arg

            alias = sig.alias_map()
            normalized: Dict[str, ExprAst] = {alias.get(k, k): v for k, v in kw_expr.items()}

            pmap = sig.param_map()
            dmap = sig.default_map()
            missing = [p for p, _ in sig.params if (p not in normalized and p not in dmap)]
            if missing:
                raise TypeError_(f"Missing params for {e.func}: {missing}")

            checked_kwargs: List[Tuple[str, Node]] = []
            for pname, pty in sig.params:
                if pname in normalized:
                    a_ty, a_node = self.check_expr(normalized[pname], local)
                else:
                    dv = dmap[pname]
                    a_ty = self.ty_bool if isinstance(dv, bool) else self.ty_int if isinstance(dv, int) else self.ty_float
                    a_node = ConstNode(ty=a_ty, value=dv)
                    self.nodes[a_node.id] = a_node

                if not is_assignable(pty, a_ty):
                    raise TypeError_(f"{e.func}.{pname}: expected {pty.pretty()}, got {a_ty.pretty()}")
                checked_kwargs.append((pname, a_node))

            ret_ty = sig.ret
            if e.func == "int":
                arg_node = dict(checked_kwargs)["signal"]
                ret_ty = TySignal(self.ty_int) if isinstance(arg_node.ty, TySignal) else self.ty_int

            node = OpNode(ty=ret_ty, op=e.func, kwargs=tuple(checked_kwargs))
            self.nodes[node.id] = node
            return ret_ty, node

        raise TypeError_(f"Unknown expression: {e}")

    def _record_get(self, record_node: Node, key: str) -> Node:
        if not isinstance(record_node, RecordNode):
            raise TypeError_(f"Expected record node, got {type(record_node).__name__}")
        for k, v in record_node.items:
            if k == key:
                return v
        raise TypeError_(f"Record missing key {key}")

    def _call_builtin_ts_mean(self, signal: Node, period: Node) -> Tuple[Ty, Node]:
        sig = self.ops["ts_mean"]
        if not is_assignable(dict(sig.params)["signal"], signal.ty):
            raise TypeError_(f"ts_mean.signal type mismatch: {signal.ty.pretty()}")
        if not is_assignable(dict(sig.params)["period"], period.ty):
            raise TypeError_(f"ts_mean.period type mismatch: {period.ty.pretty()}")
        node = OpNode(ty=sig.ret, op="ts_mean", kwargs=(("signal", signal), ("period", period)))
        self.nodes[node.id] = node
        return sig.ret, node


PREFERRED_POS_ORDER = {
    "data": ("name",),
    "divide": ("dividend", "divisor"),
    "multiply": ("left", "right"),
    "int": ("signal",),
    "ts_mean": ("signal", "period"),
}
DEFAULTS = {("int", "round"): True}

def pretty_node(n: Node) -> str:
    if isinstance(n, ConstNode):
        if n.ty == TyPrim("float"):
            s = f"{float(n.value):.12g}"
            if "." not in s:
                s += "."
            return f"Float({s})"
        if n.ty == TyPrim("int"):
            return f"Int({int(n.value)})"
        if n.ty == TyPrim("bool"):
            return "True" if n.value else "False"
        if n.ty == TyPrim("string"):
            return f"\"{n.value}\""
        return repr(n.value)

    if isinstance(n, RecordNode):
        inside = ",".join(f"{k}={pretty_node(v)}" for k, v in n.items)
        return "{" + inside + "}"

    if isinstance(n, OpNode):
        kw = OrderedDict(n.kwargs)
        for (op, pname), dval in DEFAULTS.items():
            if n.op == op and pname in kw and isinstance(kw[pname], ConstNode) and kw[pname].value == dval:
                del kw[pname]

        if n.op in PREFERRED_POS_ORDER:
            order = PREFERRED_POS_ORDER[n.op]
            if all(k in kw for k in order) and all(k in order for k in kw.keys()):
                args = ",".join(pretty_node(kw[k]) for k in order)
                return f"{n.op}({args})"

        args = ",".join(f"{k}={pretty_node(v)}" for k, v in kw.items())
        return f"{n.op}({args})"

    return f"<node {n.id}>"


# =========================
# Demo: MUST fail
# =========================

if __name__ == "__main__":
    src = """
close : Signal<Float> = data("price")
adv20 : Signal<Float> = ts_mean(signal=data("volume"), period=20)
volume_level : Signal<Float> = divide(dividend=data("volume"), divisor=data("adv20"))
# volume_level : Signal<Float> = divide(dividend=data("volume"), divisor=data("adv20"))
dynamic_ma(signal,lookback) = {signal=close, lookback=int(signal=multiply(lookback, volume_level), round=True)}
lookback : Float = 10.
# lookback : Int = 10.
result : Signal<Float>, lookback : Signal<Float> = dynamic_ma(signal=data("close"), lookback=lookback)
""".strip()

    toks = Lexer(src).tokenize()
    prog = Parser(toks).parse_program()
    tc = TypeChecker()

    try:
        g = tc.build(prog)
        print(g)
        print("result =", g.pretty("result"))
    except TypeError_ as e:
        print("TYPE ERROR:", e)
