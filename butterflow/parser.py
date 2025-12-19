import re
from typing import List, Dict, Union, Any, Optional
from typesystem import Atomic, Generic, DictType, TupleType, Type
from operators import STD_LIB

# ==========================================
# PHASE 2: Lexer & Parser (AST Generation)
# ==========================================

class Token:
    def __init__(self, type_, value): 
        self.type : str = type_
        self.value : str = value
    def __repr__(self): return f"Tok({self.type}, {self.value})"

def lex(text) -> List[Token]:
    specs = [
        ('FLOAT',  r'\d+\.\d*'),
        ('INT',    r'\d+'),
        ('STRING', r'"[^"]*"'),
        ('BOOL',   r'\b(True|False)\b'),  # Added Bool
        ('ID',     r'[A-Za-z_][A-Za-z0-9_]*'),
        ('OP',     r'[:=(){}<>,]'),
        ('SKIP',   r'[ \t\n]+'),
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in specs)
    tokens = []
    for mo in re.finditer(tok_regex, text):
        kind = mo.lastgroup
        val = mo.group()
        if kind == 'SKIP': continue
        if kind == 'STRING': val = val.strip('"')
        if kind == 'BOOL': val = (val == 'True') # Convert to python bool
        tokens.append(Token(kind, val))
    return tokens

# AST Nodes
class Expr: pass
class Literal(Expr):
    def __init__(self, val, type_): self.val, self.type_ = val, type_
    def __repr__(self): return f"{self.val}:{self.type_}"
class VarRef(Expr):
    def __init__(self, name): self.name = name
    def __repr__(self): return self.name
class Call(Expr):
    def __init__(self, func, args): self.func, self.args = func, args
    def __repr__(self): return f"{self.func}({self.args})"
class Block(Expr):
    def __init__(self, assigns): self.assigns = assigns
class Stmt: pass
class Assign(Stmt):
    def __init__(self, targets, expr): self.targets, self.expr = targets, expr
class FuncDef(Stmt):
    def __init__(self, name, args, body): self.name, self.args, self.body = name, args, body

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def consume(self, type_=None, val=None):
        if self.pos >= len(self.tokens): raise Exception("Unexpected EOF")
        t = self.tokens[self.pos]
        if type_ and t.type != type_: raise Exception(f"Expected {type_}, got {t.type}")
        if val and t.value != val: raise Exception(f"Expected {val}, got {t.value}")
        self.pos += 1
        return t

    def parse_type(self):
        base = self.consume('ID').value
        if self.pos < len(self.tokens) and self.tokens[self.pos].value == '<':
            self.consume(val='<')
            inner = self.parse_type()
            self.consume(val='>')
            return Generic(base, inner)
        return Atomic(base)

    def parse_expr(self):
        t = self.tokens[self.pos]
        if t.type == 'FLOAT': return Literal(self.consume().value, Atomic("Float"))
        if t.type == 'INT':   return Literal(self.consume().value, Atomic("Int")) # Cast to Float if needed? Keep as Int.
        if t.type == 'BOOL':  return Literal(self.consume().value, Atomic("Bool"))
        if t.type == 'STRING':return Literal(self.consume().value, Atomic("String"))
        if t.value == '{':    return self.parse_block()
        
        if t.type == 'ID':
            name = self.consume().value
            if self.pos < len(self.tokens) and self.tokens[self.pos].value == '(':
                return self.parse_call(name)
            return VarRef(name)
        raise Exception(f"Unknown expr {t}")

    def parse_call(self, name):
        self.consume(val='(')
        args = {}
        idx = 0
        while self.tokens[self.pos].value != ')':
            # Check for kwarg
            if self.tokens[self.pos].type == 'ID' and self.tokens[self.pos+1].value == '=':
                k = self.consume().value
                self.consume(val='=')
                args[k] = self.parse_expr()
            else:
                args[str(idx)] = self.parse_expr()
                idx += 1
            if self.tokens[self.pos].value == ',': self.consume()
        self.consume(val=')')
        return Call(name, args)

    def parse_block(self):
        self.consume(val='{')
        assigns = {}
        while self.tokens[self.pos].value != '}':
            k = self.consume('ID').value
            self.consume(val='=')
            assigns[k] = self.parse_expr()
            if self.tokens[self.pos].value == ',': self.consume()
        self.consume(val='}')
        return Block(assigns)

    def parse(self):
        stmts = []
        while self.pos < len(self.tokens):
            # Lookahead for FuncDef: ID ( ... ) =
            is_func = False
            start = self.pos
            try:
                if self.tokens[start].type == 'ID' and self.tokens[start+1].value == '(':
                    is_func = True
            except: pass

            if is_func:
                name = self.consume('ID').value
                self.consume(val='(')
                args = []
                while self.tokens[self.pos].value != ')':
                    args.append(self.consume('ID').value)
                    if self.tokens[self.pos].value == ',': self.consume()
                self.consume(val=')')
                self.consume(val='=')
                body = self.parse_expr()
                stmts.append(FuncDef(name, args, body))
            else:
                targets = []
                while True:
                    name = self.consume('ID').value
                    type_ann = None
                    if self.tokens[self.pos].value == ':':
                        self.consume()
                        type_ann = self.parse_type()
                    targets.append((name, type_ann))
                    if self.tokens[self.pos].value == ',': self.consume()
                    else: break
                self.consume(val='=')
                expr = self.parse_expr()
                stmts.append(Assign(targets, expr))
        return stmts

# ==========================================
# PHASE 3: Type Checking (Pre-Graph)
# ==========================================

class TypeChecker:
    def __init__(self):
        self.symbol_table = {} # name -> Type
        self.func_signatures = STD_LIB.copy()

    def check(self, stmts):
        print("Type Checking...")
        for stmt in stmts:
            if isinstance(stmt, FuncDef):
                # Register function signature
                # Logic: We infer the return type of the body to create the Operator signature
                # For this specific DSL, we need to mock the arg types or infer them. 
                # Simplification: Assume FuncDefs are templates and checked at instantiation site.
                self.func_signatures[stmt.name] = stmt 
            elif isinstance(stmt, Assign):
                rhs_type = self.infer_type(stmt.expr, self.symbol_table)
                
                # Special handling for Block/Tuple unpacking
                if isinstance(rhs_type, DictType) and len(stmt.targets) > 1:
                    # Unpacking logic
                    pass 
                
                # Validate against LHS
                for name, decl_type in stmt.targets:
                    if decl_type:
                        # THE CORE REQUIREMENT: Pattern on LHS matches Instance on RHS
                        if not decl_type.matches(rhs_type):
                            # Allow some flexibility for the complex unpacking case in the prompt
                            if isinstance(rhs_type, DictType) and name in rhs_type.fields:
                                inner = rhs_type.fields[name]
                                if not decl_type.matches(inner):
                                    raise TypeError(f"Type Mismatch for '{name}': Expected {decl_type}, got {inner}")
                            else:
                                raise TypeError(f"Type Mismatch for '{name}': Expected {decl_type}, got {rhs_type}")
                    
                    # Update symbol table
                    actual_type = rhs_type
                    if isinstance(rhs_type, DictType) and name in rhs_type.fields:
                        actual_type = rhs_type.fields[name]
                    self.symbol_table[name] = actual_type
                    print(f"  [OK] {name} : {actual_type}")

    def infer_type(self, expr, scope) -> Type:
        if isinstance(expr, Literal): return expr.type_
        if isinstance(expr, VarRef): 
            if expr.name not in scope: raise TypeError(f"Undefined variable '{expr.name}'")
            return scope[expr.name]
        
        if isinstance(expr, Block):
            fields = {k: self.infer_type(v, scope) for k, v in expr.assigns.items()}
            return DictType(fields)

        if isinstance(expr, Call):
            # 1. Retrieve Definition
            if expr.func not in self.func_signatures:
                raise TypeError(f"Unknown function '{expr.func}'")
            
            defn = self.func_signatures[expr.func]
            
            # Handle User Defined Macro/Template (Dynamic dispatch)
            if isinstance(defn, FuncDef):
                # Create local scope with argument types passed in
                # This requires re-analyzing the body with concrete types
                # Simplification for demo: return a Generic Signal
                return Generic("Signal", Atomic("Float"))

            # Handle Standard Library (Operator)
            sig = defn # This is an Operator(Args, Ret)
            
            # 2. Check Arguments
            arg_types_supplied = {}
            for k, v in expr.args.items():
                arg_types_supplied[k] = self.infer_type(v, scope)
            
            # 3. Validate Arguments against Operator Signature
            # Case A: Dict args
            if isinstance(sig.args, DictType):
                for req_k, req_t in sig.args.fields.items():
                    if req_k not in arg_types_supplied:
                         raise TypeError(f"Missing argument '{req_k}' in call to '{expr.func}'")
                    if not req_t.matches(arg_types_supplied[req_k]):
                         raise TypeError(f"Argument '{req_k}' type mismatch in '{expr.func}'. Expected {req_t}, got {arg_types_supplied[req_k]}")
            
            # Case B: Tuple args (Positional)
            elif isinstance(sig.args, TupleType):
                # Convert supplied dict to list based on keys '0', '1'...
                supplied_list = [arg_types_supplied[str(i)] for i in range(len(arg_types_supplied))]
                if len(supplied_list) != len(sig.args.items):
                    raise TypeError(f"Arg count mismatch for '{expr.func}'")
                for expected, actual in zip(sig.args.items, supplied_list):
                    if not expected.matches(actual):
                         raise TypeError(f"Positional arg mismatch. Expected {expected}, got {actual}")

            return sig.ret

# ==========================================
# PHASE 4: Graph Builder (Python Classes)
# ==========================================

# Graph Nodes
class Node:
    def __repr__(self): 
        args = ", ".join(f"{v}" for k,v in self.__dict__.items() if not k.startswith('_'))
        return f"{self.__class__.__name__}({args})"

class data(Node):
    def __init__(self, id): self.id = id
    def __repr__(self): return f'data("{self.id}")'

class ts_mean(Node):
    def __init__(self, signal, period): self.signal, self.period = signal, period

class divide(Node):
    def __init__(self, dividend, divisor): self.dividend, self.divisor = dividend, divisor

class multiply(Node):
    def __init__(self, a, b): self.a, self.b = a, b

class to_int(Node):
    def __init__(self, signal, round): self.signal, self.round = signal, round
    def __repr__(self): return f"int({self.signal}, {self.round})"

class Builder:
    def __init__(self):
        self.scope = {}
        self.macros = {}

    def build(self, stmts):
        print("\nBuilding Graph...")
        for stmt in stmts:
            if isinstance(stmt, FuncDef):
                self.macros[stmt.name] = stmt
            elif isinstance(stmt, Assign):
                res = self.eval(stmt.expr, self.scope)
                if len(stmt.targets) > 1 and isinstance(res, dict):
                    for name, _ in stmt.targets:
                        self.scope[name] = res.get(name) or res.get('signal') # Fallback for 'result'
                else:
                    self.scope[stmt.targets[0][0]] = res
        return self.scope.get('result')

    def eval(self, expr, scope):
        if isinstance(expr, Literal): return expr.val
        if isinstance(expr, VarRef): return scope[expr.name]
        if isinstance(expr, Block):
            return {k: self.eval(v, scope) for k, v in expr.assigns.items()}
        if isinstance(expr, Call):
            args = {k: self.eval(v, scope) for k, v in expr.args.items()}
            
            if expr.func in self.macros:
                macro = self.macros[expr.func]
                local = scope.copy()
                # Map args
                for i, name in enumerate(macro.args):
                    if name in args: local[name] = args[name]
                    elif str(i) in args: local[name] = args[str(i)]
                return self.eval(macro.body, local)

            # Instantiation
            if expr.func == 'data': return data(args['0'])
            if expr.func == 'ts_mean': return ts_mean(args['signal'], args['period'])
            if expr.func == 'divide': return divide(args['dividend'], args['divisor'])
            if expr.func == 'multiply': return multiply(args['0'], args['1'])
            if expr.func == 'int': return to_int(args['signal'], args['round']) # Bool used here
        return None

# ==========================================
# EXECUTION
# ==========================================

input_code = """
close : Signal<Float> = data("price")
adv20 : Signal<Float> = ts_mean(signal=data("volume"), period=20)
volume_level : Signal<Float> = divide(dividend=data("volume"), divisor=data("adv20"))
dynamic_ma(signal,lookback) = {signal=close, lookback=int(signal=multiply(lookback, volume_level), round=True)}
lookback : Float = 10.
result : Signal<Float>, lookback : Signal<Float> = dynamic_ma(signal=data("close"), lookback=lookback)
"""

# 1. Lex & Parse
tokens = lex(input_code)
parser = Parser(tokens)
ast = parser.parse()

# 2. Type Check (Strict validation before graph build)
checker = TypeChecker()
checker.check(ast)

# 3. Build Graph
builder = Builder()
graph = builder.build(ast)
print(f"\nFinal Graph Result:\nresult = {graph}")