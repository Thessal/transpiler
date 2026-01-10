import pandas as pd
import signal
from typing import Dict, List
from butterflow import lex, Parser, TypeChecker, Builder, Runtime
import numpy as np
from glob import glob
import json
from butterflow.parser import Expr, Call, VarRef, Literal
import numpy as np


class Teacher:
    # Teacher is a function that generates score from data

    def score(self, input) -> Dict:
        # {"score_name":score_float}
        raise NotImplemented

    def test(self, input, scores=dict(), error_msg=[], prefix="", processor=lambda x: x, rules=[], fail_score=1., pass_score=5., na_score=None):
        try:
            if type(input) == type(None):
                score = na_score
            output = processor(input)
            score = pass_score
            for msg, s, rule in rules:
                if rule(output):
                    score = s
                    error_msg.append(msg)
                    break
        except Exception as e:
            output = None
            score = fail_score
            error_msg.append(str(e))

        scores[prefix] = score
        return output, scores, error_msg


class SyntaxTeacher(Teacher):

    def score(self, input_code: str) -> Dict:
        scores = dict()
        error_msg = []
        tokens, scores, error_msg = self.lex(input_code, scores, error_msg)
        ast, scores, error_msg = self.parse(tokens, scores, error_msg)
        ast, scores, error_msg = self.typechecking(ast, scores, error_msg)
        graph, scores, error_msg = self.build(ast, scores, error_msg)
        return graph, scores, "\n".join(error_msg)

    def lex(self, input_code, scores=dict(), error_msg=[]):
        prefix = "lex"
        processor = lex
        rules = [
            ("lex result empty", 2., lambda tokens: len(tokens) == 0)
        ]
        tokens, scores, error_msg = self.test(
            input_code, scores=scores, error_msg=error_msg, prefix=prefix, processor=processor, rules=rules)
        return tokens, scores, error_msg

    def parse(self, tokens, scores: Dict, error_msg: List):
        prefix = "parse"
        def processor(tokens): return Parser(tokens).parse()
        rules = []
        ast, scores, error_msg = self.test(
            tokens, scores=scores, error_msg=error_msg, prefix=prefix, processor=processor, rules=rules)
        return ast, scores, error_msg

    def typechecking(self, ast, scores: Dict, error_msg: List):
        prefix = "type_check"
        checker = TypeChecker()

        def processor(ast):
            checker.check(ast)
            return ast
        rules = []
        ast, scores, error_msg = self.test(
            ast, scores=scores, error_msg=error_msg, prefix=prefix, processor=processor, rules=rules)
        return ast, scores, error_msg

    def build(self, ast, scores: Dict, error_msg: List):
        # Build Graph
        prefix = "build"
        builder = Builder()
        processor = builder.build
        rules = [
            ("graph is not an expression", 1., lambda graph: not isinstance(graph, Expr)),
            ("expression is literal", 2., lambda graph: isinstance(graph, Literal)),
            ("result is referencing other variable", 4.9, lambda graph: isinstance(graph, VarRef)),
        ]
        graph, scores, error_msg = self.test(
            ast, scores=scores, error_msg=error_msg, prefix=prefix, processor=processor, rules=rules)
        # print(f"\nFinal Graph Result:\nresult = {graph}")
        return graph, scores, error_msg


class SemanticTeacher(Teacher):
    def __init__(self):
        # Initialize runtime
        runtime_data = {
            f'data("{x}")': np.load(f"data/{x}.npy")
            for x in ["open", "high", "low", "close", "volume"]}
        x_close = runtime_data['data("close")']
        x_close_d1 = np.roll(runtime_data['data("close")'], shift=1, axis=0)
        x_close_d1[0] = x_close[0]
        x_logret = np.log(x_close / x_close_d1)
        runtime_data['data("price")'] = x_close
        runtime_data['data("returns")'] = x_logret  # logret
        self.runtime = Runtime(data=runtime_data)
        raise NotImplementedError

    def score(self, graph:Expr ) -> Dict:
        scores = dict()
        # error_msg = []
        # tokens, scores, error_msg = self.lex(input_code, scores, error_msg)
        # ast, scores, error_msg = self.parse(tokens, scores, error_msg)
        # ast, scores, error_msg = self.typechecking(ast, scores, error_msg)
        # graph, scores, error_msg = self.build(ast, scores, error_msg)
        # return graph, scores, "\n".join(error_msg)
    

        # signal.alarm(10) # NOTE: causes truble when debugging
    # def calculate()
    # def data_shape(self):
    # def data_coverage(self):
        # position_input = compute(runtime, input_code)
        # position_raw, position = normalize_position(position_input, x_logret)
        # stat = calculate_stat(position_raw, position, x_logret)
    # def data_coverage_after_normalizaiton(self):
    # def data_skew(self):
#         "returns": np.nanmean(returns)*252,
#         "sharpe": np.nanmean(returns)/np.nanstd(returns)*np.sqrt(252),
#         "max_turnover": np.nanmin(np.nanmean(np.abs(np.diff(position)), axis=1)),
#         "mdd": np.min(np.cumsum(returns)),
#         "max_position": np.nanmax(np.abs(position))
    # def timeout


