import pandas as pd
import signal
from typing import Any, Callable, Dict, List, Tuple
from butterflow import lex, Parser, TypeChecker, Builder, Runtime
import numpy as np
from glob import glob
import json
# from butterflow.parser import Expr, Call, VarRef, Literal
from butterflow.operators import Node
import numpy as np
import sys
from morpho.backtest import Backtester


def set_timeout(seconds=10):
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        signal.alarm(seconds)


class Teacher:
    # Teacher is a function that generates score from data

    def score(self, input) -> Dict:
        # {"score_name":score_float}
        raise NotImplemented

    def test(self, input, scores=dict(), error_msg=[], prefix="", processor=lambda x: x, rules=[], fail_score=1., pass_score=5., na_score=None):
        try:
            if type(input) == type(None):
                output = None
                score = na_score
            else:
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

    def score(self, input_code: str) -> Tuple[Any, Dict, str]:
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
            ("root node of computation graph is not a Node",
             1., lambda graph: not isinstance(graph, Node)),
            # ("expression is literal", 2., lambda graph: isinstance(graph, Literal)),
            # ("result is referencing other variable", 4.9, lambda graph: isinstance(graph, VarRef)),
        ]
        graph, scores, error_msg = self.test(
            ast, scores=scores, error_msg=error_msg, prefix=prefix, processor=processor, rules=rules)
        # print(f"\nFinal Graph Result:\nresult = {graph}")
        return graph, scores, error_msg


class SemanticTeacher(Teacher):
    # TODO
    def __init__(self, propritary=False):
        # Initialize runtime
        runtime_data = {
            f'data("{x}")': np.load(f"data/{x}.npy")
            for x in ["open", "high", "low", "close", "volume"]}
        x_open = runtime_data['data("open")']
        x_close = runtime_data['data("close")']
        x_close_d1 = np.roll(runtime_data['data("close")'], shift=1, axis=0)
        x_close_d1[0] = x_close[0]
        x_logret = np.log(x_close / x_close_d1)
        runtime_data['data("price")'] = x_close
        runtime_data['data("returns")'] = x_logret  # logret
        self.runtime = Runtime(data=runtime_data)
        self.backtester = Backtester(x_close, x_open)
        self.data_shape = x_close.shape
        self.factors = dict()

        self.propritary = propritary
        if propritary:
            from morpho.score_propritary import prop
            self.prop = prop
            for f_name, f_fn in self.prop["factors"].items():
                self.factors[f_name] = f_fn(x_close)

        raise NotImplementedError

    def score(self, graph: Node) -> Tuple[Any, Dict, str]:
        pos, scr, msg = self.position(graph)
        pos, scr, msg = self.normalize(pos, scr, msg)
        pos, scr, msg = self.position_stats(pos, scr, msg)
        pnl, scr, msg = self.pnl(pos, scr, msg)
        pnl, scr, msg = self.pnl_stats(pnl, scr, msg)
        return pnl, scr, msg

    def position(self, graph, scores=dict(), error_msg=[]):
        # calculate position
        prefix = "position"

        def processor(graph):
            set_timeout(seconds=10)
            return self.runtime.run(graph)
        rules = [
            ("Resulting type is not ndarray", 1.,
             lambda pos: type(pos) != np.ndarray),
            ("Resulting size mismatch", 1., lambda pos: pos.shape != self.data_shape),
            ("Too many empty position", 1., lambda pos: np.nanmin(
                np.nansum(np.isfinite(pos), axis=1)) < 3),
        ]
        position, scores, error_msg = self.test(
            graph, scores=scores, error_msg=error_msg, prefix=prefix, processor=processor, rules=rules)

        if self.propritary:
            position, scores, error_msg = self.prop["position"](
                graph, position, scores, error_msg)
        return position, scores, error_msg

    @staticmethod
    def _normalize(position_input):
        # normalize position
        # adjust weight for l and s each
        # check neutralization
        position_raw = position_input - \
            np.nanmean(position_input, axis=1, keepdims=True)
        ls = position_raw / \
            np.nansum(np.where(position_raw >= 0, position_raw, np.nan),
                      axis=1, keepdims=True)
        ss = position_raw / \
            np.nansum(np.where(position_raw < 0, position_raw, np.nan),
                      axis=1, keepdims=True)
        position_raw = np.where(position_raw >= 0, ls, ss)
        position = np.nan_to_num(position_raw, 0)
        return position

    def normalize(self, position, scores: Dict, error_msg: List):
        prefix = "normalize"
        processor = self._normalize
        rules = []
        pos, scores, error_msg = self.test(
            position, scores=scores, error_msg=error_msg, prefix=prefix, processor=processor, rules=rules)

        if self.propritary:
            pos, scores, error_msg = self.prop["normalize"](
                pos, position, scores, error_msg)
        raise NotImplemented

    def position_stats(self, position, scores: Dict, error_msg: List):
        scores["concentration"] = np.nanmax(np.abs(position))
        if self.propritary:
            scores = self.prop["position_stats"](position)
        return position, scores, error_msg

    def pnl(self, position, scores: Dict, error_msg: List):

        prefix = "pnl"
        processor = self.backtester.run
        rules = []
        pnl, scores, error_msg = self.test(
            position, scores=scores, error_msg=error_msg, prefix=prefix, processor=processor, rules=rules)
        if self.propritary:
            pnl, scores, error_msg = self.prop["position_stats"](
                position, pnl, scores, error_msg)
        return pnl, scores, error_msg

    def pnl_stats(self, pnl, scores: Dict, error_msg: List):
        returns, turnover = pnl
        scores["mdd"] = np.min(
            np.cummax(np.cumsum(returns)) - np.cumsum(returns)),
        scores["ret"] = np.nanmean(returns),
        scores["tvr"] = np.nanmean(turnover),
        scores["max_tvr"] = np.nanmax(turnover)
        return pnl, scores, error_msg
