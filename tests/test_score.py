import pytest
from morpho.score import SemanticTeacher, SyntaxTeacher
import numpy as np

CODE = """ts_diff_test(signal_1 : Signal<Float>, period_1 : Int) : Signal<Float> = {
    result = subtract(x=signal_1, y=ts_delay(signal=signal_1, period=period_1))
}
result : Signal<Float> = ts_diff_test(signal_1=data(id="close"), period_1=10)"""
CODE_SHORT_ASSIGNMENT = """ts_diff_test(signal : Signal<Float>, period : Int) : Signal<Float> = { result = subtract(x=signal, y=ts_delay(signal=signal, period=period)) }
result = ts_diff_test(signal=data(id="close"), period=10)"""
CODE_CHAIN = "result=subtract(x=data(id=\"close\"),y=ts_delay(signal=data(id=\"close\"), period=10))"

CODE_EXCESS_PARENTHESIS = "result = ts_diff_test(signal=data(id=\"close\"), period=(10))"
CODE_TYPE_MISMATCH = "result = 10"


def test_syntax():
    teacher = SyntaxTeacher()
    graph, scores, error_msg = teacher.score(CODE)
    assert scores == {'lex': 5.0, 'parse': 5.0,
                      'type_check': 5.0, 'build': 5.0}
    graph, scores, error_msg = teacher.score(CODE_SHORT_ASSIGNMENT)
    assert scores == {'lex': 5.0, 'parse': 5.0,
                      'type_check': 5.0, 'build': 5.0}
    graph, scores, error_msg = teacher.score(CODE_CHAIN)
    assert scores == {'lex': 5.0, 'parse': 5.0,
                      'type_check': 5.0, 'build': 5.0}
    graph, scores, error_msg = teacher.score(CODE_EXCESS_PARENTHESIS)
    assert scores == {'lex': 5.0, 'parse': 1.0,
                      'type_check': None, 'build': None}
    graph, scores, error_msg = teacher.score(CODE_TYPE_MISMATCH)
    assert scores == {'lex': 5.0, 'parse': 5.0,
                      'type_check': 1.0, 'build': None}


def test_semantic():
    np.random.seed(42)
    x_open = 1. + np.abs((np.random.random(size=(300, 30))- 0.4).cumsum(axis=0))
    x_close = x_open + np.random.random(size=(300, 30)).cumsum(axis=0) * 0.01
    np.save("./tests/test_data/npy/open.npy", x_open)
    np.save("./tests/test_data/npy/close.npy", x_close)
    teacher = SyntaxTeacher()
    graph, _, _ = teacher.score(CODE)
    teacher = SemanticTeacher(datadir="./tests/test_data/npy")
    graph, scores, error_msg = teacher.score(graph)
    scores_valid = {'position': 1.0, 'normalize': 5.0, 'position_concentration': 0.3061104724374416, 'pnl': 5.0, 'mdd': 0.07822559653025984, 'ret': 0.02507018085354917, 'std': 0.02953318665503461, 'tvr': 0.029341928110631306, 'max_tvr': 0.06874805407743513}
    assert all(str(scores[k])[:10] == str(v)[:10] for k,v in scores_valid.items())

if __name__ == "__main__":
    pytest.main()