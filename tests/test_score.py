import pytest
from morpho.score import SyntaxTeacher #, SematicTeacher

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
    assert scores == {'lex': 5.0, 'parse': 5.0, 'type_check': 5.0, 'build': 5.0}
    graph, scores, error_msg = teacher.score(CODE_SHORT_ASSIGNMENT)
    assert scores == {'lex': 5.0, 'parse': 5.0, 'type_check': 5.0, 'build': 5.0}
    graph, scores, error_msg = teacher.score(CODE_CHAIN)
    assert scores == {'lex': 5.0, 'parse': 5.0, 'type_check': 5.0, 'build': 5.0}
    graph, scores, error_msg = teacher.score(CODE_EXCESS_PARENTHESIS)
    assert scores == {'lex': 5.0, 'parse': 1.0, 'type_check': None, 'build': None}
    graph, scores, error_msg = teacher.score(CODE_TYPE_MISMATCH)
    assert scores == {'lex': 5.0, 'parse': 5.0, 'type_check': 1.0, 'build': None}

if __name__ == "__main__":
    pytest.main()
    