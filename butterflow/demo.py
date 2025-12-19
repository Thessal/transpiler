
# --- 4. Demonstration ---

import numpy as np

engine = Reducer()
data_src = Signal(np.array([10, 20, 30, 40, 50]))

print("--- Scenario 1: Config-First (Currying) ---")
# User defines the generic "Smooth" operator first
# Note: We provide 'window' first. The engine doesn't care about order.
op_smooth = engine.apply(engine.get_op("SMA"), window=3)

print(f"Current State: {op_smooth}") 
# Output: SMA(window=3, input=?)

print("\n--- Scenario 2: Data Injection ---")
# Now we provide the missing 'input'
result = engine.apply(op_smooth, input=data_src)

print(f"Result: {result}")

print("\n--- Scenario 3: Mixed Order One-Shot ---")
# We can provide arguments in ANY order.
# Here we provide 'input' before 'window', opposite of how we might think conventionally.
res2 = engine.apply(engine.get_op("SMA"), input=data_src, window=2)
print(f"Result (Window=2): {res2}")







# =========================
# Demo
# =========================
if __name__ == "__main__":
    program = '''
close : Signal<Real> := input("close")

# currying:
d1    : Op<Signal<Real>, Signal<Real>> := delay(1)
m5    : Op<Signal<Real>, Signal<Real>> := rolling_mean(5)

# tuple-of-operators input (also valid as compose(d1, m5)):
f     : Op<Signal<Real>, Signal<Real>> := compose((d1, m5))

out   : Signal<Real> := f(close)
'''

    _, nodes, dag = compile_program(program)

    # reduce (flatten + saturate when fully matched)
    red = Reducer(dag.mk)
    out_node = red.reduce(nodes["out"])

    # evaluate
    T, N = 20, 6
    rng = np.random.default_rng(0)
    close = rng.normal(size=(T, N)).astype(float)
    close[0, 0] = np.nan

    ev = Evaluator({"close": close})
    out = ev.eval(out_node)
    print(out.shape)
    print(out[:2])
