# Morpho

Transpiler for Butterflow.

TODO:

* Integration : bge-m3, Morpho / Rhetenor
* Export to Didius / Validation

## Example

### Input
```
inputs = {
    "idea_file": "../demo/demo_input.txt",
    "spec_file": "language/spec.txt",
    "doc_files": ["alpha/momentum.txt", "alpha/reversion.txt"],
    "func_files": ["operators/ts_delay.txt", "operators/ts_mean.txt"]
}
```
demo_input.txt:
```
import numpy as np
from typing import Union
def ema(data: Union[np.ndarray, list], window_size: int) -> np.ndarray:
    """Calculate the Exponential Moving Average (EMA) for the given data.
    ...
    ...
    return moving_averages
```

### Output
```
logret : Signal<Float> = log(signal=data(id="returns"))
intermediate_return_40 : Signal<Float> = ts_mean(signal=logret, lookback = 40)
result : Signal<Float> = multiply(signal=intermediate_return_40, 1.0)
```