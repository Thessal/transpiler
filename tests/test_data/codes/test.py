import numpy as np
import pandas as pd


def ts_diff(x: np.ndarray) -> np.ndarray:
    return pd.DataFrame(x).diff().values
