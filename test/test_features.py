import os
import sys

sys.path.append(os.path.abspath('./'))
from src.generate_features import clean, get_factors
import numpy as np
import pandas as pd


def test_clean_np():
    x = np.random.random((10, 4))
    x[1, 3] = np.nan
    x[4, 1] = np.nan
    cleaned = clean(x)
    assert cleaned.shape == (8, 4)
    assert isinstance(cleaned, np.ndarray)


def test_clean_pd():
    x = pd.DataFrame(np.random.random((10, 4)))
    x[1, 3] = np.nan
    x[4, 1] = np.nan
    cleaned = clean(x)
    assert cleaned.shape == (8, 4)
    assert isinstance(cleaned, pd.DataFrame)


def test_factors():
    data = pd.DataFrame({'a': [1, 2, 3, 4, 5],
                         'b': [-2, -1, 0, 1, 2],
                         'c': [9, 2, 6, -4, 0]})
    # Instantiated based on manual test of above data
    factors = pd.DataFrame({
        'a': [-0.90141, -0.00000, -0.00000],
        'b': [-0.90141, -0.00000, -0.000000],
        'c': [ 0.78011,  0.00000,  0.000000]})
    fac = get_factors(data, 2)
    fac = np.round(fac, 5)
    assert np.all(fac.values == factors.values)
