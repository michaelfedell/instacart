import os
import sys

sys.path.append(os.path.abspath('./'))
from src.generate_features import clean
import numpy as np


def test_clean():
    x = np.random.random((10, 4))
    x[1, 3] = np.nan
    x[4, 1] = np.nan
    assert clean(x).shape == (8, 4)
