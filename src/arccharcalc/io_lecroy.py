"""Reading LeCroy WaveRunner oscilloscope CSV files."""

from pathlib import Path

import numpy as np


def read_lecroy_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one LeCroy CSV file.

    The file has 5 header lines, then two comma-separated columns:
    time (μs) and amplitude (V).

    Returns (t, v): time in microseconds and amplitude in volts, as numpy arrays.
    """
    t, v = np.loadtxt(path, delimiter=",", skiprows=5, unpack=True)
    return t*1e6, v