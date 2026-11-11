"""Breakdown characteristics: one rule and one unit per quantity."""

import numpy as np

# IEC 60060-1: front time T1 = 1.67 * (t90 - t30).
FRONT_FACTOR = 1.67


def _crossing(t: np.ndarray, front: np.ndarray, level: float) -> float:
    """Return the time (μs) where `front` first reaches `level`, linearly interpolated."""
    k = np.argmax(front >= level)  # first sample at or above the level
    if k == 0:
        return float(t[0])
    # straight line between samples k-1 and k
    return float(t[k - 1] + (level - front[k - 1]) * (t[k] - t[k - 1]) / (front[k] - front[k - 1]))


def front_time(t: np.ndarray, u: np.ndarray, peak: float) -> float:
    """Return the front time T1 = 1.67 * (t90 - t30) in μs (IEC 60060-1).

    u is positive (polarity already applied); peak (kV) sets the 30 % and 90 % levels.
    Only the front is searched, i.e. the samples up to the maximum of u.
    Raises ValueError if the front never reaches 90 % of peak (breakdown on the front).
    """
    front = u[: np.argmax(u) + 1]
    if front[-1] < 0.9 * peak:
        raise ValueError("The front never reaches 90 % of the peak")
    t30 = _crossing(t, front, 0.3 * peak)
    t90 = _crossing(t, front, 0.9 * peak)
    return FRONT_FACTOR * (t90 - t30)
