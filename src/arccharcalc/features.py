"""Breakdown characteristics: one rule and one unit per quantity."""

import numpy as np
from scipy.signal import find_peaks, savgol_filter

# IEC 60060-1: front time T1 = 1.67 * (t90 - t30).
FRONT_FACTOR = 1.67

# Flashover current Ip1 (Angeli-Knai 2022, 3.1.3): first peak of the discharge current whose
# prominence is at least this fraction of the maximum (ignores small wiggles before it).
PEAK_PROMINENCE = 0.3

# Final jump fj (Angeli-Knai 2022, 3.1.2): maximum slope of the discharge current, from FJ_SEARCH_FROM_US
# (t_sync, skips the front, where I_exc has an unexplained +-6 A artefact) up to the first peak.
# The slope is a Savitzky-Golay derivative: fj moved < 1 ns for windows of 5-50 ns (2026-11-20, 9 shots).
FJ_WINDOW_US = 0.02
FJ_ORDER = 3
FJ_SEARCH_FROM_US = 0.3


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


def first_peak(i: np.ndarray) -> int:
    """Return the index of the first peak of i (A), the flashover current Ip1.

    A peak counts if its prominence is at least PEAK_PROMINENCE of max(i).
    Raises ValueError if there is none.
    """
    peaks, _ = find_peaks(i, prominence=PEAK_PROMINENCE * i.max())
    if len(peaks) == 0:
        raise ValueError("No current peak found")
    return int(peaks[0])


def t_final_jump(t: np.ndarray, i: np.ndarray, k_peak: int) -> float:
    """Return the time (μs) of the final jump: the maximum of di/dt before the peak k_peak.

    t is t_sync (μs, uniform), i the discharge current (A); only t >= FJ_SEARCH_FROM_US is searched.
    Raises ValueError if that start is not before the peak.
    """
    dt = t[1] - t[0]
    n = int(round(FJ_WINDOW_US / dt)) | 1  # odd window length, in samples
    didt = savgol_filter(i, n, FJ_ORDER, deriv=1, delta=dt)
    lo = np.searchsorted(t, FJ_SEARCH_FROM_US)
    if lo >= k_peak:
        raise ValueError("The current peak comes before the search start of the final jump")
    return float(t[lo + np.argmax(didt[lo : k_peak + 1])])


def cg_qv(u: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    """Return (C_g, standard error) in nF: least squares line Q = C_g * U through the origin.

    u in kV, q in μC (μC / kV = nF), only the samples of the fit window (the caller
    chooses it, e.g. the withstand front). Closed form: C_g = Σ(U·Q) / Σ(U²).
    The standard error assumes independent (white) residuals; the noise here is
    correlated, so it underestimates the real error.
    """
    uu = u @ u  # Σ U², `@` = dot product of two 1-D arrays
    cg = (u @ q) / uu
    r = q - cg * u  # residuals
    se = np.sqrt((r @ r) / (len(u) - 1) / uu)
    return float(cg), float(se)
