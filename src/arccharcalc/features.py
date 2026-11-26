"""Breakdown characteristics: one rule and one unit per quantity."""

import numpy as np
from scipy.signal import find_peaks, savgol_filter

# IEC 60060-1: front time T1 = 1.67 * (t90 - t30).
FRONT_FACTOR = 1.67

# Current peaks (Angeli-Knai 2022, 3.1.3): every local maximum of the discharge current with height at least
# PEAK_HEIGHT of max(i), in time order. The first is the flashover current Ip1 (not necessarily the overall
# maximum), the second Ip2. A peak must also rise PEAK_PROMINENCE_A above its surroundings, ~5x the C4 noise (~1 A):
# 9 shots (2026-11-26), real peaks 15.7-443 A, a noise wiggle on the flat top of 15.5/3 0.07 A.
PEAK_HEIGHT = 0.8
PEAK_PROMINENCE_A = 5.0

# Final jump fj (Angeli-Knai 2022, 3.1.2): maximum slope of the discharge current, from FJ_SEARCH_FROM_US
# (t_sync, skips the front, where I_exc has an unexplained +-6 A artefact) up to the first peak.
# The slope is a Savitzky-Golay derivative: fj moved < 1 ns for windows of 5-50 ns (2026-11-20, 9 shots).
FJ_WINDOW_US = 0.02
FJ_ORDER = 3
FJ_SEARCH_FROM_US = 0.3

# Leader inception li (Angeli-Knai 2022, 3.1.1): start of the steady rise of the discharge current, i.e. of the
# last run of di/dt > 0 that lasts up to the search end. Savitzky-Golay window LI_WINDOW_US: shorter windows break
# the rise on single streamer pulses (9 shots, C2, 2026-11-26). The last zero crossing of the current is not used:
# after the front it never returns to 0 before the rise (+0.1...+1 A, 2.4-6.1 us earlier than the rise).
LI_WINDOW_US = 1.0
# C2 clips (~7 A at 15.5/16.25 kV, 70/141 A at 30 kV): the li search on C2 stops where it reaches this fraction of its maximum.
LI_C2_CEILING = 0.95
# The search starts at t_sync = 0 (not after the front, as fj): at high charging voltages the leader can start on the front.
LI_SEARCH_FROM_US = 0.0


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


def current_peaks(i: np.ndarray) -> np.ndarray:
    """Return the indices of the peaks of i (A) in time order: [0] is Ip1, [1] (if any) Ip2.

    A peak is a local maximum with height at least PEAK_HEIGHT of max(i) and prominence
    at least PEAK_PROMINENCE_A. Raises ValueError if there is none.
    """
    peaks, _ = find_peaks(i, height=PEAK_HEIGHT * i.max(), prominence=PEAK_PROMINENCE_A)
    if len(peaks) == 0:
        raise ValueError("No current peak found")
    return peaks


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


def t_steady_rise(t: np.ndarray, i: np.ndarray, k_end: int) -> float | None:
    """Return the time (μs) where the steady rise of i (A) starts: the last run of di/dt > 0 reaching k_end.

    t is t_sync (μs, uniform); only LI_SEARCH_FROM_US <= t < t[k_end] is searched.
    Returns None if the rise already runs at the search start (its start is not seen)
    or if di/dt is not positive just before k_end.
    """
    dt = t[1] - t[0]
    n = int(round(LI_WINDOW_US / dt)) | 1  # odd window length, in samples
    didt = savgol_filter(i, n, FJ_ORDER, deriv=1, delta=dt)
    lo = np.searchsorted(t, LI_SEARCH_FROM_US)
    not_rising = np.flatnonzero(didt[lo:k_end] <= 0)
    if len(not_rising) == 0 or not_rising[-1] == k_end - lo - 1:
        return None
    return float(t[lo + not_rising[-1] + 1])  # first sample of the last run of di/dt > 0


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
