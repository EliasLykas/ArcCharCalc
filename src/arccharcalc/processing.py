"""Basic waveform quantities: baseline (offset) removal, voltage in kV, current in A, charge in uC."""

import numpy as np
from scipy.signal import savgol_filter

# Savitzky-Golay filter for dU/dt (decision 2026-11-19, plan/REPORT_derivative-filter.md):
# window 0.3 us, order 3 (tuned on the 11 kV withstand: peak slope within 2 % of the analytic one).
SG_WINDOW_US = 0.3
SG_ORDER = 3

# Voltage divider ratio: C3 (low-voltage side) -> insulator (Excel, cell Z4).
DIVIDER_RATIO = 1559.93

# Current sensor scale: C2 and C4 (volts) -> amperes (campaign README; the
# sensor itself, 0.01 V/A, is still unconfirmed).
CURRENT_SCALE = 100

# Wave onset detection (decision 2026-10-29, see plan/PLAN.md). A rough baseline
# comes from the first BASELINE_US of the record; the onset is the first run of
# ONSET_SAMPLES samples lying more than ONSET_SIGMAS noise sigmas away from it.
# It does not depend on how the oscilloscope triggers.
BASELINE_US = 2.0
ONSET_SIGMAS = 8
ONSET_SAMPLES = 20

# The offset window ends this long before the onset, so the start of the front
# stays out of the average. The front begins ~0.1 us before C3 exceeds 8 sigma
# (measured on 10 shots, 2026-10-29); 0.2 us leaves a safety factor of 2.
ONSET_MARGIN_US = 0.2


def find_onset(t: np.ndarray, v: np.ndarray) -> float:
    """Return the time (μs) where the wave starts in v.

    Use the voltage channel (C3): the front is clearest there.
    """
    n = np.searchsorted(t, t[0] + BASELINE_US) # returns the index of the first sample after the baseline window
    ref = v[:n] # first n samples for baseline reference
    median = np.median(ref)
    sigma = 1.4826 * np.median(np.abs(ref - median))  # MAD -> std of normal noise
    if sigma == 0:
        raise ValueError("Flat baseline: cannot estimate the noise level")
    far = (np.abs(v - median) > ONSET_SIGMAS * sigma).astype(int)
    runs = np.convolve(far, np.ones(ONSET_SAMPLES, dtype=int), mode="valid") #it makes an array of the sum of ONSET_SAMPLES consecutive elements of far
    found = runs == ONSET_SAMPLES # True where all ONSET_SAMPLES samples of the run are further than ONSET_SIGMAS*sigma from the median
    if not found.any():
        raise ValueError("No wave found in the record")
    return float(t[np.argmax(found)]) # argmax = index of the first True, t[index] = time of the first sample of the first run


def offset_window_end(t: np.ndarray, c3: np.ndarray) -> float:
    """Return the end time (μs) of the offset window, taken from the voltage channel.

    The same window is used for the current channels (C2, C4) of the shot.
    """
    return find_onset(t, c3) - ONSET_MARGIN_US # the window ends ONSET_MARGIN_US before the onset


def remove_offset(t: np.ndarray, v: np.ndarray, t_window: float) -> np.ndarray:
    """Subtract the mean of v over the samples before t_window (μs)."""
    
    return v - v[t < t_window].mean()


def baseline_stats(
    t: np.ndarray, v: np.ndarray, t_window : float
) -> tuple[float, float, float]:
    """Return (mean, std, half difference) of v over the samples before t_end.

    The half difference is mean(second half) - mean(first half) of the window:
    close to zero when the baseline is quiet, large when it drifts.
    """
    window = v[t < t_window]
    half = len(window) // 2
    return window.mean(), window.std(), window[half:].mean() - window[:half].mean()


def voltage_kv(t: np.ndarray, c3: np.ndarray, t_window: float) -> np.ndarray:
    """Convert C3 (volts, low-voltage side) to the insulator voltage in kV."""
    return remove_offset(t, c3, t_window) * DIVIDER_RATIO / 1000


def current_a(t: np.ndarray, c: np.ndarray, t_window: float) -> np.ndarray:
    """Convert a current channel (C2 or C4, volts) to amperes.

    t_end is the offset window end of the shot, taken from C3 (offset_window_end).
    """
    return remove_offset(t, c, t_window) * CURRENT_SCALE


def charge_uc(t: np.ndarray, i: np.ndarray, t_start: float) -> np.ndarray:
    """Return the charge (uC) passed since t_start (μs), the onset of the wave.

    Cumulative trapezoid rule; the current before t_start counts as zero. With t
    in μs and i in A the result is directly in μC. Excel uses the rectangle rule,
    which differs from this by i * dt / 2 (0.05 uC at a 400 A peak, 0.001 uC at
    2 A): see plan/PLAN.md, decision 2026-10-29.
    """
    i = np.where(t >= t_start, i, 0.0)
    return np.concatenate([[0.0], np.cumsum((i[1:] + i[:-1]) / 2 * np.diff(t))])

def energy_mj(t: np.ndarray, u: np.ndarray, i: np.ndarray, t_start: float) -> np.ndarray:
    """Return the energy (mJ) E = ∫ u·i dt delivered since t_start (μs).

    Same scheme as charge_uc: cumulative trapezoid rule, zero before t_start. With u in kV,
    i in A and t in μs the result is directly in mJ (kV·A·μs = mJ). Pass the discharge
    current i_exc to get the discharge energy.
    """
    p = np.where(t >= t_start, u * i, 0.0)  # instantaneous power (kW)
    return np.concatenate([[0.0], np.cumsum((p[1:] + p[:-1]) / 2 * np.diff(t))])


def capacitive_current_a(t: np.ndarray, u: np.ndarray, c_g: float) -> np.ndarray:
    """Return the capacitive current I_c = C_g * dU/dt (A) of the reference u (kV).

    c_g in nF (nF * kV/μs = A). The derivative is a Savitzky-Golay filter (window
    SG_WINDOW_US, order SG_ORDER): smooths and differentiates in one step, with no
    time shift. t must be uniformly spaced (μs), as in the oscilloscope records.
    """
    dt = t[1] - t[0]
    n = int(round(SG_WINDOW_US / dt))
    n += 1 - n % 2  # the window length must be odd
    return c_g * savgol_filter(u, n, SG_ORDER, deriv=1, delta=dt)


# --- Not used: kept to be discussed (decision 2026-11-13, C2 compresses before its ceiling). ---
#def current_joint(c2: np.ndarray, c4: np.ndarray) -> np.ndarray:
#    """Return the joint current (A) from C2 and C4, with the same offset window as C3.
#
#    The joint current is the sum of the two channels, after removing their offsets.
#    """
#    #intex = np.argmax(c2)  # index of the peak of C2, used to align the two channels
#
#    far = (np.abs(c2) == np.abs(c2).max()).astype(int)
#    runs = np.convolve(far, np.ones(ONSET_SAMPLES, dtype=int), mode="valid")
#    found = runs == ONSET_SAMPLES
#    if not found.any():
#            raise ValueError("No wave found in the record")
#    intex = np.argmax(found)  # index of the first sample of the first run 
#    joint = np.concatenate((c2[:intex], c4[intex:]))  # sum the two channels, with C4 starting at the peak of C2
#    return joint

# --- Charge uncertainty: PROPOSAL, commented out until discussed ---
# The baseline noise is not white (correlated over up to ~500 ns), so the standard
# error of the offset is NOT sigma / sqrt(N). Instead, split the offset window into
# blocks, take the mean of each block, and use the scatter of those means.
# Measured on 4 shots (2026-10-30): for blocks of 100-500 samples the estimate is
# stable and 6-9 times larger than the white-noise formula (C4 ~0.05 A, C2 ~0.0006 A,
# C3 ~0.009 V); for blocks of 1000+ samples only 2-4 blocks remain and it is unreliable.
# The offset error is constant, so it integrates linearly: error(Q) = SE * (t - t_start).
# Covers only the offset; not the noise inside the pulse, nor the error of t_start.
#
# OFFSET_SE_BLOCK = 250  # samples (= ns), inside the 100-500 plateau
# OFFSET_SE_MIN_BLOCKS = 8
#
#
# def offset_se(t: np.ndarray, x: np.ndarray, t_window: float) -> float:
#     """Return the standard error of the offset of x, in the units of x (volts).
#
#     Computed from the scatter of block means over the offset window (before t_window).
#     """
#     window = x[t < t_window]
#     n_blocks = len(window) // OFFSET_SE_BLOCK
#     if n_blocks < OFFSET_SE_MIN_BLOCKS:
#         raise ValueError("Offset window too short to estimate the offset error")
#     blocks = window[: n_blocks * OFFSET_SE_BLOCK].reshape(n_blocks, OFFSET_SE_BLOCK)
#     return float(blocks.mean(axis=1).std(ddof=1) / np.sqrt(n_blocks))
#
#
# def charge_error_uc(t: np.ndarray, t_start: float, offset_se_a: float) -> np.ndarray:
#     """Return the charge error (uC) caused by the offset error (A), zero before t_start."""
#     return offset_se_a * np.clip(t - t_start, 0.0, None)
#
#
# Use in main(), for each current channel:
#     se = offset_se(t, channels[name], t_window) * CURRENT_SCALE   # A
#     print(f"{name} offset error (A): {se:.4f}, charge error (uC): {charge_error_uc(t, onset, se)[k]:.3f}")
