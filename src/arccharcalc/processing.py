"""Basic waveform quantities: baseline (offset) removal, voltage in kV, current in A, charge in uC."""

import numpy as np

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


def find_onset(t: np.ndarray, x: np.ndarray) -> float:
    """Return the time (μs) where the wave starts in x.

    Use the voltage channel (C3): the front is clearest there.
    """
    n = np.searchsorted(t, t[0] + BASELINE_US) # returns the index of the first sample after the baseline window
    ref = x[:n] # first n samples for baseline reference
    median = np.median(ref)
    sigma = 1.4826 * np.median(np.abs(ref - median))  # MAD -> std of normal noise
    if sigma == 0:
        raise ValueError("Flat baseline: cannot estimate the noise level")
    far = (np.abs(x - median) > ONSET_SIGMAS * sigma).astype(int)
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


def remove_offset(t: np.ndarray, x: np.ndarray, t_window: float) -> np.ndarray:
    """Subtract the mean of x over the samples before t_window (μs)."""
    
    return x - x[t < t_window].mean()


def baseline_stats(
    t: np.ndarray, x: np.ndarray, t_window : float
) -> tuple[float, float, float]:
    """Return (mean, std, half difference) of x over the samples before t_end.

    The half difference is mean(second half) - mean(first half) of the window:
    close to zero when the baseline is quiet, large when it drifts.
    """
    window = x[t < t_window]
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


# --- Charge uncertainty: PROPOSAL, commented out until discussed with Petros ---
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
