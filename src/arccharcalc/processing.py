"""Basic waveform quantities: baseline (offset) removal and voltage in kV."""

import numpy as np

# Voltage divider ratio: C3 (low-voltage side) -> insulator (Excel, cell Z4).
DIVIDER_RATIO = 1559.93

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
    n = np.searchsorted(t, t[0] + BASELINE_US)
    ref = x[:n]
    median = np.median(ref)
    sigma = 1.4826 * np.median(np.abs(ref - median))  # MAD -> std of normal noise
    if sigma == 0:
        raise ValueError("Flat baseline: cannot estimate the noise level")
    far = (np.abs(x - median) > ONSET_SIGMAS * sigma).astype(int)
    runs = np.convolve(far, np.ones(ONSET_SAMPLES, dtype=int), mode="valid")
    found = runs == ONSET_SAMPLES
    if not found.any():
        raise ValueError("No wave found in the record")
    return float(t[np.argmax(found)])


def offset_window_end(t: np.ndarray, c3: np.ndarray) -> float:
    """Return the end time (μs) of the offset window, taken from the voltage channel.

    The same window is used for the current channels (C2, C4) of the shot.
    """
    return find_onset(t, c3) - ONSET_MARGIN_US


def remove_offset(t: np.ndarray, x: np.ndarray, t_end: float) -> np.ndarray:
    """Subtract the mean of x over the samples before t_end (μs)."""
    # Not the FFT DC term: that is the mean of the whole record, impulse included
    # (+17.35 V instead of -0.18 V on C3 of shot 15.5 kV / 00003).
    return x - x[t < t_end].mean()


def baseline_stats(
    t: np.ndarray, x: np.ndarray, t_end: float
) -> tuple[float, float, float]:
    """Return (mean, std, half difference) of x over the samples before t_end.

    The half difference is mean(second half) - mean(first half) of the window:
    close to zero when the baseline is quiet, large when it drifts.
    """
    window = x[t < t_end]
    half = len(window) // 2
    return window.mean(), window.std(), window[half:].mean() - window[:half].mean()


def voltage_kv(t: np.ndarray, c3: np.ndarray, t_end: float) -> np.ndarray:
    """Convert C3 (volts, low-voltage side) to the insulator voltage in kV."""
    return remove_offset(t, c3, t_end) * DIVIDER_RATIO / 1000
