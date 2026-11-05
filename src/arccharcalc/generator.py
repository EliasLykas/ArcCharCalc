"""Impulse generator: theoretical double exponential (per unit) from its R, C."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarxParams:
    """Equivalent (total) R, C of the impulse generator.

    Values from Petros's Excel, sheet "56-1_14.5_3", cells C3-C7. Only the shape of the
    wave is derived from them; its amplitude comes from the measurement.
    """

    r_f: float  # Ω, front resistor
    r_t: float  # Ω, tail resistor
    c_c: float  # nF, charging capacitor
    c_f: float  # nF, load (front) capacitor


LAB_MARX = MarxParams(r_f=2054.0, r_t=4950.0, c_c=15.0, c_f=0.2)


def time_constants(m: MarxParams) -> tuple[float, float]:
    """Return (T1, T2) in μs: T1 = R_T*(C_C + C_F), T2 = R_F*C_C*C_F/(C_C + C_F).

    With R in Ω and C in nF the product is in ns, hence the factor 1e-3.
    """
    t1 = m.r_t * (m.c_c + m.c_f) * 1e-3
    t2 = m.r_f * m.c_c * m.c_f / (m.c_c + m.c_f) * 1e-3
    return t1, t2


def _peak(t1: float, t2: float) -> tuple[float, float]:
    """Return (t_peak in μs, value there) of exp(-t/t1) - exp(-t/t2)."""
    t_peak = t1 * t2 / (t1 - t2) * np.log(t1 / t2)
    return t_peak, np.exp(-t_peak / t1) - np.exp(-t_peak / t2)


def double_exp_pu(t: np.ndarray, m: MarxParams = LAB_MARX) -> np.ndarray:
    """Return the double exponential with unit peak, zero for t <= 0 (t in μs).

    shape(t) = (exp(-t/T1) - exp(-t/T2)) / peak, so that max(shape) = 1 at t_peak.
    """
    t1, t2 = time_constants(m)
    peak = _peak(t1, t2)[1]
    t_pos = np.clip(t, 0, None)  # keeps exp(-t/t2) from overflowing for t < 0
    return np.where(t > 0, (np.exp(-t_pos / t1) - np.exp(-t_pos / t2)) / peak, 0.0)


def double_exp_slope(t: np.ndarray, m: MarxParams = LAB_MARX) -> np.ndarray:
    """Return d(double_exp_pu)/dt in 1/μs, zero for t <= 0 (t in μs).

    slope(t) = (exp(-t/T2)/T2 - exp(-t/T1)/T1) / peak, zero at t_peak.
    """
    t1, t2 = time_constants(m)
    peak = _peak(t1, t2)[1]
    t_pos = np.clip(t, 0, None)  # keeps exp(-t/t2) from overflowing for t < 0
    return np.where(t > 0, (np.exp(-t_pos / t2) / t2 - np.exp(-t_pos / t1) / t1) / peak, 0.0)
