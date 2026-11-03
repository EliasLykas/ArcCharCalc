"""Reference voltage: the theoretical double exponential of the Marx generator."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarxParams:
    """Characteristics of the impulse generator (Petros's Excel, sheet "56-1_14.5_3", cells C3-C7).


    """

    r_f: float  # Ω, front resistor (per stage)
    r_t: float  # Ω, tail resistor
    c_c: float  # nF, charging capacitor (per stage)
    c_f: float  # nF, load (front) capacitor
    n: int = 1  # number of stages
    u0: float | None = None  # kV, charging voltage per stage (the DC voltage); only needed for up_ref


LAB_MARX = MarxParams(r_f=2054.0, r_t=4950.0, c_c=15.0, c_f=0.2, n=1)


def time_constants(m: MarxParams) -> tuple[float, float]:
    """Return (T1, T2) in μs: T1 = R_T*(C_C + C_F), T2 = R_F*C_C*C_F/(C_C + C_F).

    With R in Ω and C in nF the product is in ns, hence the factor 1e-3.
    """
    c_c = m.c_c
    r_f = m.r_f
    t1 = m.r_t * (c_c + m.c_f) * 1e-3
    t2 = r_f * c_c * m.c_f / (c_c + m.c_f) * 1e-3
    return t1, t2


def _peak(t1: float, t2: float) -> tuple[float, float]:
    """Return (t_peak in μs, value there) of exp(-t/t1) - exp(-t/t2)."""
    t_peak = t1 * t2 / (t1 - t2) * np.log(t1 / t2)
    return t_peak, np.exp(-t_peak / t1) - np.exp(-t_peak / t2)


def double_exp_shape(t: np.ndarray, m: MarxParams = LAB_MARX) -> np.ndarray:
    """Return the double exponential with unit peak, zero for t <= 0 (t in μs).

    shape(t) = (exp(-t/T1) - exp(-t/T2)) / peak, so that max(shape) = 1 at t_peak.
    """
    t1, t2 = time_constants(m)
    peak = _peak(t1, t2)[1]
    t_pos = np.clip(t, 0, None)  # keeps exp(-t/t2) from overflowing for t < 0
    return np.where(t > 0, (np.exp(-t_pos / t1) - np.exp(-t_pos / t2)) / peak, 0.0)


def utilization(m: MarxParams = LAB_MARX) -> float:
    """Return the utilization factor η = peak voltage / (n*U0) of the generator.

    η = T1*T2 / ((T1 - T2)*R_F*C_F) * peak. Excel cell C13 gives 0.9593 for Petros.
    """
    t1, t2 = time_constants(m)
    r_f_c_f = m.r_f * m.n * m.c_f * 1e-3  # μs
    return t1 * t2 / ((t1 - t2) * r_f_c_f) * _peak(t1, t2)[1]


def up_ref(m: MarxParams) -> float:
    """Return the peak voltage (kV) the generator is expected to give: n*U0*η."""
    if m.u0 is None:
        raise ValueError("MarxParams.u0 (charging voltage) is needed for up_ref")
    return m.n * m.u0 * utilization(m)
