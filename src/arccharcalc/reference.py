"""Time alignment: shift a measured wave in time so that its front matches a fixed reference."""

import numpy as np
from scipy.optimize import least_squares

# Residual (kV) above which a robust loss starts to down-weight a sample: about
# 2x the residual rms of the withstand shot (0.5 kV, scratch 2026-10-31).
# shortcut: fixed value, tune per campaign once robust losses are compared.
ROBUST_SCALE_KV = 1.0


def match(
    t: np.ndarray,
    u: np.ndarray,
    t_ref: np.ndarray,
    u_ref: np.ndarray,
    polarity: int,
    lo: float = 0.3,
    hi: float = 0.9,
    loss: str = "linear",
) -> tuple[float, float, float]:
    """Return (tau, tau_se, rms) in (μs, μs, kV): time subtracted from the measurement.

    The reference (t_ref, u_ref) stays fixed; u_ref is positive (no polarity) and its
    maximum is the theoretical peak. The measurement polarity * u, on the axis
    t_sync = t - tau, is matched to the reference: polarity * u(t) ≈ u_ref(t - tau),
    with u_ref linearly interpolated. tau is the only unknown.

    Window: the measured front, from the first sample >= lo * peak up to the first
    sample >= hi * peak, or up to the measured maximum if the wave never reaches
    hi * peak (breakdown on the front, pre-discharges). Same function for Phase A
    (withstand -> theoretical double exponential) and Phase B (breakdown -> adapted
    withstand). `loss` is passed to least_squares ("linear", "soft_l1", "huber", ...).

    tau_se comes from the Jacobian and assumes white noise, so it underestimates
    the real error when the residual is systematic.
    """
    if polarity not in (1, -1):
        raise ValueError("polarity must be +1 or -1")
    pu = polarity * u
    if pu.max() < -pu.min():
        raise ValueError("The wave is mostly of the other polarity: wrong polarity or wrong shot?")

    peak = u_ref.max()
    front = pu[: np.argmax(pu) + 1]  # samples up to the measured maximum
    if front[-1] < lo * peak:
        raise ValueError(f"The measurement never reaches {lo:.0%} of the reference peak")
    intex_lo = np.argmax(front >= lo * peak)  # argmax of a bool array = first True
    above_hi = front >= hi * peak
    intex_hi = np.argmax(above_hi) if above_hi.any() else len(front) - 1
    t_w, u_w = t[intex_lo : intex_hi + 1], front[intex_lo : intex_hi + 1]
    if len(t_w) < 2:
        raise ValueError("Fewer than 2 samples in the front window")

    slope_ref = np.gradient(u_ref, t_ref)
    tau0 = t[intex_lo] - t_ref[np.argmax(u_ref >= lo * peak)]  # align the lo crossings

    def residual(p: np.ndarray) -> np.ndarray:
        return u_w - np.interp(t_w - p[0], t_ref, u_ref)

    def jacobian(p: np.ndarray) -> np.ndarray:
        # d/dtau of -u_ref(t - tau) = +u_ref'(t - tau)
        return np.interp(t_w - p[0], t_ref, slope_ref)[:, None]

    res = least_squares(residual, [tau0], jac=jacobian, loss=loss, f_scale=ROBUST_SCALE_KV)
    r = res.fun
    s2 = (r @ r) / (len(r) - 1)
    tau_se = np.sqrt(s2 / (res.jac[:, 0] @ res.jac[:, 0]))
    return float(res.x[0]), float(tau_se), float(np.sqrt(np.mean(r**2)))
