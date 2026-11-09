"""Reference voltage: the theoretical double exponential shifted onto a measured wave."""

import numpy as np
from scipy.optimize import least_squares

from arccharcalc.generator import LAB_MARX, MarxParams, double_exp_pu, double_exp_slope

# Residual (kV) above which a robust loss starts to down-weight a sample: about
# 2x the residual rms of the withstand shot (0.5 kV, scratch 2026-10-31).
# shortcut: fixed value, tune per campaign once robust losses are compared.
ROBUST_SCALE_KV = 1.0


def match(
    t: np.ndarray,
    u: np.ndarray,
    amplitude: float,
    polarity: int,
    t_from: float,
    t_to: float,
    tau0: float,
    m: MarxParams = LAB_MARX,
    loss: str = "linear",
) -> tuple[float, float, float]:
    """Return (tau, tau_se, rms) in (μs, μs, kV): shift of the theoretical wave onto u.

    Model: amplitude * double_exp_pu(t - tau), fitted to polarity * u over
    t_from <= t <= t_to; tau is the only unknown, tau0 its first guess (e.g. the
    onset). The same function serves the withstand shot and the breakdown shots:
    the caller chooses amplitude and window. `loss` is passed to least_squares
    ("linear" = ordinary least squares, "soft_l1", "huber", ...).

    tau_se comes from the Jacobian and assumes white noise, so it underestimates
    the real error when the residual is systematic.
    """
    if polarity not in (1, -1):
        raise ValueError("polarity must be +1 or -1")
    pu = polarity * u
    if pu.max() < -pu.min():
        raise ValueError("The wave is mostly of the other polarity: wrong polarity or wrong shot?")

    w = (t >= t_from) & (t <= t_to)
    t_w, u_w = t[w], pu[w]

    def residual(p: np.ndarray) -> np.ndarray:
        return u_w - amplitude * double_exp_pu(t_w - p[0], m)

    def jacobian(p: np.ndarray) -> np.ndarray:
        # d/dtau of -amplitude * pu(t - tau) = +amplitude * slope(t - tau)
        return (amplitude * double_exp_slope(t_w - p[0], m))[:, None]

    res = least_squares(residual, [tau0], jac=jacobian, loss=loss, f_scale=ROBUST_SCALE_KV)
    r = res.fun
    s2 = (r @ r) / (len(r) - 1)
    tau_se = np.sqrt(s2 / (res.jac[:, 0] @ res.jac[:, 0]))
    return float(res.x[0]), float(tau_se), float(np.sqrt(np.mean(r**2)))
