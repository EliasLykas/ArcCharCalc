"""The processing chain of one withstand shot and of one breakdown shot. Only computes: printing and plotting
are done from the database afterwards (`report.py`)."""

from dataclasses import dataclass

import numpy as np

from arccharcalc.features import LI_C2_CEILING, cg_qv, current_peaks, front_time, t_final_jump, t_steady_rise
from arccharcalc.generator import double_exp_pu
from arccharcalc.processing import (
    capacitive_current_a,
    charge_uc,
    current_a,
    energy_mj,
    find_onset,
    offset_window_end,
    voltage_kv,
)
from arccharcalc.reference import breakdown_amplitude, match

# Check (kept for verification, not run): the commented "check" blocks below print values that are not stored
# in the database. To use one, uncomment it together with these imports and the constant:
# from arccharcalc.processing import baseline_stats
# CHARGE_AT_US = 10.0  # charge is printed this long after the onset (a fixed point, only for checks)


@dataclass(frozen=True)
class Withstand:
    """What the breakdown shots need from the withstand shot (Phase A result)."""

    level: str  # kV, as in the file name
    shot: int
    t_sync: np.ndarray  # μs, t_w - tau_A
    u: np.ndarray  # kV, positive (polarity applied)
    up: float  # kV, u.max()
    c_g: float  # nF, from current channel "C4"
    tau: float  # μs, tau_A (shift onto the double exponential)
    t1: float  # μs, front time 1.67 * (t90 - t30)
    q: np.ndarray | None  # μC, charge from C4 from the onset (for the Q-V plot); None in runs stored before 2026-11-27


@dataclass(frozen=True)
class Breakdown:
    """Results of one breakdown shot (Phase B, 3 and 4), to print or to store."""

    level: str  # kV, as in the file name
    shot: int
    up: float  # kV, max of the voltage with polarity applied
    amplitude: float  # kV, peak of the scaled reference (breakdown_amplitude)
    tau: float  # μs, tau_B (shift onto the withstand reference)
    t1: float | None  # μs, front time; None if the front does not reach 90 %
    events: dict[str, dict[str, float]]  # e.g. {"fj": {"t": μs, "u": kV, "i_exc": A, "q_exc": μC, "e_exc": mJ}, "Ip1": {...}}
    # On the samples of t_sync, polarity applied: "t_sync" (μs), "u" (kV), "i_c4", "i_c2", "i_c", "i_exc" (A),
    # "q_c4", "q_c2", "q_g", "q_exc" (μC), "e_exc" (mJ).
    waveforms: dict[str, np.ndarray]


def align_withstand(t_w: np.ndarray, ch_w: dict[str, np.ndarray], polarity: int) -> float:
    """Phase A, the automatic part (`match`): shift tau_A (us) of the withstand shot onto the double exponential.

    Kept apart from `process_withstand` so that tau_A can be corrected by hand (GUI) before processing.
    t_w, ch_w: time (us) and raw channels (V) of the withstand shot, as `read_shot`/`load_shot` give them.
    """
    t_end_w = offset_window_end(t_w, ch_w["C3"])
    u_w = polarity * voltage_kv(t_w, ch_w["C3"], t_end_w)
    tau_a, _, _ = match(t_w, u_w, t_w, u_w.max() * double_exp_pu(t_w), 1)  # u_w is already positive
    return float(tau_a)


def process_withstand(
    t_w: np.ndarray, ch_w: dict[str, np.ndarray], w_level: str, w_shot: int, polarity: int, tau_a: float
) -> Withstand:
    """Phase A for a given tau_A (from `align_withstand`): T1, C_g from the front, and the t_sync axis t_w - tau_A.

    t_w, ch_w: time (us) and raw channels (V) of the withstand shot, as `read_shot`/`load_shot` give them.
    """
    t_end_w = offset_window_end(t_w, ch_w["C3"])  # one offset window for all withstand channels
    u_w = polarity * voltage_kv(t_w, ch_w["C3"], t_end_w)
    up_w = u_w.max()
    t1_w = front_time(t_w, u_w, up_w)

    # Geometric capacitance C_g: Q = C_g * U on the withstand front, from the onset to the first 90 % of Up.
    onset_w = find_onset(t_w, ch_w["C3"])
    k90 = np.argmax(u_w >= 0.9 * up_w)  # first sample at 90 %
    front_w = (t_w >= onset_w) & (np.arange(len(t_w)) <= k90)  # boolean mask of the fit window
    # Only C4: C2 clips the current front, so its charge, and hence its C_g, would be wrong (user, 2026-11-24).
    q_w = charge_uc(t_w, polarity * current_a(t_w, ch_w["C4"], t_end_w), onset_w)
    c_g, cg_se = cg_qv(u_w[front_w], q_w[front_w])  # nF; [front_w] keeps only the samples of the fit window
    # Check (kept for verification, not run): standard error of C_g and a fit with free intercept.
    # slope, intercept = np.polyfit(u_w[front_w], q_w[front_w], 1)
    # print(f"C_g from C4 (pF): {c_g * 1000:.2f} +/- {cg_se * 1000:.2f}, "
    #       f"free intercept: {slope * 1000:.2f} pF, intercept {intercept:+.4f} uC")

    t_w_sync = t_w - tau_a  # align the withstand on its own t_sync axis
    return Withstand(w_level, w_shot, t_w_sync, u_w, up_w, c_g, tau_a, t1_w, q_w)


def align_breakdown(
    t: np.ndarray, channels: dict[str, np.ndarray], level: str, polarity: int, w: Withstand
) -> tuple[float, float]:
    """Phase B, the automatic part (`match`): amplitude (kV) of the scaled withstand reference and shift tau_B (us).

    Kept apart from `process_breakdown` so that both can be corrected by hand (GUI) before processing.
    t, channels: time (us) and raw channels (V) of the breakdown shot, as `read_shot`/`load_shot` give them.
    """
    u = voltage_kv(t, channels["C3"], offset_window_end(t, channels["C3"]))
    up = (polarity * u).max()  # with polarity, so it is also right for negative shots
    amplitude = breakdown_amplitude(up, w.up * float(level) / float(w.level))
    tau_b, _, _ = match(t, u, w.t_sync, amplitude * w.u / w.up, polarity)  # reference scaled to the amplitude
    return float(amplitude), float(tau_b)


def process_breakdown(
    t: np.ndarray, channels: dict[str, np.ndarray], level: str, shot: int, polarity: int, w: Withstand,
    amplitude: float, tau_b: float,
) -> Breakdown:
    """Phase B and Phase 3-4 for a given amplitude and tau_B (from `align_breakdown`): I_exc, Q_exc, E_exc and the events.

    t, channels: time (us) and raw channels (V) of the breakdown shot, as `read_shot`/`load_shot` give them.
    """
    # Check (kept for verification, not run): time axis of the shot.
    # print("\nSamples:          ", len(t))
    # print("First time (us):  ", t[0])
    # print("Last time (us):   ", t[-1])
    # print("Time step (us):   ", t[1] - t[0])

    t_end = offset_window_end(t, channels["C3"])  # one offset window per shot, from C3, for all channels
    u = voltage_kv(t, channels["C3"], t_end)
    # Check (kept for verification, not run): quietness of the C3 offset window.
    # mean, std, halves = baseline_stats(t, channels["C3"], t_end)
    # print(f"\nOffset window end (us): {t_end:.3f}")
    # print(f"C3 baseline: mean {mean:.4f} V, std {std:.4f} V, half difference {halves:+.4f} V")

    up = (polarity * u).max()  # with polarity, so it is also right for negative shots
    u_ref = amplitude * w.u / w.up  # the withstand reference scaled to the amplitude, as in align_breakdown
    try:
        t1 = front_time(t, polarity * u, amplitude)  # expected peak if chopped on the front
    except ValueError:  # breakdown before 90 % of the peak
        t1 = None
    t_sync = t - tau_b  # align the measurement on the withstand t_sync axis

    # Currents: same offset window as C3 (t_end), charge from the onset.
    onset = find_onset(t, channels["C3"])
    currents, charges = {}, {}
    for name in ("C4", "C2"):
        currents[name] = polarity * current_a(t, channels[name], t_end)
        charges[name] = charge_uc(t, currents[name], onset)
        # Check (kept for verification, not run): offset window of the current channel and charge at a fixed point.
        # mean, std, halves = baseline_stats(t, channels[name], t_end)
        # k = np.searchsorted(t, onset + CHARGE_AT_US)
        # print(f"{name} baseline: mean {mean * 100:+.4f} A, std {std * 100:.4f} A, "
        #       f"half difference {halves * 100:+.4f} A")
        # print(f"{name} peak (A): {currents[name].max():.2f}, charge {CHARGE_AT_US:g} us after onset (uC): {charges[name][k]:.3f}")

    # Phase 3, step 1: capacitive charge Q_g = C_g * U_ref and discharge charge Q_exc = Q - Q_g, on the same time.
    # C4 and its own C_g (same channel). u_ref is interpolated on the measured samples (t_sync), np.interp keeps its edge values outside.
    q_g = w.c_g * np.interp(t_sync, w.t_sync, u_ref)  # nF * kV = uC
    q_exc = charges["C4"] - q_g
    # Phase 3, step 2: capacitive current I_c = C_g * dU_ref/dt (Savitzky-Golay on the reference), I_exc = I - I_c.
    # The derivative is taken on the reference's own grid, then interpolated like u_ref above.
    i_c = np.interp(t_sync, w.t_sync, capacitive_current_a(w.t_sync, u_ref, w.c_g))
    i_exc = currents["C4"] - i_c
    # Phase 4, step 1: discharge energy E_exc = ∫ U * I_exc dt from the onset (U with polarity applied, so it is positive).
    e_exc = energy_mj(t, polarity * u, i_exc, onset)
    # Check (kept for verification, not run): charges and energy at a fixed point after the onset.
    # k = np.searchsorted(t, onset + CHARGE_AT_US)
    # print(f"onset + {CHARGE_AT_US:g} us:  Q {charges['C4'][k]:.3f}, Q_g {q_g[k]:.3f}, Q_exc {q_exc[k]:+.3f} uC")
    # print(f"E_exc onset + {CHARGE_AT_US:g} us: {e_exc[k]:.2f} mJ")

    # Phase 4, step 2: events fj (final jump, before Ip1), Ip1 (first peak of I_exc) and Ip2 (second peak,
    # only if there is one), with U, I_exc, Q_exc, E_exc at each.
    k_pk = current_peaks(i_exc)
    times = {"fj": t_final_jump(t_sync, i_exc, k_pk[0]), "Ip1": float(t_sync[k_pk[0]])}
    if len(k_pk) > 1:
        times["Ip2"] = float(t_sync[k_pk[1]])
    # li (leader inception) on C2, lower noise than C4, same I_c; search up to fj or C2 near its ceiling, whichever is first.
    i_c2 = currents["C2"]
    k_end = min(int(np.searchsorted(t_sync, times["fj"])), int(np.argmax(i_c2 >= LI_C2_CEILING * i_c2.max())))
    t_li = t_steady_rise(t_sync, i_c2 - i_c, k_end)
    if t_li is not None:  # None if the rise already runs at t_sync = 0
        times["li"] = t_li
    events = {}
    for name, te in times.items():
        at = lambda x: float(np.interp(te, t_sync, x))  # value of a waveform at the event time
        events[name] = {"t": float(te), "u": at(polarity * u), "i_exc": at(i_exc), "q_exc": at(q_exc), "e_exc": at(e_exc)}

    waveforms = {
        "t_sync": t_sync, "u": polarity * u,
        "i_c4": currents["C4"], "i_c2": currents["C2"], "i_c": i_c, "i_exc": i_exc,
        "q_c4": charges["C4"], "q_c2": charges["C2"], "q_g": q_g, "q_exc": q_exc,
        "e_exc": e_exc,
    }
    return Breakdown(level, shot, float(up), float(amplitude), float(tau_b), t1, events, waveforms)
