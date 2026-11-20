"""The processing chain of one withstand shot and of one breakdown shot (moved out of main(), no change in the numbers)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arccharcalc.features import cg_qv, first_peak, front_time, t_final_jump
from arccharcalc.generator import double_exp_pu
from arccharcalc.io_lecroy import read_shot
from arccharcalc.plotting import plot_current_voltage, plot_events, plot_waveform
from arccharcalc.processing import (
    baseline_stats,
    capacitive_current_a,
    charge_uc,
    current_a,
    energy_mj,
    find_onset,
    offset_window_end,
    voltage_kv,
)
from arccharcalc.reference import breakdown_amplitude, match

# Charge is printed this long after the onset (a fixed point, only for checks).
CHARGE_AT_US = 10.0


@dataclass(frozen=True)
class Withstand:
    """What the breakdown shots need from the withstand shot (Phase A result)."""

    level: str  # kV, as in the file name
    shot: int
    t_sync: np.ndarray  # μs, t_w - tau_A
    u: np.ndarray  # kV, positive (polarity applied)
    up: float  # kV, u.max()
    c_g: dict[str, float]  # nF, per current channel ("C2", "C4")


def process_withstand(w_dir: Path, w_level: str, w_shot: int, polarity: int) -> Withstand:
    """Phase A: align the withstand shot on the double exponential (tau_A) and get C_g from its front."""
    t_w, ch_w = read_shot(w_dir, w_level, w_shot)

    # Phase A: create the reference from the withstand shot, on its own t_sync axis (Phase A).
    t_end_w = offset_window_end(t_w, ch_w["C3"])  # one offset window for all withstand channels
    u_w = polarity * voltage_kv(t_w, ch_w["C3"], t_end_w)
    up_w = u_w.max()
    double_exp_ref = up_w * double_exp_pu(t_w)
    tau_ref, _, _ = match(t_w, u_w, t_w, double_exp_ref, 1)  # u_w is already positive

    print(f"\n Withstand shot: {w_level}kV/{w_shot}, Up (kV): {up_w:.3f}, tau_A (us): {tau_ref:+.4f}")
    print(f"Front time T1 (us): withstand {front_time(t_w, u_w, up_w):.3f}, "
          f"double exponential {front_time(t_w, double_exp_ref, up_w):.3f}")

    # Geometric capacitance C_g: Q = C_g * U on the withstand front, from the onset to the first 90 % of Up.
    onset_w = find_onset(t_w, ch_w["C3"])
    k90 = np.argmax(u_w >= 0.9 * up_w)  # first sample at 90 %
    front_w = (t_w >= onset_w) & (np.arange(len(t_w)) <= k90)  # boolean mask of the fit window
    c_g = {}  # nF, per channel
    for name in ("C2", "C4"):  # C2: fine scale, does not clip here (3.35 A); C4 for comparison
        q_w = charge_uc(t_w, polarity * current_a(t_w, ch_w[name], t_end_w), onset_w)
        cg, cg_se = cg_qv(u_w[front_w], q_w[front_w])
        c_g[name] = cg
        slope, intercept = np.polyfit(u_w[front_w], q_w[front_w], 1)  # free intercept, as a check
        print(f"C_g from {name} (pF): {cg * 1000:.2f} +/- {cg_se * 1000:.2f}, "
              f"free intercept: {slope * 1000:.2f} pF, intercept {intercept:+.4f} uC")

    t_w_sync = t_w - tau_ref  # align the withstand on its own t_sync axis
    plot_waveform(t_w_sync, u_w, t_w, double_exp_ref, f"Withstand shot {w_level}kV, shot: {w_shot}, Up (kV): {up_w:.3f}, tau_A (us): {tau_ref:+.4f}, in reference to a perfect double exponential", label_1="Withstand", label_2="Double exponential")
    return Withstand(w_level, w_shot, t_w_sync, u_w, up_w, c_g)


def process_breakdown(level_dir: Path, level: str, shot: int, polarity: int, w: Withstand) -> None:
    """Phase B and Phase 3-4: align the shot on the withstand (tau_B), get I_exc, Q_exc, E_exc and the events."""
    t, channels = read_shot(level_dir, level, shot)

    print("\nSamples:          ", len(t))
    print("First time (us):  ", t[0])
    print("Last time (us):   ", t[-1])
    print("Time step (us):   ", t[1] - t[0])

    t_end = offset_window_end(t, channels["C3"])  # one offset window per shot, from C3, for all channels
    mean, std, halves = baseline_stats(t, channels["C3"], t_end)
    u = voltage_kv(t, channels["C3"], t_end)
    print(f"\nOffset window end (us): {t_end:.3f}")
    print(f"C3 baseline: mean {mean:.4f} V, std {std:.4f} V, half difference {halves:+.4f} V")
    print(f"Up (kV): {(polarity * u).max():.3f}")  # with polarity, so it is also right for negative shots

    # Phase B: match the measurement to the withstand reference, scaled to the measured peak.
    up = (polarity * u).max()
    up_expected = w.up * float(level) / float(w.level)
    amplitude = breakdown_amplitude(up, up_expected)
    u_ref = amplitude * w.u / w.up  # scale the withstand reference to the measured peak
    tau_b, _, _ = match(t, u, w.t_sync, u_ref, polarity)
    print(f"\nBreakdown shot: {level}kV/{shot}, Up (kV): {up:.3f}, tau_B (us): {tau_b:+.4f}, in reference to a Withstand shot {w.level}kV/{w.shot}")
    try:
        print(f"Front time T1 (us): {front_time(t, polarity * u, amplitude):.3f}")  # expected peak if chopped on the front
    except ValueError:  # breakdown before 90 % of the peak
        print("Front time T1 (us): - (front does not reach 90 %)")
    t_sync = t - tau_b  # align the measurement on the withstand t_sync axis
    plot_waveform(t_sync, polarity * u, w.t_sync, u_ref, f"Breakdown shot {level}kV, shot: {shot}, Up (kV): {up:.3f}, tau_b (us): {tau_b:+.4f}, in reference to a Withstand shot {w.level}kV/{w.shot}", label_1="Breakdown", label_2="Withstand")

    # Currents: same offset window as C3 (t_end), charge from the onset, plotted on t_sync.
    onset = find_onset(t, channels["C3"])
    k = np.searchsorted(t, onset + CHARGE_AT_US)
    currents, charges = {}, {}
    for name in ("C4", "C2"):
        mean, std, halves = baseline_stats(t, channels[name], t_end)
        currents[name] = polarity * current_a(t, channels[name], t_end)
        q = charges[name] = charge_uc(t, currents[name], onset)
        print(f"{name} baseline: mean {mean * 100:+.4f} A, std {std * 100:.4f} A, "
              f"half difference {halves * 100:+.4f} A")
        print(f"{name} peak (A): {currents[name].max():.2f}, charge {CHARGE_AT_US:g} us after onset (uC): {q[k]:.3f}")
    plot_waveform(t_sync, currents["C4"], t_sync, currents["C2"], f"Currents, breakdown shot {level}kV/{shot}, on t_sync",
                  label_1="C4 (coarse)", label_2="C2 (fine)", ylabel="Current (A)")
    plot_current_voltage(t_sync, currents["C4"], polarity * u, f"Breakdown shot {level}kV/{shot}, on t_sync",
                         i_label="Current C4", u_label="Voltage")

    # Phase 3, step 1: capacitive charge Q_g = C_g * U_ref and discharge charge Q_exc = Q - Q_g, on the same time.
    # C4 and its own C_g (same channel). u_ref is interpolated on the measured samples (t_sync), np.interp keeps its edge values outside.
    q_g = w.c_g["C4"] * np.interp(t_sync, w.t_sync, u_ref)  # nF * kV = uC
    q_exc = charges["C4"] - q_g
    k0 = np.searchsorted(t_sync, 0.0)  # sample at t_sync = 0
    print(f"\nC_g (C4) = {w.c_g['C4'] * 1000:.2f} pF")
    print(f"t_sync = 0:  Q {charges['C4'][k0]:.3f}, Q_g {q_g[k0]:.3f}, Q_exc {q_exc[k0]:+.3f} uC")
    print(f"onset + {CHARGE_AT_US:g} us:  Q {charges['C4'][k]:.3f}, Q_g {q_g[k]:.3f}, Q_exc {q_exc[k]:+.3f} uC")
    # Phase 3, step 2: capacitive current I_c = C_g * dU_ref/dt (Savitzky-Golay on the reference), I_exc = I - I_c.
    # The derivative is taken on the reference's own grid, then interpolated like u_ref above.
    i_c = np.interp(t_sync, w.t_sync, capacitive_current_a(w.t_sync, u_ref, w.c_g["C4"]))
    i_exc = currents["C4"] - i_c
    print(f"t_sync = 0:  I {currents['C4'][k0]:+.3f}, I_c {i_c[k0]:+.3f}, I_exc {i_exc[k0]:+.3f} A;  peak I_c {i_c.max():.2f} A")
    # Phase 4, step 1: discharge energy E_exc = ∫ U * I_exc dt from the onset (U with polarity applied, so it is positive).
    e_exc = energy_mj(t, polarity * u, i_exc, onset)
    print(f"E_exc at t_sync = 0: {e_exc[k0]:+.4f} mJ;  onset + {CHARGE_AT_US:g} us: {e_exc[k]:.2f} mJ")
    # Phase 4, step 2: events fj (final jump) and Ip1 (flashover), with U, I_exc, Q_exc, E_exc at each.
    k_pk = first_peak(i_exc)
    events = {"fj": t_final_jump(t_sync, i_exc, k_pk), "Ip1": float(t_sync[k_pk])}
    for name, te in events.items():
        at = lambda x: np.interp(te, t_sync, x)  # value of a waveform at the event time
        print(f"{name:3s} t_sync {te:7.3f} us: U {at(polarity * u):6.1f} kV, I_exc {at(i_exc):6.1f} A, "
              f"Q_exc {at(q_exc):7.2f} uC, E_exc {at(e_exc):8.1f} mJ")
    plot_events(t_sync, polarity * u, i_exc, q_exc, e_exc, events,
                f"Events, breakdown shot {level}kV/{shot}, on t_sync",
                window=(min(events.values()) - 0.5, max(events.values()) + 1.0))
    plot_waveform(t_sync, i_exc, t_sync, i_c, f"Discharge current, breakdown shot {level}kV/{shot}, on t_sync",
                  label_1="I_exc = I - I_c", label_2="I_c = C_g * dU_ref/dt", ylabel="Current (A)")
    plot_waveform(t_sync, charges["C4"], t_sync, q_g, f"Charge, breakdown shot {level}kV/{shot}, on t_sync",
                  label_1="Q (C4)", label_2="Q_g = C_g * U_ref", ylabel="Charge (uC)")
