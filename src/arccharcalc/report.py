"""Print and plot the results of one run, from what is stored in the database (see `db.load_withstand`, `db.load_breakdowns`)."""

import matplotlib.pyplot as plt
import numpy as np

from arccharcalc.features import front_time
from arccharcalc.generator import double_exp_pu
from arccharcalc.pipeline import Breakdown, Withstand
from arccharcalc.plotting import plot_current_voltage, plot_events, plot_waveform


def print_run(w: Withstand, breakdowns: list[Breakdown]) -> None:
    """Print the withstand result (Phase A) and, per breakdown shot, Up, tau_B, T1, values at t_sync = 0 and the events."""
    t_w = w.t_sync + w.tau  # back to the withstand's own time axis
    print(f"\n Withstand shot: {w.level}kV/{w.shot}, Up (kV): {w.up:.3f}, tau_A (us): {w.tau:+.4f}")
    print(f"Front time T1 (us): withstand {w.t1:.3f}, "
          f"double exponential {front_time(t_w, w.up * double_exp_pu(t_w), w.up):.3f}")

    for b in breakdowns:
        wf = b.waveforms
        print(f"\nBreakdown shot: {b.level}kV/{b.shot}, Up (kV): {b.up:.3f}, tau_B (us): {b.tau:+.4f}, "
              f"in reference to a Withstand shot {w.level}kV/{w.shot}")
        print("Front time T1 (us): " + ("- (front does not reach 90 %)" if b.t1 is None else f"{b.t1:.3f}"))
        print(f"C4 peak (A): {wf['i_c4'].max():.2f}")
        print(f"C2 peak (A): {wf['i_c2'].max():.2f}")

        k0 = np.searchsorted(wf["t_sync"], 0.0)  # sample at t_sync = 0
        print(f"\nC_g (C4) = {w.c_g['C4'] * 1000:.2f} pF")
        print(f"t_sync = 0:  Q {wf['q_c4'][k0]:.3f}, Q_g {wf['q_g'][k0]:.3f}, Q_exc {wf['q_exc'][k0]:+.3f} uC")
        print(f"t_sync = 0:  I {wf['i_c4'][k0]:+.3f}, I_c {wf['i_c'][k0]:+.3f}, I_exc {wf['i_exc'][k0]:+.3f} A;  "
              f"peak I_c {wf['i_c'].max():.2f} A")
        print(f"E_exc at t_sync = 0: {wf['e_exc'][k0]:+.4f} mJ")
        for name, ev in b.events.items():
            # Q and Q_g are not stored per event: interpolated on the stored waveforms, like the event values themselves.
            q, q_g = (float(np.interp(ev["t"], wf["t_sync"], wf[x])) for x in ("q_c4", "q_g"))
            print(f"{name:3s} t_sync {ev['t']:7.3f} us: U {ev['u']:6.1f} kV, I_exc {ev['i_exc']:6.1f} A, "
                  f"Q {q:7.2f} uC, Q_g {q_g:5.2f} uC, Q_exc {ev['q_exc']:7.2f} uC, E_exc {ev['e_exc']:8.1f} mJ")


def plot_run(w: Withstand, breakdowns: list[Breakdown]) -> None:
    """Plot the withstand against the double exponential and, per breakdown shot, the six figures of Phases B, 3 and 4.

    The references are rebuilt from the stored results: double exponential = Up_w * double_exp_pu(t_w),
    u_ref = amplitude * u_w / Up_w. All windows are shown together at the end (the call blocks until they are closed).
    """
    t_w = w.t_sync + w.tau
    plot_waveform(w.t_sync, w.u, t_w, w.up * double_exp_pu(t_w),
                  f"Withstand shot {w.level}kV, shot: {w.shot}, Up (kV): {w.up:.3f}, tau_A (us): {w.tau:+.4f}, "
                  "in reference to a perfect double exponential", label_1="Withstand", label_2="Double exponential")

    for b in breakdowns:
        wf, name = b.waveforms, f"breakdown shot {b.level}kV/{b.shot}"
        t_sync = wf["t_sync"]
        u_ref = b.amplitude * w.u / w.up  # the withstand reference scaled as in process_breakdown
        plot_waveform(t_sync, wf["u"], w.t_sync, u_ref,
                      f"Breakdown shot {b.level}kV, shot: {b.shot}, Up (kV): {b.up:.3f}, tau_b (us): {b.tau:+.4f}, "
                      f"in reference to a Withstand shot {w.level}kV/{w.shot}", label_1="Breakdown", label_2="Withstand")
        plot_waveform(t_sync, wf["i_c4"], t_sync, wf["i_c2"], f"Currents, {name}, on t_sync",
                      label_1="C4 (coarse)", label_2="C2 (fine)", ylabel="Current (A)")
        plot_current_voltage(t_sync, wf["i_c4"], wf["u"], f"Breakdown shot {b.level}kV/{b.shot}, on t_sync",
                             i_label="Current C4", u_label="Voltage")
        times = {ev_name: ev["t"] for ev_name, ev in b.events.items()}
        plot_events(t_sync, wf["u"], wf["i_exc"], wf["q_exc"], wf["e_exc"], times, f"Events, {name}, on t_sync",
                    window=(min(times.values()) - 0.5, max(times.values()) + 1.0))
        plot_waveform(t_sync, wf["i_exc"], t_sync, wf["i_c"], f"Discharge current, {name}, on t_sync",
                      label_1="I_exc = I - I_c", label_2="I_c = C_g * dU_ref/dt", ylabel="Current (A)")
        plot_waveform(t_sync, wf["q_c4"], t_sync, wf["q_g"], f"Charge, {name}, on t_sync",
                      label_1="Q (C4)", label_2="Q_g = C_g * U_ref", ylabel="Charge (uC)")
    plt.show()
