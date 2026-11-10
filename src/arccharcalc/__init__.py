import sys
from pathlib import Path

from arccharcalc.generator import double_exp_pu
from arccharcalc.io_lecroy import read_shot
import numpy as np

from arccharcalc.plotting import plot_waveform
from arccharcalc.processing import (
    baseline_stats,
    charge_uc,
    current_a,
    find_onset,
    offset_window_end,
    voltage_kv,
)
from arccharcalc.reference import breakdown_amplitude, match

# Charge is printed this long after the onset (a fixed point, only for checks).
CHARGE_AT_US = 10.0


def main() -> None:
    # usage: arccharcalc <level folder> <level> <shot> [polarity [<withstand folder> <withstand level> <withstand shot>]]
    # e.g. ".../15.5kV" 15.5 3 +1 ".../11kV (Withstand Voltage)" 11 0

    if len(sys.argv) < 4:
        raise ValueError("No breakdown shot given: usage: arccharcalc <level folder> <level> <shot> [polarity [<withstand folder> <withstand level> <withstand shot>]]")
    if len(sys.argv) < 8:  # argv[5..7]: withstand folder, level, shot
        raise ValueError("No reference shot given: usage: arccharcalc <level folder> <level> <shot> [polarity [<withstand folder> <withstand level> <withstand shot>]]")
    
    polarity = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    w_dir, w_level, w_shot = Path(sys.argv[5]), sys.argv[6], int(sys.argv[7])
    t_w, ch_w = read_shot(w_dir, w_level, w_shot)
    
    # Phase A: create the reference from the withstand shot, on its own t_sync axis (Phase A).
    u_w = polarity * voltage_kv(t_w, ch_w["C3"], offset_window_end(t_w, ch_w["C3"]))
    up_w = u_w.max()
    double_exp_ref = up_w * double_exp_pu(t_w)
    tau_ref, _, _ = match(t_w, u_w, t_w, double_exp_ref, 1)  # u_w is already positive

    print(f"\n Withstand shot: {w_level}kV/{w_shot}, Up (kV): {up_w:.3f}, tau_A (us): {tau_ref:+.4f}")
    
    t_w_sync = t_w - tau_ref  # align the withstand on its own t_sync axis
    plot_waveform(t_w_sync, u_w, t_w, double_exp_ref, f"Withstand shot {w_level}kV, shot: {w_shot}, Up (kV): {up_w:.3f}, tau_A (us): {tau_ref:+.4f}, in reference to a perfect double exponential", label_1="Withstand", label_2="Double exponential")



    #Phase B: match the measurement to the withstand reference, scaled to the measured peak.
    level_dir, level, shot = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    t, channels = read_shot(level_dir, level, shot)

    print("\nSamples:          ", len(t))
    print("First time (us):  ", t[0])
    print("Last time (us):   ", t[-1])
    print("Time step (us):   ", t[1] - t[0])


    mean, std, halves = baseline_stats(t, channels["C3"], offset_window_end(t, channels["C3"]))
    u = voltage_kv(t, channels["C3"], offset_window_end(t, channels["C3"]))
    print(f"\nOffset window end (us): {offset_window_end(t, channels["C3"]):.3f}")
    print(f"C3 baseline: mean {mean:.4f} V, std {std:.4f} V, half difference {halves:+.4f} V")
    print(f"Up (kV): {u.max():.3f}")


    up = (polarity * u).max() 
    up_expected = up_w * float(level) / float(w_level)
    amplitude = breakdown_amplitude(up, up_expected)
    #rule = "measured peak" if amplitude == up else "expected peak (breakdown on the front)"
    u_ref = amplitude * u_w / up_w  # scale the withstand reference to the measured peak
    tau_b, _, _ = match(t, u, t_w_sync, u_ref, polarity)
    print (f"\nBreakdown shot: {level}kV/{shot}, Up (kV): {up:.3f}, tau_B (us): {tau_b:+.4f}, in reference to a Withstand shot {w_level}kV/{w_shot}")
    t_sync = t - tau_b  # align the measurement on the withstand t_sync axis
    plot_waveform(t_sync, polarity * u, t_w_sync, u_ref, f"Breakdown shot {level}kV, shot: {shot}, Up (kV): {up:.3f}, tau_b (us): {tau_b:+.4f}, in reference to a Withstand shot {w_level}kV/{w_shot}", label_1="Breakdown", label_2="Withstand")

    """ # Current channels use the same offset window as C3; charge starts at the onset.
    onset = find_onset(t, channels["C3"])
    k = np.searchsorted(t, onset + CHARGE_AT_US)
    for name in ("C4", "C2"):
        mean, std, halves = baseline_stats(t, channels[name], offset_window_end(t, channels[name]))
        i = current_a(t, channels[name], offset_window_end(t, channels[name]))
        q = charge_uc(t, i, onset)
        print(f"{name} baseline: mean {mean * 100:+.4f} A, std {std * 100:.4f} A, "
              f"half difference {halves * 100:+.4f} A")
        print(f"{name} peak (A): {i.max():.2f}, charge {CHARGE_AT_US:g} us after onset (uC): {q[k]:.3f}") """
    


