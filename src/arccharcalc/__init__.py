import sys
from pathlib import Path

from arccharcalc.io_lecroy import read_shot
import numpy as np

from arccharcalc.processing import (
    baseline_stats,
    charge_uc,
    current_a,
    find_onset,
    offset_window_end,
    voltage_kv,
)

# Charge is printed this long after the onset (a fixed point, only for checks).
CHARGE_AT_US = 10.0


def main() -> None:
    # usage: arccharcalc <level folder> <level> <shot>, e.g. ".../15.5kV" 15.5 3
    level_dir, level, shot = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    t, channels = read_shot(level_dir, level, shot)

    print("Samples:          ", len(t))
    print("First time (us):  ", t[0])
    print("Last time (us):   ", t[-1])
    print("Time step (us):   ", t[1] - t[0])

    t_end = offset_window_end(t, channels["C3"])
    mean, std, halves = baseline_stats(t, channels["C3"], t_end)
    u = voltage_kv(t, channels["C3"], t_end)
    print(f"Offset window end (us): {t_end:.3f}")
    print(f"C3 baseline: mean {mean:.4f} V, std {std:.4f} V, half difference {halves:+.4f} V")
    print(f"Up (kV): {u.max():.3f}")

    # Current channels use the same offset window as C3; charge starts at the onset.
    onset = find_onset(t, channels["C3"])
    k = np.searchsorted(t, onset + CHARGE_AT_US)
    for name in ("C4", "C2"):
        mean, std, halves = baseline_stats(t, channels[name], t_end)
        i = current_a(t, channels[name], t_end)
        q = charge_uc(t, i, onset)
        print(f"{name} baseline: mean {mean * 100:+.4f} A, std {std * 100:.4f} A, "
              f"half difference {halves * 100:+.4f} A")
        print(f"{name} peak (A): {i.max():.2f}, charge {CHARGE_AT_US:g} us after onset (uC): {q[k]:.3f}")

