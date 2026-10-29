import sys
from pathlib import Path

from arccharcalc.io_lecroy import read_shot
from arccharcalc.processing import (
    baseline_stats,
    offset_window_end,
    voltage_kv,
)


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

