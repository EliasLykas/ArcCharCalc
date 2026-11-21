"""Reading LeCroy WaveRunner oscilloscope CSV files."""

from datetime import datetime
from pathlib import Path

import numpy as np


def read_lecroy_header(path: Path) -> tuple[int, str]:
    """Read the segment size and the trigger time from the 4 header lines of a LeCroy CSV.

    Line 2 is like `Segments,1,SegmentSize,50001`, line 4 like `#1,15-Apr-2024 20:41:44,0`.

    Returns (n_samples, trig_time): the number of samples and the trigger time as
    ISO text "YYYY-MM-DD HH:MM:SS" (sorts correctly as text).
    """
    with open(path, encoding="ascii") as f:
        lines = [f.readline() for _ in range(4)]
    n_samples = int(lines[1].split(",")[3])
    trig = datetime.strptime(lines[3].split(",")[1], "%d-%b-%Y %H:%M:%S")
    return n_samples, trig.isoformat(sep=" ")


def read_lecroy_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one LeCroy CSV file.

    The file has 4 header lines and a `Time,Ampl` row (5 lines skipped), then
    two comma-separated columns: time (s in the file) and amplitude (V).

    Returns (t, v): time in microseconds and amplitude in volts, as numpy arrays.
    """
    t, v = np.loadtxt(path, delimiter=",", skiprows=5, unpack=True)
    return t*1e6, v


def read_shot(level_dir: Path, level: str, shot: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read channels C2, C3 and C4 of one shot.

    Files are named like `C3_15.5_00003.csv` inside the folder of the charging
    level (e.g. `15.5kV/`, or `11kV (Withstand Voltage)/` with level "11"):
    `level` is the "15.5" part, `shot` the oscilloscope number (counted from 0).

    The channels of one shot can differ by one sample (50001 vs 50002, e.g. in
    the 11 kV shot), so all are cut to the shortest length.

    Returns (t, channels): the common time axis in microseconds and a dict
    {"C2": ..., "C3": ..., "C4": ...} of amplitudes in volts.
    """
    raw = {
        name: read_lecroy_csv(level_dir / f"{name}_{level}_{shot:05d}.csv")
        for name in ("C2", "C3", "C4")
    }
    n = min(len(t_ch) for t_ch, _ in raw.values())
    t = raw["C2"][0][:n]
    channels = {}
    for name, (t_ch, v) in raw.items():
        if not np.array_equal(t, t_ch[:n]):
            raise ValueError(f"Time axis of {name} differs from the other channels")
        channels[name] = v[:n]
    return t, channels