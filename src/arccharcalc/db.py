"""SQLite database with a copy of the oscilloscope shots and the results of every run, see plan/ADR-001_database.md."""

import hashlib
import re
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from arccharcalc.io_lecroy import read_lecroy_header, read_shot
from arccharcalc.pipeline import Breakdown, Withstand

# Waveforms are stored as raw bytes of little-endian float64, exactly the numbers read_shot gives.
# shortcut: float64 ~0.4 MB per channel and shot; if the file grows past ~10 GB, use float32/zlib or HDF5.
BLOB_DTYPE = "<f8"

SCHEMA = """
CREATE TABLE IF NOT EXISTS campaign (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,      -- campaign folder name, e.g. 2024-04-15_56-1-2sheds_LI-1.2-50_pos
    polarity   INTEGER NOT NULL CHECK (polarity IN (-1, 1)),
    source_dir TEXT NOT NULL              -- where the raw CSVs were read from
);
CREATE TABLE IF NOT EXISTS shot (
    id           INTEGER PRIMARY KEY,
    campaign_id  INTEGER NOT NULL REFERENCES campaign(id),
    level        TEXT NOT NULL,           -- as in the file name, e.g. "15.5"
    number       INTEGER NOT NULL,        -- oscilloscope number, counted from 0
    is_withstand INTEGER NOT NULL CHECK (is_withstand IN (0, 1)),
    trig_time    TEXT NOT NULL,           -- "YYYY-MM-DD HH:MM:SS", from the C3 header
    n_c2 INTEGER NOT NULL, n_c3 INTEGER NOT NULL, n_c4 INTEGER NOT NULL,  -- samples per file, before the cut
    sha256       TEXT NOT NULL,           -- of the C2, C3, C4 files together
    t  BLOB NOT NULL,                     -- us, common time axis (read_shot)
    c2 BLOB NOT NULL, c3 BLOB NOT NULL, c4 BLOB NOT NULL,  -- V, as recorded (no offset, no scaling)
    UNIQUE (campaign_id, level, number)
);
CREATE TABLE IF NOT EXISTS run (
    id                INTEGER PRIMARY KEY,
    started           TEXT NOT NULL,      -- local time "YYYY-MM-DD HH:MM:SS"
    git_commit        TEXT NOT NULL,      -- code version; "-dirty" = uncommitted changes, "unknown" = no git
    campaign_id       INTEGER NOT NULL REFERENCES campaign(id),
    withstand_shot_id INTEGER NOT NULL REFERENCES shot(id)
);
CREATE TABLE IF NOT EXISTS shot_result (  -- one row per run and shot (withstand and breakdown)
    run_id       INTEGER NOT NULL REFERENCES run(id),
    shot_id      INTEGER NOT NULL REFERENCES shot(id),
    up_kv        REAL NOT NULL,
    t1_us        REAL,                    -- NULL if the front does not reach 90 %
    tau_us       REAL NOT NULL,           -- tau_A (withstand) or tau_B (breakdown)
    amplitude_kv REAL,                    -- breakdown only: peak of the scaled reference
    cg_c2_nf     REAL, cg_c4_nf REAL,     -- withstand only: C_g from the Q-V fit
    PRIMARY KEY (run_id, shot_id)
);
CREATE TABLE IF NOT EXISTS event (        -- fj, Ip1, ... of a breakdown shot
    run_id   INTEGER NOT NULL REFERENCES run(id),
    shot_id  INTEGER NOT NULL REFERENCES shot(id),
    name     TEXT NOT NULL,
    t_us     REAL NOT NULL,               -- on t_sync
    u_kv     REAL NOT NULL,
    i_exc_a  REAL NOT NULL,
    q_exc_uc REAL NOT NULL,
    e_exc_mj REAL NOT NULL,
    PRIMARY KEY (run_id, shot_id, name)
);
CREATE TABLE IF NOT EXISTS waveform (     -- computed waveforms, kept per run during development
    run_id  INTEGER NOT NULL REFERENCES run(id),
    shot_id INTEGER NOT NULL REFERENCES shot(id),
    name    TEXT NOT NULL,                -- e.g. t_sync, u, i_exc (units: see Withstand / Breakdown in pipeline.py)
    data    BLOB NOT NULL,
    PRIMARY KEY (run_id, shot_id, name)
);
"""

CHANNELS = ("C2", "C3", "C4")


def connect(path: Path) -> sqlite3.Connection:
    """Open (or create) the database file and make sure the tables exist."""
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


def _blob(x: np.ndarray) -> bytes:
    """Bytes of a waveform, in the one dtype the database uses."""
    return np.asarray(x, dtype=BLOB_DTYPE).tobytes()


def _sha256(paths: list[Path]) -> str:
    """One SHA-256 over the contents of several files, in the given order."""
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()


def _shot_id(con: sqlite3.Connection, campaign: str, level: str, number: int) -> int:
    """Database id of one shot. ValueError if it is not in the database."""
    row = con.execute(
        "SELECT s.id FROM shot s JOIN campaign c ON c.id = s.campaign_id WHERE c.name = ? AND s.level = ? AND s.number = ?",
        (campaign, level, number),
    ).fetchone()
    if row is None:
        raise ValueError(f"Shot {level}/{number:05d} of campaign {campaign} is not in the database")
    return row[0]


def _git_commit() -> str:
    """Commit of the code that runs now, with "-dirty" if there are uncommitted changes ("unknown" without git)."""
    code_dir = Path(__file__).parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=code_dir,
                                capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=code_dir,
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # git not installed, or not a git repository
        return "unknown"
    return commit + ("-dirty" if status else "")


def start_run(con: sqlite3.Connection, campaign: str, w_level: str, w_shot: int) -> int:
    """Insert a new run (time, code version, withstand shot) and return its id."""
    w_id = _shot_id(con, campaign, w_level, w_shot)
    started = datetime.now().isoformat(sep=" ", timespec="seconds")
    return con.execute(
        "INSERT INTO run (started, git_commit, campaign_id, withstand_shot_id)"
        " SELECT ?, ?, campaign_id, id FROM shot WHERE id = ?",
        (started, _git_commit(), w_id),
    ).lastrowid


def _save_waveforms(con: sqlite3.Connection, run_id: int, shot_id: int, waveforms: dict[str, np.ndarray]) -> None:
    """Store computed waveforms of one shot in one run."""
    con.executemany(
        "INSERT INTO waveform (run_id, shot_id, name, data) VALUES (?, ?, ?, ?)",
        [(run_id, shot_id, name, _blob(x)) for name, x in waveforms.items()],
    )


def save_withstand(con: sqlite3.Connection, run_id: int, campaign: str, w: Withstand) -> None:
    """Store the Phase A results of the withstand shot: Up, T1, tau_A, C_g and the waveforms t_sync, u."""
    shot_id = _shot_id(con, campaign, w.level, w.shot)
    con.execute(
        "INSERT INTO shot_result (run_id, shot_id, up_kv, t1_us, tau_us, cg_c2_nf, cg_c4_nf) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, shot_id, w.up, w.t1, w.tau, None, w.c_g),  # cg_c2_nf NULL: C_g only from C4 since 2026-11-24
    )
    _save_waveforms(con, run_id, shot_id, {"t_sync": w.t_sync, "u": w.u})


def save_breakdown(con: sqlite3.Connection, run_id: int, campaign: str, b: Breakdown) -> None:
    """Store the results of one breakdown shot: Up, T1, tau_B, amplitude, the events and all waveforms."""
    shot_id = _shot_id(con, campaign, b.level, b.shot)
    con.execute(
        "INSERT INTO shot_result (run_id, shot_id, up_kv, t1_us, tau_us, amplitude_kv) VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, shot_id, b.up, b.t1, b.tau, b.amplitude),
    )
    con.executemany(
        "INSERT INTO event (run_id, shot_id, name, t_us, u_kv, i_exc_a, q_exc_uc, e_exc_mj) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [(run_id, shot_id, name, ev["t"], ev["u"], ev["i_exc"], ev["q_exc"], ev["e_exc"]) for name, ev in b.events.items()],
    )
    _save_waveforms(con, run_id, shot_id, b.waveforms)


def _load_waveforms(con: sqlite3.Connection, run_id: int, shot_id: int) -> dict[str, np.ndarray]:
    """Computed waveforms of one shot in one run, as {name: array} (read-only views of the stored bytes)."""
    rows = con.execute("SELECT name, data FROM waveform WHERE run_id = ? AND shot_id = ?", (run_id, shot_id))
    return {name: np.frombuffer(data, dtype=BLOB_DTYPE) for name, data in rows}


def load_withstand(con: sqlite3.Connection, run_id: int) -> Withstand:
    """Read back the withstand result of one run, as `process_withstand` returned it. ValueError if the run is missing."""
    row = con.execute(
        "SELECT s.id, s.level, s.number, r.up_kv, r.t1_us, r.tau_us, r.cg_c4_nf"
        " FROM run JOIN shot_result r ON r.run_id = run.id AND r.shot_id = run.withstand_shot_id"
        " JOIN shot s ON s.id = r.shot_id WHERE run.id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Run {run_id} is not in the database")
    shot_id, level, number, up, t1, tau, cg_c4 = row
    wf = _load_waveforms(con, run_id, shot_id)
    return Withstand(level, number, wf["t_sync"], wf["u"], up, cg_c4, tau, t1)


def load_breakdowns(con: sqlite3.Connection, run_id: int) -> list[Breakdown]:
    """Read back the breakdown results of one run, as `process_breakdown` returned them (ordered by level, shot)."""
    rows = con.execute(
        "SELECT s.id, s.level, s.number, r.up_kv, r.amplitude_kv, r.tau_us, r.t1_us"
        " FROM run JOIN shot_result r ON r.run_id = run.id AND r.shot_id != run.withstand_shot_id"
        " JOIN shot s ON s.id = r.shot_id WHERE run.id = ? ORDER BY CAST(s.level AS REAL), s.number",
        (run_id,),
    ).fetchall()
    breakdowns = []
    for shot_id, level, number, up, amplitude, tau, t1 in rows:
        events = {
            name: {"t": t, "u": u, "i_exc": i, "q_exc": q, "e_exc": e}
            for name, t, u, i, q, e in con.execute(
                "SELECT name, t_us, u_kv, i_exc_a, q_exc_uc, e_exc_mj FROM event"
                " WHERE run_id = ? AND shot_id = ? ORDER BY t_us",
                (run_id, shot_id),
            )
        }
        breakdowns.append(Breakdown(level, number, up, amplitude, tau, t1, events, _load_waveforms(con, run_id, shot_id)))
    return breakdowns


def campaign_polarity(con: sqlite3.Connection, campaign: str) -> int:
    """Polarity (+1/-1) stored for a campaign at import. ValueError if the campaign is not in the database."""
    row = con.execute("SELECT polarity FROM campaign WHERE name = ?", (campaign,)).fetchone()
    if row is None:
        raise ValueError(f"Campaign {campaign} is not in the database (run `arccharcalc import` first)")
    return row[0]


def load_shot(con: sqlite3.Connection, campaign: str, level: str, number: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read one shot from the database, in the same form as `read_shot`.

    Returns (t, channels): the common time axis in microseconds and a dict
    {"C2": ..., "C3": ..., "C4": ...} of amplitudes in volts. The arrays are
    read-only views of the stored bytes (numpy raises an error if code writes into them).
    """
    row = con.execute(
        "SELECT s.t, s.c2, s.c3, s.c4 FROM shot s JOIN campaign c ON c.id = s.campaign_id"
        " WHERE c.name = ? AND s.level = ? AND s.number = ?",
        (campaign, level, number),
    ).fetchone()
    if row is None:
        raise ValueError(f"Shot {level}/{number:05d} of campaign {campaign} is not in the database")
    t, *channels = (np.frombuffer(blob, dtype=BLOB_DTYPE) for blob in row)
    return t, dict(zip(CHANNELS, channels))


def import_campaign(con: sqlite3.Connection, campaign_dir: Path, polarity: int) -> tuple[int, int]:
    """Copy all shots of one campaign folder into the database. The raw files are only read.

    Every subfolder (one per charging level) is searched for `C3_<level>_<no>.csv`;
    C2 and C4 of the same shot are read with `read_shot`. A subfolder whose name
    contains "Withstand" holds withstand shots.

    A shot already in the database with the same SHA-256 is skipped. If its files
    changed, a ValueError is raised: raw data should never change.

    Returns (new, skipped): the number of shots added and skipped.
    """
    with con:  # one transaction: all shots or none
        row = con.execute("SELECT id, polarity FROM campaign WHERE name = ?", (campaign_dir.name,)).fetchone()
        if row is None:
            campaign_id = con.execute(
                "INSERT INTO campaign (name, polarity, source_dir) VALUES (?, ?, ?)",
                (campaign_dir.name, polarity, str(campaign_dir.resolve())),
            ).lastrowid
        elif row[1] != polarity:
            raise ValueError(f"Campaign {campaign_dir.name} is already stored with polarity {row[1]:+d}")
        else:
            campaign_id = row[0]

        new = skipped = 0
        for level_dir in sorted(p for p in campaign_dir.iterdir() if p.is_dir()):
            for c3_file in sorted(level_dir.glob("C3_*.csv")):
                m = re.fullmatch(r"C3_(.+)_(\d{5})\.csv", c3_file.name)
                if m is None:
                    raise ValueError(f"Unexpected file name: {c3_file}")
                level, number = m.group(1), int(m.group(2))
                files = [level_dir / f"{name}_{level}_{number:05d}.csv" for name in CHANNELS]
                sha = _sha256(files)

                old = con.execute(
                    "SELECT sha256 FROM shot WHERE campaign_id = ? AND level = ? AND number = ?",
                    (campaign_id, level, number),
                ).fetchone()
                if old is not None:
                    if old[0] != sha:
                        raise ValueError(f"Raw files of {level}/{number:05d} changed since the import")
                    skipped += 1
                    continue

                t, channels = read_shot(level_dir, level, number)
                n = [read_lecroy_header(f)[0] for f in files]
                trig_time = read_lecroy_header(files[1])[1]  # C3
                con.execute(
                    "INSERT INTO shot (campaign_id, level, number, is_withstand, trig_time, n_c2, n_c3, n_c4,"
                    " sha256, t, c2, c3, c4) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (campaign_id, level, number, int("Withstand" in level_dir.name), trig_time, *n,
                     sha, _blob(t), *(_blob(channels[name]) for name in CHANNELS)),
                )
                new += 1
    return new, skipped
