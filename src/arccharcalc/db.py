"""SQLite database with a copy of the oscilloscope shots (and later the results), see plan/ADR-001_database.md."""

import hashlib
import re
import sqlite3
from pathlib import Path

import numpy as np

from arccharcalc.io_lecroy import read_lecroy_header, read_shot

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
