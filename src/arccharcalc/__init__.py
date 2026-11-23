import sqlite3
import sys
from pathlib import Path

from arccharcalc.db import (
    campaign_polarity,
    connect,
    import_campaign,
    load_breakdowns,
    load_shot,
    load_withstand,
    save_breakdown,
    save_withstand,
    start_run,
)
from arccharcalc.pipeline import process_breakdown, process_withstand
from arccharcalc.report import export_csv, plot_run, print_run

USAGE = ("usage: arccharcalc <database file> <campaign> <level> <shot> <withstand level> <withstand shot>\n"
         "   or: arccharcalc import <database file> <campaign folder> <polarity>\n"
         "   or: arccharcalc show <database file> <run id>\n"
         "   or: arccharcalc export <database file> <run id> <output folder>")


def _open_existing(db_file: str) -> sqlite3.Connection:
    """Connect to a database file that must already exist (connect() would silently create an empty one)."""
    if not Path(db_file).is_file():
        raise ValueError(f"Database file {db_file} not found (run `arccharcalc import` first)")
    return connect(Path(db_file))


def _show_run(con: sqlite3.Connection, run_id: int) -> None:
    """Print and plot one run, read back from the database."""
    w = load_withstand(con, run_id)
    breakdowns = load_breakdowns(con, run_id)
    print_run(w, breakdowns)
    plot_run(w, breakdowns)


def main() -> None:
    # Copy a campaign into the database (the raw CSVs are only read here):
    #   arccharcalc import ../results/arccharcalc.sqlite ../data/raw/2024-04-15_56-1-2sheds_LI-1.2-50_pos +1
    # Process one breakdown shot against the withstand shot of the same campaign (all 6 arguments required):
    #   arccharcalc ../results/arccharcalc.sqlite 2024-04-15_56-1-2sheds_LI-1.2-50_pos 15.5 3 11 0
    # Print and plot a stored run again:
    #   arccharcalc show ../results/arccharcalc.sqlite 1
    # Write the computed waveforms of a stored run to CSV, one file per shot (for checks in Excel):
    #   arccharcalc export ../results/arccharcalc.sqlite 1 ../results/csv

    if len(sys.argv) > 1 and sys.argv[1] == "import":
        if len(sys.argv) < 5:
            raise ValueError(USAGE)
        con = connect(Path(sys.argv[2]))
        new, skipped = import_campaign(con, Path(sys.argv[3]), int(sys.argv[4]))
        con.close()
        print(f"Imported {new} new shots, skipped {skipped} already in the database")
        return

    if len(sys.argv) > 1 and sys.argv[1] == "show":
        if len(sys.argv) < 4:
            raise ValueError(USAGE)
        con = _open_existing(sys.argv[2])
        _show_run(con, int(sys.argv[3]))
        con.close()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "export":
        if len(sys.argv) < 5:
            raise ValueError(USAGE)
        con = _open_existing(sys.argv[2])
        run_id = int(sys.argv[3])
        paths = export_csv(load_withstand(con, run_id), load_breakdowns(con, run_id), Path(sys.argv[4]), run_id)
        con.close()
        for p in paths:
            print(f"Wrote {p}")
        return

    if len(sys.argv) < 7:
        raise ValueError(USAGE)
    db_file, campaign, level, shot, w_level, w_shot = sys.argv[1:7]
    con = _open_existing(db_file)
    polarity = campaign_polarity(con, campaign)
    with con:  # one transaction: the whole run is stored, or nothing if anything fails
        run_id = start_run(con, campaign, w_level, int(w_shot))
        withstand = process_withstand(*load_shot(con, campaign, w_level, int(w_shot)), w_level, int(w_shot), polarity)
        save_withstand(con, run_id, campaign, withstand)
        breakdown = process_breakdown(*load_shot(con, campaign, level, int(shot)), level, int(shot), polarity, withstand)
        save_breakdown(con, run_id, campaign, breakdown)
    print(f"\nStored as run {run_id} in {db_file}")
    _show_run(con, run_id)  # what is printed and plotted is what the database holds
    con.close()
