import sys
from pathlib import Path

from arccharcalc.db import campaign_polarity, connect, import_campaign, load_shot
from arccharcalc.pipeline import process_breakdown, process_withstand

USAGE = ("usage: arccharcalc <database file> <campaign> <level> <shot> <withstand level> <withstand shot>\n"
         "   or: arccharcalc import <database file> <campaign folder> <polarity>")


def main() -> None:
    # Copy a campaign into the database (the raw CSVs are only read here):
    #   arccharcalc import ../results/arccharcalc.sqlite ../data/raw/2024-04-15_56-1-2sheds_LI-1.2-50_pos +1
    # Process one breakdown shot against the withstand shot of the same campaign (all 6 arguments required):
    #   arccharcalc ../results/arccharcalc.sqlite 2024-04-15_56-1-2sheds_LI-1.2-50_pos 15.5 3 11 0

    if len(sys.argv) > 1 and sys.argv[1] == "import":
        if len(sys.argv) < 5:
            raise ValueError(USAGE)
        con = connect(Path(sys.argv[2]))
        new, skipped = import_campaign(con, Path(sys.argv[3]), int(sys.argv[4]))
        con.close()
        print(f"Imported {new} new shots, skipped {skipped} already in the database")
        return

    if len(sys.argv) < 7:
        raise ValueError(USAGE)
    db_file, campaign, level, shot, w_level, w_shot = sys.argv[1:7]
    if not Path(db_file).is_file():  # connect() would silently create an empty database
        raise ValueError(f"Database file {db_file} not found (run `arccharcalc import` first)")

    con = connect(Path(db_file))
    polarity = campaign_polarity(con, campaign)
    withstand = process_withstand(*load_shot(con, campaign, w_level, int(w_shot)), w_level, int(w_shot), polarity)
    process_breakdown(*load_shot(con, campaign, level, int(shot)), level, int(shot), polarity, withstand)
    con.close()
