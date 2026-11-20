import sys
from pathlib import Path

from arccharcalc.pipeline import process_breakdown, process_withstand


def main() -> None:
    # usage: arccharcalc <level folder> <level> <shot> <polarity> <withstand folder> <withstand level> <withstand shot>
    # e.g. ".../15.5kV" 15.5 3 +1 ".../11kV (Withstand Voltage)" 11 0
    # all 7 arguments are required

    if len(sys.argv) < 4:
        raise ValueError("No breakdown shot given: usage: arccharcalc <level folder> <level> <shot> <polarity> <withstand folder> <withstand level> <withstand shot> (all required)")
    if len(sys.argv) < 8:  # argv[4..7]: polarity, withstand folder, level, shot
        raise ValueError("Polarity or withstand shot missing: usage: arccharcalc <level folder> <level> <shot> <polarity> <withstand folder> <withstand level> <withstand shot> (all required)")

    polarity = int(sys.argv[4])
    withstand = process_withstand(Path(sys.argv[5]), sys.argv[6], int(sys.argv[7]), polarity)
    process_breakdown(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), polarity, withstand)
