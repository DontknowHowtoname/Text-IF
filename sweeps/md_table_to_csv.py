"""Convert a GitHub-flavored Markdown table to CSV.

Defaults to converting ``sweeps/out/comparison_table.md`` ->
``sweeps/out/comparison_table.csv`` (overwrites). Strips ``**`` bold and
``<u>...</u>`` underline markup so cells are pure values.

Usage:
    python sweeps/md_table_to_csv.py
    python sweeps/md_table_to_csv.py --input path/to/table.md --output path/to/out.csv
"""
import argparse
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_IN = REPO / "sweeps" / "out" / "comparison_table.md"
DEFAULT_OUT = REPO / "sweeps" / "out" / "comparison_table.csv"

_SEP_CELL = re.compile(r":?-+:?")  # |---|:---:|---:| etc.


def _clean(cell: str) -> str:
    """Strip **bold** and <u>underline</u> markup, trim whitespace."""
    cell = cell.strip()
    cell = cell.replace("**", "")
    cell = re.sub(r"</?u>", "", cell)
    return cell


def convert(in_path: Path, out_path: Path) -> int:
    """Parse in_path markdown table -> write out_path CSV (utf-8-sig).

    Returns the number of data rows written (excluding header).
    """
    rows = []
    for line in in_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        parts = [c.strip() for c in line.strip("|").split("|")]
        # skip separator row
        if all(_SEP_CELL.fullmatch(c) for c in parts):
            continue
        rows.append([_clean(c) for c in parts])

    if not rows:
        print(f"ERROR: no table rows found in {in_path}", file=sys.stderr)
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig (BOM) so Excel/WPS open M³FD etc. without garbling.
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)

    print(f"wrote {out_path} ({len(rows)} rows incl. header, {len(rows[0])} cols)")
    return len(rows) - 1  # data rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input",  type=str, default=str(DEFAULT_IN),
                    help=f"Input markdown file (default: {DEFAULT_IN})")
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUT),
                    help=f"Output CSV file, overwritten (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 1

    convert(in_path, Path(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
