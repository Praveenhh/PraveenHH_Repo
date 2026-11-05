"""Command-line interface for the constraint-based Sudoku solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from solver import format_grid, parse_puzzle, solve, stats_dict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--puzzle", help="81 row-major cells; use digits 1-9, 0, or .")
    source.add_argument("--file", type=Path, help="Text file containing one 81-cell puzzle")
    parser.add_argument("--max-solutions", type=int, default=2, help="Stop after this many solutions (default: 2)")
    parser.add_argument("--output", type=Path, help="Optional path to save the JSON result")
    args = parser.parse_args()
    if args.max_solutions < 1:
        parser.error("--max-solutions must be at least 1")

    try:
        puzzle_text = args.puzzle if args.puzzle is not None else args.file.read_text(encoding="utf-8")
        result = solve(parse_puzzle(puzzle_text), max_solutions=args.max_solutions)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    payload = {
        "status": result.status,
        "solutions_found": len(result.solutions),
        "solutions": [format_grid(grid) for grid in result.solutions],
        "stats": stats_dict(result.stats),
        "elapsed_seconds": result.elapsed_seconds,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result.solutions else 1


if __name__ == "__main__":
    raise SystemExit(main())
