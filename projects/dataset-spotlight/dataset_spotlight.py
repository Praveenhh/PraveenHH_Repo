"""Create a quick, local overview of a Hugging Face dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def format_report(dataset_id: str, split_name: str, split: Any, examples: int) -> str:
    """Format dataset metadata and a small sample as Markdown."""
    row_count = split.num_rows if hasattr(split, "num_rows") else "unknown (streaming)"
    lines = [
        f"# Dataset Spotlight: `{dataset_id}`",
        "",
        f"- **Split:** `{split_name}`",
        f"- **Rows:** {row_count}",
        f"- **Columns:** {', '.join(f'`{name}`' for name in split.column_names)}",
        "",
        "## Features",
        "",
    ]

    for name, feature in split.features.items():
        lines.append(f"- `{name}`: `{feature}`")

    lines.extend(["", f"## Sample ({examples} row(s))", "", "```json"])
    preview = [split[index] for index in range(min(examples, len(split)))]
    lines.append(json.dumps(preview, ensure_ascii=False, indent=2, default=str))
    lines.extend(["```", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a Hugging Face dataset and optionally save a Markdown report."
    )
    parser.add_argument("dataset", help="Dataset ID on the Hugging Face Hub")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument(
        "--examples", type=int, default=3, help="Number of rows to preview (default: 3)"
    )
    parser.add_argument("--output", type=Path, help="Optional Markdown report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.examples < 0:
        raise SystemExit("--examples must be zero or greater")

    try:
        dataset = load_dataset(args.dataset, split=args.split)
    except Exception as error:
        raise SystemExit(
            f"Could not load {args.dataset!r} split {args.split!r}: {error}"
        ) from error

    report = format_report(args.dataset, args.split, dataset, args.examples)
    print(report)
    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"\nSaved report to {args.output}")


if __name__ == "__main__":
    main()
