"""Run sentiment prediction with a saved transformer classifier."""

from __future__ import annotations

import argparse

from transformers import pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="artifacts/distilbert-imdb")
    parser.add_argument("text", nargs="+", help="One or more review texts")
    args = parser.parse_args()

    classifier = pipeline(
        "text-classification",
        model=args.model_dir,
        tokenizer=args.model_dir,
        truncation=True,
    )
    for text in args.text:
        result = classifier(text)[0]
        print(f"{result['label']} (score={result['score']:.4f}): {text}")


if __name__ == "__main__":
    main()
