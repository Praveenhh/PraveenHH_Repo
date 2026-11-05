"""Classify one or more sentences using a fine-tuned BERT checkpoint."""

from __future__ import annotations

import argparse

from transformers import pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="artifacts/bert-sst2")
    parser.add_argument("sentences", nargs="+", help="Sentences to classify")
    args = parser.parse_args()

    classifier = pipeline(
        "text-classification",
        model=args.model_dir,
        tokenizer=args.model_dir,
        truncation=True,
    )
    for sentence in args.sentences:
        result = classifier(sentence)[0]
        print(f"{result['label']} (score={result['score']:.4f}): {sentence}")


if __name__ == "__main__":
    main()
