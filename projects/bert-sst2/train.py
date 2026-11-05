"""Fine-tune BERT for sentiment classification on GLUE SST-2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

DEFAULT_MODEL = "google-bert/bert-base-uncased"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/bert-sst2"))
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 uses the full train split")
    parser.add_argument("--max-eval-samples", type=int, default=0, help="0 uses the full validation split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for name in ("max_length", "train_batch_size", "eval_batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.epochs <= 0 or args.learning_rate <= 0:
        parser.error("--epochs and --learning-rate must be greater than 0")
    if args.max_train_samples < 0 or args.max_eval_samples < 0:
        parser.error("sample limits must be 0 or greater")
    return args


def compute_metrics(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted")),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset = load_dataset("glue", "sst2")
    train_data = dataset["train"]
    eval_data = dataset["validation"]
    if args.max_train_samples:
        train_data = train_data.shuffle(seed=args.seed).select(
            range(min(args.max_train_samples, len(train_data)))
        )
    if args.max_eval_samples:
        eval_data = eval_data.shuffle(seed=args.seed).select(
            range(min(args.max_eval_samples, len(eval_data)))
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch: dict[str, list[str]]) -> dict:
        return tokenizer(batch["sentence"], truncation=True, max_length=args.max_length)

    train_data = train_data.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    eval_data = eval_data.map(tokenize, batched=True, remove_columns=["sentence", "idx"])
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    result = {
        "base_model": args.model_name,
        "dataset": "glue/sst2",
        "train_samples": len(train_data),
        "validation_samples": len(eval_data),
        "seed": args.seed,
        "max_length": args.max_length,
        "metrics": metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
