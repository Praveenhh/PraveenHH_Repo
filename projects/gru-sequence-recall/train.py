"""Train a GRU for variable-delay binary sequence recall."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class RecallGRU(nn.Module):
    """Classify the marked bit from the final valid GRU state."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 2))

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            inputs, lengths.detach().to("cpu"), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return self.classifier(hidden[-1])


def make_batch(
    batch_size: int,
    max_delay: int,
    device: torch.device,
    generator: torch.Generator | None = None,
    *,
    balanced_delays: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate marked-bit examples and return inputs, labels, lengths, and delays."""
    if batch_size < 1 or max_delay < 1:
        raise ValueError("batch_size and max_delay must be at least 1")
    if balanced_delays:
        delays = (torch.arange(batch_size) % max_delay) + 1
        delays = delays[torch.randperm(batch_size, generator=generator)]
    else:
        delays = torch.randint(1, max_delay + 1, (batch_size,), generator=generator)
    lengths = delays + 2
    labels = torch.randint(0, 2, (batch_size,), generator=generator)
    inputs = torch.zeros(batch_size, int(lengths.max()), 3, dtype=torch.float32)
    inputs[:, 0, 0] = labels.float()
    inputs[:, 0, 1] = 1.0
    for row, delay in enumerate(delays.tolist()):
        inputs[row, 1 : delay + 1, 0] = torch.randint(
            0, 2, (delay,), generator=generator
        ).float()
        inputs[row, delay + 1, 2] = 1.0
    return inputs.to(device), labels.to(device), lengths.to(device), delays.to(device)


def evaluate(model: RecallGRU, data, criterion, device: torch.device) -> tuple[float, float]:
    inputs, labels, lengths, _ = data
    model.eval()
    total_loss = 0.0
    predictions: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(labels), 512):
            end = start + 512
            logits = model(inputs[start:end].to(device), lengths[start:end])
            target = labels[start:end].to(device)
            total_loss += float(criterion(logits, target).item()) * len(target)
            predictions.append(logits.argmax(dim=1).cpu())
    predicted = torch.cat(predictions).numpy()
    target_np = labels.numpy()
    return total_loss / len(target_np), float(accuracy_score(target_np, predicted))


def collect_metrics(model: RecallGRU, data, device: torch.device) -> dict:
    inputs, labels, lengths, delays = data
    model.eval()
    all_probabilities: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(labels), 512):
            end = start + 512
            logits = model(inputs[start:end].to(device), lengths[start:end])
            all_probabilities.append(torch.softmax(logits, dim=1).cpu())
    probabilities = torch.cat(all_probabilities).numpy()
    targets = labels.numpy()
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    per_delay: dict[str, dict[str, int | float]] = {}
    for delay in torch.unique(delays, sorted=True).tolist():
        mask = delays.numpy() == delay
        count = int(mask.sum())
        correct = int((predictions[mask] == targets[mask]).sum())
        per_delay[str(delay)] = {
            "accuracy": correct / count,
            "correct": correct,
            "count": count,
        }
    try:
        roc_auc = float(roc_auc_score(targets, probabilities[:, 1]))
    except ValueError:
        roc_auc = None
    try:
        average_precision = float(average_precision_score(targets, probabilities[:, 1]))
    except ValueError:
        average_precision = None
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_precision": float(precision_score(targets, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(targets, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(targets, predictions, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(targets, predictions, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "per_class": classification_report(
            targets, predictions, labels=[0, 1], target_names=["remembered_0", "remembered_1"],
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": matrix.tolist(),
        "accuracy_by_delay": per_delay,
        "test_sample_count": int(len(targets)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/gru-recall"))
    parser.add_argument("--max-delay", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--validation-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    for name in (
        "max_delay", "epochs", "train_size", "validation_size", "test_size",
        "batch_size", "hidden_size", "num_layers", "patience",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than zero")
    return args


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; install a compatible PyTorch build")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    val_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    test_generator = torch.Generator(device="cpu").manual_seed(args.seed + 2)
    validation_data = make_batch(
        args.validation_size, args.max_delay, torch.device("cpu"), val_generator,
        balanced_delays=True,
    )
    test_data = make_batch(
        args.test_size, args.max_delay, torch.device("cpu"), test_generator,
        balanced_delays=True,
    )

    model = RecallGRU(args.hidden_size, args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    training_generator = torch.Generator(device="cpu").manual_seed(args.seed + 3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pt"
    best_validation_accuracy = -1.0
    stale_epochs = 0
    history: list[dict[str, int | float]] = []

    print(f"Device: {device}; GRU layers: {args.num_layers}; max delay: {args.max_delay}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        remaining = args.train_size
        while remaining:
            count = min(args.batch_size, remaining)
            batch = make_batch(
                count, args.max_delay, device, training_generator
            )
            inputs, labels, lengths, _ = batch
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs, lengths), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * count
            seen += count
            remaining -= count

        validation_loss, validation_accuracy = evaluate(model, validation_data, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} train_loss={row['train_loss']:.4f} "
            f"val_loss={validation_loss:.4f} val_accuracy={validation_accuracy:.4f}"
        )
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "max_delay": args.max_delay,
                    "seed": args.seed,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs without validation improvement.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    metrics = collect_metrics(model, test_data, device)
    metrics.update(
        {
            "architecture": "RecallGRU",
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "max_delay": args.max_delay,
            "seed": args.seed,
            "selected_checkpoint_validation_accuracy": best_validation_accuracy,
        }
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    print("\nHeld-out test metrics:")
    for metric in (
        "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1", "roc_auc", "average_precision",
    ):
        print(f"{metric}: {metrics[metric]}")
    print(f"Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
