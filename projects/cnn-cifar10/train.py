"""Train a compact CNN on CIFAR-10 and report held-out classification metrics."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class SmallCifarCNN(nn.Module):
    """Compact convolutional baseline for 32×32 RGB images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cifar10-cnn"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--limit-train-samples", type=int, default=0, help="0 uses all available training examples")
    parser.add_argument("--limit-test-samples", type=int, default=0, help="0 uses the full test split")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs and batch-size must be positive; num-workers cannot be negative")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be greater than zero")
    if not 0 < args.validation_fraction < 1:
        parser.error("validation-fraction must be between 0 and 1")
    if args.limit_train_samples < 0 or args.limit_test_samples < 0:
        parser.error("sample limits cannot be negative")
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; install a compatible PyTorch build")
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def make_loaders(args: argparse.Namespace, seed: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )
    augmented = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_transform)
    clean_train = datasets.CIFAR10(args.data_dir, train=True, download=False, transform=eval_transform)
    test_set = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=eval_transform)

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(augmented), generator=generator).tolist()
    if args.limit_train_samples:
        order = order[: min(args.limit_train_samples, len(order))]
    validation_count = max(1, int(len(order) * args.validation_fraction))
    if len(order) - validation_count < 1:
        raise ValueError("Training sample limit must leave at least one train and validation sample")
    validation_indices = order[:validation_count]
    train_indices = order[validation_count:]
    if args.limit_test_samples:
        test_set = Subset(test_set, range(min(args.limit_test_samples, len(test_set))))

    train_loader = DataLoader(
        Subset(augmented, train_indices), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        Subset(clean_train, validation_indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, validation_loader, test_loader, len(train_indices), validation_count


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def collect_predictions(model, loader, device):
    model.eval()
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            probabilities_all.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels_all.append(labels.numpy())
    return np.concatenate(labels_all), np.concatenate(probabilities_all)


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = probabilities.argmax(axis=1)
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.int32)[labels]
    per_class_report = classification_report(
        labels, predictions, labels=list(range(len(CLASS_NAMES))),
        target_names=list(CLASS_NAMES), output_dict=True, zero_division=0,
    )
    try:
        macro_roc_auc = float(roc_auc_score(labels, probabilities, multi_class="ovr", average="macro"))
    except ValueError:
        macro_roc_auc = None
    per_class_ap = average_precision_score(one_hot, probabilities, average=None)
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(CLASS_NAMES))))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "top_3_accuracy": float(np.mean(np.argsort(probabilities, axis=1)[:, -3:] == labels[:, None]).__float__()),
        "macro_precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(labels, predictions, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(labels, predictions, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "macro_ovr_roc_auc": macro_roc_auc,
        "macro_ovr_average_precision": float(np.mean(per_class_ap)),
        "per_class_average_precision": {
            name: float(value) for name, value in zip(CLASS_NAMES, per_class_ap)
        },
        "per_class": per_class_report,
        "confusion_matrix": matrix.tolist(),
        "class_names": list(CLASS_NAMES),
    }


def save_plots(history: list[dict], matrix: list[list[int]], output_dir: Path) -> None:
    epochs = [item["epoch"] for item in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="Train")
    axes[0].plot(epochs, [item["validation_loss"] for item in history], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot(epochs, [item["train_accuracy"] for item in history], label="Train")
    axes[1].plot(epochs, [item["validation_accuracy"] for item in history], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(np.asarray(matrix), cmap="Blues")
    axis.set(
        xticks=range(len(CLASS_NAMES)), yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        xlabel="Predicted label", ylabel="True label", title="CIFAR-10 test confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, validation_loader, test_loader, train_count, validation_count = make_loaders(args, args.seed)

    model = SmallCifarCNN(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history: list[dict] = []
    best_validation_accuracy = -1.0
    checkpoint_path = args.output_dir / "best_model.pt"

    print(f"Device: {device}; train={train_count}; validation={validation_count}; test={len(test_loader.dataset)}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss, validation_accuracy = run_epoch(model, validation_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={validation_loss:.4f} val_acc={validation_accuracy:.4f}"
        )
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": list(CLASS_NAMES),
                    "seed": args.seed,
                    "architecture": "SmallCifarCNN",
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_labels, test_probabilities = collect_predictions(model, test_loader, device)
    metrics = calculate_metrics(test_labels, test_probabilities)
    metrics["selected_checkpoint_validation_accuracy"] = best_validation_accuracy
    metrics["test_sample_count"] = int(len(test_labels))
    metrics["seed"] = args.seed
    metrics["architecture"] = "SmallCifarCNN"

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    save_plots(history, metrics["confusion_matrix"], args.output_dir)
    print("\nHeld-out test metrics:")
    for key in (
        "accuracy", "balanced_accuracy", "top_3_accuracy", "macro_precision",
        "macro_recall", "macro_f1", "weighted_precision", "weighted_recall",
        "weighted_f1", "macro_ovr_roc_auc", "macro_ovr_average_precision",
    ):
        print(f"{key}: {metrics[key]}")
    print(f"Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
