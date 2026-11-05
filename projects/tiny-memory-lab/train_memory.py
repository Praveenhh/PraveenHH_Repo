"""Train an LSTM to recall a marked bit after random distractors."""

from __future__ import annotations

import argparse
import random

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class RecallLSTM(nn.Module):
    """Encode a variable-length sequence and classify the remembered bit."""

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            inputs, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(hidden[-1])


def make_batch(
    batch_size: int,
    max_delay: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate marked-bit sequences with variable distractor delays."""
    delays = torch.randint(
        1, max_delay + 1, (batch_size,), generator=generator, device="cpu"
    )
    lengths = delays + 2  # remembered value, distractors, query token
    max_length = int(lengths.max().item())
    inputs = torch.zeros(batch_size, max_length, 3, dtype=torch.float32)
    labels = torch.randint(0, 2, (batch_size,), generator=generator, device="cpu")

    inputs[:, 0, 0] = labels.float()
    inputs[:, 0, 1] = 1.0  # mark the value to remember
    for row, delay in enumerate(delays.tolist()):
        if delay:
            inputs[row, 1 : delay + 1, 0] = torch.randint(
                0, 2, (delay,), generator=generator, device="cpu"
            ).float()
        inputs[row, delay + 1, 2] = 1.0  # query arrives after distractors

    return inputs.to(device), labels.to(device), lengths.to(device)


def evaluate(
    model: RecallLSTM,
    size: int,
    batch_size: int,
    max_delay: int,
    device: torch.device,
    seed: int,
) -> tuple[float, dict[int, tuple[int, int]]]:
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    correct = 0
    total = 0
    by_delay: dict[int, list[int]] = {}
    with torch.inference_mode():
        remaining = size
        while remaining:
            count = min(batch_size, remaining)
            inputs, labels, lengths = make_batch(count, max_delay, device, generator)
            predictions = model(inputs, lengths).argmax(dim=1)
            matches = predictions.eq(labels).cpu()
            delays = (lengths - 2).cpu()
            correct += int(matches.sum().item())
            total += count
            for delay, match in zip(delays.tolist(), matches.tolist()):
                values = by_delay.setdefault(delay, [0, 0])
                values[0] += int(match)
                values[1] += 1
            remaining -= count
    return correct / max(total, 1), {key: (value[0], value[1]) for key, value in sorted(by_delay.items())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-delay", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    for name in ("max_delay", "epochs", "train_size", "test_size", "batch_size", "hidden_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available in this PyTorch installation")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = RecallLSTM(args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    train_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)

    print(f"Device: {device}; maximum distractor delay: {args.max_delay}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        remaining = args.train_size
        while remaining:
            count = min(args.batch_size, remaining)
            inputs, labels, lengths = make_batch(
                count, args.max_delay, device, train_generator
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, lengths)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * count
            seen += count
            remaining -= count

        accuracy, _ = evaluate(
            model,
            args.test_size,
            args.batch_size,
            args.max_delay,
            device,
            args.seed + 10_000,
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={total_loss / seen:.4f} test_accuracy={accuracy:.3f}"
        )

    _, by_delay = evaluate(
        model,
        args.test_size,
        args.batch_size,
        args.max_delay,
        device,
        args.seed + 20_000,
    )
    print("\nHeld-out accuracy by distractor delay:")
    for delay, (correct, count) in by_delay.items():
        print(f"  delay={delay:2d}: {correct / count:.3f} ({correct}/{count})")


if __name__ == "__main__":
    main()
