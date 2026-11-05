from __future__ import annotations

import torch

from train_memory import RecallLSTM, make_batch


def test_generated_batch_has_marked_value_and_query() -> None:
    inputs, labels, lengths = make_batch(
        batch_size=16,
        max_delay=5,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(1),
    )

    assert inputs.shape[0] == 16
    assert inputs.shape[2] == 3
    assert labels.shape == (16,)
    assert torch.all(inputs[:, 0, 1] == 1)
    for row, length in enumerate(lengths.tolist()):
        assert inputs[row, length - 1, 2] == 1
        assert inputs[row, : length - 1, 2].sum() == 0


def test_model_outputs_two_class_logits() -> None:
    inputs, _, lengths = make_batch(8, 4, torch.device("cpu"))
    logits = RecallLSTM(hidden_size=16)(inputs, lengths)

    assert logits.shape == (8, 2)
