from __future__ import annotations

import numpy as np
import torch

from train import CLASS_NAMES, SmallCifarCNN, calculate_metrics


def test_cnn_outputs_ten_class_logits() -> None:
    model = SmallCifarCNN()
    output = model(torch.zeros(2, 3, 32, 32))
    assert output.shape == (2, 10)


def test_metrics_include_aggregate_and_per_class_results() -> None:
    labels = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.eye(3)[labels]
    expanded = np.zeros((len(labels), len(CLASS_NAMES)))
    expanded[:, :3] = probabilities

    metrics = calculate_metrics(labels, expanded)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert len(metrics["confusion_matrix"]) == 10
    assert "airplane" in metrics["per_class"]
    assert metrics["macro_ovr_roc_auc"] is not None
