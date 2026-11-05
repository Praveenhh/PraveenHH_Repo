from __future__ import annotations

import unittest

import torch

from train import RecallGRU, collect_metrics, make_batch


class GruSequenceRecallTests(unittest.TestCase):
    def test_batch_marks_target_and_places_query_after_delay(self) -> None:
        inputs, labels, lengths, delays = make_batch(
            16, 5, torch.device("cpu"), torch.Generator().manual_seed(4), balanced_delays=True
        )
        self.assertEqual(inputs.shape[0], 16)
        self.assertEqual(inputs.shape[2], 3)
        self.assertTrue(torch.equal(inputs[:, 0, 0].long(), labels))
        self.assertTrue(torch.all(inputs[:, 0, 1] == 1))
        for row, length in enumerate(lengths.tolist()):
            self.assertEqual(length, int(delays[row]) + 2)
            self.assertEqual(float(inputs[row, length - 1, 2]), 1.0)
            self.assertEqual(float(inputs[row, : length - 1, 2].sum()), 0.0)

    def test_gru_returns_two_class_logits_for_variable_lengths(self) -> None:
        inputs, _, lengths, _ = make_batch(8, 4, torch.device("cpu"))
        logits = RecallGRU(hidden_size=16, num_layers=2)(inputs, lengths)
        self.assertEqual(tuple(logits.shape), (8, 2))

    def test_metrics_include_per_delay_and_class_results(self) -> None:
        inputs, labels, lengths, delays = make_batch(
            20, 4, torch.device("cpu"), torch.Generator().manual_seed(2), balanced_delays=True
        )
        model = RecallGRU(hidden_size=8, num_layers=1)
        metrics = collect_metrics(model, (inputs, labels, lengths, delays), torch.device("cpu"))
        self.assertEqual(metrics["test_sample_count"], 20)
        self.assertEqual(len(metrics["confusion_matrix"]), 2)
        self.assertEqual(set(metrics["accuracy_by_delay"]), {"1", "2", "3", "4"})
        self.assertIn("remembered_0", metrics["per_class"])
        self.assertIn("remembered_1", metrics["per_class"])

    def test_invalid_batch_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_batch(0, 3, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
