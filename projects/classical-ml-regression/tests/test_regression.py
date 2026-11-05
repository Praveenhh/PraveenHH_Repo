from __future__ import annotations

import unittest

import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, r2_score

from train import build_candidates, regression_metrics


class ClassicalRegressionTests(unittest.TestCase):
    def test_metrics_match_known_values(self) -> None:
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.0, 3.0, 2.0, 4.0])
        metrics = regression_metrics(actual, predicted)
        self.assertAlmostEqual(metrics["mae"], mean_absolute_error(actual, predicted))
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(0.5))
        self.assertAlmostEqual(metrics["r2"], r2_score(actual, predicted))
        self.assertAlmostEqual(metrics["median_absolute_error"], 0.5)
        self.assertIn("explained_variance", metrics)

    def test_candidate_pipelines_fit_small_regression_data(self) -> None:
        x, y = make_regression(n_samples=40, n_features=4, random_state=3)
        for model in build_candidates(seed=5).values():
            model.fit(x, y)
            self.assertEqual(model.predict(x[:3]).shape, (3,))


if __name__ == "__main__":
    unittest.main()
