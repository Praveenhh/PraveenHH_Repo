from __future__ import annotations

import unittest

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from train import evaluate_clusters


class UnsupervisedDigitsTests(unittest.TestCase):
    def test_metrics_and_contingency_cover_clusters_and_digits(self) -> None:
        labels = np.tile(np.arange(10), 4)
        clusters = labels.copy()
        features = np.eye(10)[labels]
        metrics = evaluate_clusters(features, clusters, labels)
        self.assertAlmostEqual(metrics["adjusted_rand_index"], 1.0)
        self.assertAlmostEqual(metrics["normalized_mutual_information"], 1.0)
        self.assertEqual(len(metrics["cluster_sizes"]), 10)
        self.assertEqual(sum(sum(row.values()) for row in metrics["cluster_digit_contingency"].values()), 40)
        self.assertIn("silhouette_score", metrics)
        self.assertIn("davies_bouldin_score", metrics)

    def test_pipeline_fits_without_using_labels(self) -> None:
        digits = load_digits()
        features, _labels = train_test_split(
            digits.data, test_size=0.9, random_state=8, stratify=digits.target
        )
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.9, svd_solver="full")),
                ("kmeans", KMeans(n_clusters=10, n_init=2, random_state=8)),
            ]
        )
        pipeline.fit(features)
        self.assertEqual(pipeline.named_steps["kmeans"].n_clusters, 10)


if __name__ == "__main__":
    unittest.main()
