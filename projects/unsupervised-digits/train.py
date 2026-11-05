"""Fit and evaluate a K-means clustering pipeline on scikit-learn's digits data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/digits-kmeans"))
    parser.add_argument("--n-clusters", type=int, default=10)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.n_clusters < 2:
        parser.error("--n-clusters must be at least 2")
    if args.n_init < 1:
        parser.error("--n-init must be at least 1")
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")
    if not 0 < args.pca_variance <= 1:
        parser.error("--pca-variance must be in (0, 1]")
    return args


def evaluate_clusters(features: np.ndarray, clusters: np.ndarray, labels: np.ndarray) -> dict:
    """Compute internal and label-based metrics without assigning meaning to cluster IDs."""
    unique_clusters = np.unique(clusters)
    if len(unique_clusters) < 2 or len(unique_clusters) >= len(clusters):
        raise ValueError("Clustering evaluation requires between 2 and n_samples - 1 clusters")
    contingency = np.zeros((len(unique_clusters), 10), dtype=np.int64)
    for row, cluster_id in enumerate(unique_clusters):
        for digit in range(10):
            contingency[row, digit] = int(np.sum((clusters == cluster_id) & (labels == digit)))

    return {
        "silhouette_score": float(silhouette_score(features, clusters)),
        "calinski_harabasz_score": float(calinski_harabasz_score(features, clusters)),
        "davies_bouldin_score": float(davies_bouldin_score(features, clusters)),
        "adjusted_rand_index": float(adjusted_rand_score(labels, clusters)),
        "normalized_mutual_information": float(normalized_mutual_info_score(labels, clusters)),
        "homogeneity": float(homogeneity_score(labels, clusters)),
        "completeness": float(completeness_score(labels, clusters)),
        "v_measure": float(v_measure_score(labels, clusters)),
        "cluster_ids": [int(value) for value in unique_clusters],
        "cluster_sizes": {
            str(int(value)): int(np.sum(clusters == value)) for value in unique_clusters
        },
        "cluster_digit_contingency": {
            str(int(cluster_id)): {
                str(digit): int(contingency[index, digit]) for digit in range(10)
            }
            for index, cluster_id in enumerate(unique_clusters)
        },
    }


def main() -> None:
    args = parse_args()
    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=digits.target,
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=args.pca_variance, svd_solver="full")),
            (
                "kmeans",
                KMeans(n_clusters=args.n_clusters, n_init=args.n_init, random_state=args.seed),
            ),
        ]
    )
    pipeline.fit(x_train)  # Labels are deliberately not passed to fitting.
    transformed_test = pipeline[:-1].transform(x_test)
    test_clusters = pipeline.named_steps["kmeans"].predict(transformed_test)
    metrics = evaluate_clusters(transformed_test, test_clusters, y_test)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.output_dir / "kmeans_pipeline.joblib")
    report = {
        "dataset": "sklearn.datasets.load_digits",
        "task": "unsupervised clustering; labels used only for post-hoc evaluation",
        "n_samples": int(len(digits.data)),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "n_features": int(digits.data.shape[1]),
        "n_clusters_requested": args.n_clusters,
        "n_clusters_observed_test": int(len(np.unique(test_clusters))),
        "pca_variance_target": args.pca_variance,
        "pca_components": int(pipeline.named_steps["pca"].n_components_),
        "pca_explained_variance_ratio_sum": float(
            pipeline.named_steps["pca"].explained_variance_ratio_.sum()
        ),
        "seed": args.seed,
        "metrics": metrics,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
