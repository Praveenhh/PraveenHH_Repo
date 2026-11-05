"""Compare classical regression models on scikit-learn's diabetes dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return complementary regression metrics in the target's original units."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "median_absolute_error": float(median_absolute_error(actual, predicted)),
        "explained_variance": float(explained_variance_score(actual, predicted)),
    }


def build_candidates(seed: int) -> dict[str, object]:
    """Construct reproducible model pipelines with preprocessing inside CV."""
    return {
        "dummy_mean": make_pipeline(StandardScaler(), DummyRegressor(strategy="mean")),
        "linear_regression": make_pipeline(StandardScaler(), LinearRegression()),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": make_pipeline(
            StandardScaler(),
            RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1),
        ),
        "hist_gradient_boosting": make_pipeline(
            StandardScaler(), HistGradientBoostingRegressor(max_iter=150, random_state=seed)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/diabetes-regression"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")
    return args


def main() -> None:
    args = parse_args()
    dataset = load_diabetes(as_frame=True)
    features = dataset.data
    target = dataset.target
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=args.test_size, random_state=args.seed
    )
    if args.cv_folds > len(x_train):
        raise SystemExit("--cv-folds cannot exceed the number of training samples")

    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    candidates = build_candidates(args.seed)
    comparisons: dict[str, dict[str, float]] = {}
    for name, estimator in candidates.items():
        scores = -cross_val_score(
            estimator,
            x_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        comparisons[name] = {
            "cv_rmse_mean": float(np.mean(scores)),
            "cv_rmse_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        }
        print(
            f"{name}: CV RMSE={comparisons[name]['cv_rmse_mean']:.3f} "
            f"+/- {comparisons[name]['cv_rmse_std']:.3f}"
        )

    selected_name = min(comparisons, key=lambda name: comparisons[name]["cv_rmse_mean"])
    selected_model = candidates[selected_name]
    selected_model.fit(x_train, y_train)
    predictions = selected_model.predict(x_test)
    test_metrics = regression_metrics(np.asarray(y_test), predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, args.output_dir / "best_model.joblib")
    results = {
        "dataset": "sklearn.datasets.load_diabetes",
        "target": "quantitative disease-progression measure",
        "selected_model": selected_name,
        "selection_metric": "lowest mean K-fold training RMSE",
        "seed": args.seed,
        "test_size_fraction": args.test_size,
        "cv_folds": args.cv_folds,
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "candidate_cv_results": comparisons,
        "held_out_test_metrics": test_metrics,
        "feature_names": list(features.columns),
        "warning": "Research/educational dataset only; not for diagnosis or clinical decisions.",
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSelected model: {selected_name}")
    print("Held-out test metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
