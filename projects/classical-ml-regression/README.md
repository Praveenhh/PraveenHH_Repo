# Classical ML: Diabetes Progression Regression

A reproducible scikit-learn project comparing classical regression models on the diabetes dataset included with scikit-learn. The workflow uses a held-out test set, selects a candidate using cross-validation on training data only, evaluates the selected model once on the test set, and saves the fitted pipeline and machine-readable metrics.

This is a small educational benchmark dataset, not a medical diagnostic tool. The target is a quantitative disease-progression measure; predictions must not be interpreted as a diagnosis, treatment recommendation, or estimate for an individual patient.

## Dataset and methodology

The project uses `sklearn.datasets.load_diabetes`, a small tabular dataset distributed with scikit-learn. No separate dataset download is required. A fixed-seed train/test split is created first. Candidate models are compared using shuffled K-fold cross-validation exclusively within the training partition. The model with the lowest mean cross-validated root mean squared error (RMSE) is then refit on the full training partition and evaluated on the held-out test partition.

Compared estimators:

- Dummy mean regressor (reference baseline)
- Linear regression
- Ridge regression
- Random forest regression
- Histogram-based gradient boosting regression

All estimators are wrapped in a pipeline with feature standardization, keeping the preprocessing inside each cross-validation fold. Scaling is important for linear models; it is harmless though unnecessary for tree-based models.

## Setup

Python 3.10+ is recommended. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

From this directory:

```bash
python train.py --output-dir artifacts/diabetes-regression
```

Adjust the split, cross-validation, and seed:

```bash
python train.py --test-size 0.25 --cv-folds 5 --seed 23
```

The dataset is small, so the complete run is generally suitable for CPU execution. Results may vary with scikit-learn versions, split settings, and random seeds. No target score is promised.

## Evaluation metrics

The output `metrics.json` contains the selected model's held-out test metrics and the cross-validation results for every candidate:

- **MAE (mean absolute error):** average absolute prediction error in target units.
- **RMSE (root mean squared error):** square root of mean squared error, in target units; more sensitive to large errors than MAE.
- **R²:** coefficient of determination relative to predicting the test-set mean; it can be negative when predictions are worse than that reference.
- **Median absolute error:** median absolute prediction error, less influenced by extreme errors.
- **Explained variance:** variance captured by predictions; unlike R², it does not penalize systematic offset in the same way.

Cross-validation mean and standard deviation of RMSE support model comparison on training data. They are not substitutes for the held-out test evaluation. Use the test split once for final reporting, not repeated model selection.

## Outputs

```text
artifacts/diabetes-regression/
├── best_model.joblib
└── metrics.json
```

The saved artifact is a fitted scikit-learn pipeline. Load it with `joblib.load` using a compatible scikit-learn version. The JSON file records the selected estimator, split and seed settings, per-candidate CV RMSE summary, and final test metrics.

## Reproducibility and limitations

The train/test split and cross-validation shuffling use the configured seed, and estimators with randomness receive that seed. Small datasets have substantial sampling uncertainty; run multiple seeds and report variability for more rigorous comparison. Cross-validation selection across several models can still overfit the training partition, so keep the test set reserved for final evaluation.

The dataset is small and was not designed to support clinical decision-making. Metrics summarize performance on one split and do not establish clinical validity, fairness, or generalization to other populations. Do not use this project for diagnosis, treatment, or other consequential decisions.
