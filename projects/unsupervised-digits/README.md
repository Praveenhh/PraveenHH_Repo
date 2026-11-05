# Unsupervised Learning: Digits Clustering

An end-to-end scikit-learn experiment that groups handwritten digit images without using their labels during fitting. The pipeline standardizes pixel features, reduces dimensionality with PCA, and applies K-means clustering. A held-out split is assigned to the fitted centroids for evaluation.

Digit labels are used only after clustering to measure how clusters relate to known classes. Cluster IDs are arbitrary and are not treated as predicted digit labels. This is an exploratory clustering example, not a handwriting recognition classifier.

## Dataset and workflow

The project uses the built-in scikit-learn `load_digits` dataset (1,797 grayscale 8×8 images across ten digit classes). No dataset download is required.

1. Split examples into train and test partitions with a fixed seed.
2. Fit feature standardization and PCA on training features only.
3. Fit K-means with a configurable number of clusters on the transformed training features.
4. Assign test examples to the nearest learned centroid.
5. Report internal clustering metrics and external metrics against test labels, strictly for post-hoc analysis.

The default cluster count is ten to make comparison to the ten known digit classes possible; K-means does not know those classes during fitting. K-means may converge to a local optimum, so `--n-init` controls repeated initializations.

## Setup and run

Python 3.10+ is recommended. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the full experiment from this directory:

```bash
python train.py --output-dir artifacts/digits-kmeans
```

Change the number of clusters or seed:

```bash
python train.py --n-clusters 12 --n-init 30 --seed 23
```

## Evaluation metrics

`metrics.json` reports test-set results:

### Internal metrics (do not use class labels)

- **Silhouette score** — cohesion and separation, from -1 to 1; larger is generally better.
- **Calinski–Harabasz score** — between-cluster dispersion relative to within-cluster dispersion; larger is generally better.
- **Davies–Bouldin score** — average cluster similarity; smaller is generally better.
- **Inertia** — within-cluster sum of squared distances to centroids; it tends to decrease as cluster count increases and should not be used alone to choose `k`.

### External metrics (labels used only for evaluation)

- **Adjusted Rand index (ARI)** — pairwise agreement adjusted for chance; 1 indicates matching partitions, near 0 indicates chance-level agreement.
- **Normalized mutual information (NMI)** — normalized information shared between cluster assignments and known labels.
- **Homogeneity, completeness, and V-measure** — describe whether clusters contain one class, whether each class is captured by one cluster, and their harmonic mean.

The report also includes cluster sizes and a cluster-to-class contingency table. Cluster IDs have no intrinsic meaning, so a cluster's numeric ID should not be interpreted as a digit label.

## Outputs

```text
artifacts/digits-kmeans/
├── kmeans_pipeline.joblib
└── metrics.json
```

The saved pipeline includes the fitted scaler, PCA transform, and K-means estimator. It can assign new 8×8 digit feature vectors to the closest cluster. Use a compatible scikit-learn version to load the joblib artifact.

## Limitations

- K-means favors roughly compact, similarly sized clusters and may not fit the structure of digit images well.
- PCA discards some information; the retained variance is configurable.
- Results depend on the split, seed, preprocessing, and cluster count. Compare multiple seeds and values of `k` before drawing conclusions.
- The dataset is small and curated. Results do not establish performance on unconstrained handwriting or other image data.
