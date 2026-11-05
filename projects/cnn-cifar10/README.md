# CIFAR-10 CNN Image Classification

An end-to-end image-classification experiment using a compact convolutional neural network (CNN) trained on CIFAR-10. The workflow covers reproducible data preparation, train/validation separation, training with validation-based checkpoint selection, and a final evaluation on the held-out test split.

The script reports multiple complementary classification metrics and saves machine-readable results, a confusion matrix, and training curves. It does not claim benchmark performance; results depend on the architecture, hyperparameters, software versions, and hardware.

## Dataset and task

CIFAR-10 contains 60,000 color images at 32×32 pixels across ten classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. This project uses the official 50,000-image training split, creates a validation subset from that split, and evaluates the selected checkpoint once on the official 10,000-image test split.

Images are normalized using CIFAR-10 channel statistics. Random horizontal flips and random crops are applied only to training data.

## Requirements

Python 3.10+ is recommended. Install PyTorch and Torchvision builds appropriate for your hardware using the [official instructions](https://pytorch.org/get-started/locally/), then install the other dependencies:

```bash
python -m pip install -r requirements.txt
```

The CIFAR-10 dataset is downloaded on first run. A GPU is optional; CPU training is supported but may take longer.

## Train and evaluate

From this directory:

```bash
python train.py --data-dir data --output-dir artifacts/cifar10-cnn
```

For a smaller run to verify the workflow:

```bash
python train.py --epochs 2 --limit-train-samples 5000 --limit-test-samples 1000
```

Set a fixed seed and device explicitly if needed:

```bash
python train.py --seed 17 --device cpu
python train.py --seed 17 --device cuda
```

The best checkpoint is selected by validation accuracy. The test metrics are computed after training from that checkpoint; do not use the test split for hyperparameter tuning. The full default run can require significant time and compute.

## Evaluation metrics

`metrics.json` contains the following held-out test metrics:

- **Accuracy** — fraction of all test images classified correctly.
- **Balanced accuracy** — mean recall across classes.
- **Macro precision, recall, and F1** — class metrics averaged equally across the ten classes.
- **Weighted precision, recall, and F1** — class metrics averaged by class support.
- **Top-3 accuracy** — fraction of examples whose true class is among the three highest-scoring predictions.
- **Macro one-vs-rest ROC-AUC** — macro average of class-vs-rest ROC-AUC values.
- **Macro one-vs-rest average precision (AP)** — macro average of class-vs-rest AP values.
- **Per-class precision, recall, F1, and support** — detailed results for each label.
- **Confusion matrix** — true labels by predicted labels, saved both as JSON and PNG.

ROC-AUC and AP are computed from softmax probabilities. These probability-based metrics describe class ranking, not probability calibration. All reported values should be treated as experiment results, not guaranteed model quality.

## Outputs

The output directory contains:

```text
artifacts/cifar10-cnn/
├── best_model.pt
├── metrics.json
├── training_history.json
├── confusion_matrix.png
└── training_history.png
```

The checkpoint includes the model state, architecture settings, class names, and seed. `predict.py` can load the saved checkpoint for individual image inference.

## Predict an image

```bash
python predict.py --checkpoint artifacts/cifar10-cnn/best_model.pt path/to/image.png
```

The image is resized and normalized using the same evaluation preprocessing as the test set. Predictions are specific to CIFAR-10's ten classes; arbitrary high-resolution images may not resemble the training distribution.

## Model and experiment details

The CNN uses two convolutional feature blocks with batch normalization, ReLU activations, and max pooling, followed by a compact fully connected classification head. Training uses cross-entropy loss and Adam. The validation set is drawn from the training split with a fixed seed, and the checkpoint with the best validation accuracy is retained.

The default settings are intentionally modest and configurable. For a meaningful model comparison, record the environment and run multiple seeds; report average performance and variation rather than selecting a favorable run.

## Limitations

- CIFAR-10 is a small, low-resolution benchmark dataset and may not represent real deployment images.
- The CNN is a baseline architecture, not a state-of-the-art model.
- Aggregate metrics can conceal weak performance on particular classes; inspect the per-class report and confusion matrix.
- Test-set results should not guide repeated model selection.
- Predictions may be wrong. Do not use this experiment for consequential classification decisions.
