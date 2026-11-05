# GRU Sequence Recall

A reproducible PyTorch experiment for evaluating a Gated Recurrent Unit (GRU) on delayed binary recall. Each sequence marks a binary value, presents distractor values, and ends with a query token. The model must classify the marked value after a variable delay.

This project extends the sequence-recall example with a multi-layer GRU, deterministic validation and test sets, validation-based checkpoint selection, per-delay analysis, class-wise metrics, and machine-readable artifacts. It is a synthetic modeling experiment—not a benchmark of human memory or evidence of general-purpose sequence understanding.

## Task and inputs

Each timestep has three features: the current binary value, a marker flag, and a query flag. The marked value is the target. Delay lengths range from 1 to `--max-delay`; distractor values are generated independently. Variable-length sequences are packed so padding is excluded from GRU processing.

Training examples are generated on the fly. Separate seeded validation and test sets are generated once, with delays distributed evenly, to make comparisons across epochs and delay lengths easier to interpret.

## Requirements

Python 3.10+ and PyTorch. Install the project dependency:

```bash
python -m pip install -r requirements.txt
```

Install a PyTorch build suitable for your CPU/GPU from the [official PyTorch instructions](https://pytorch.org/get-started/locally/) if needed. No external dataset is downloaded.

## Train and evaluate

From this directory:

```bash
python train.py --output-dir artifacts/gru-recall
```

Try a longer delay range:

```bash
python train.py --max-delay 20 --epochs 20 --train-size 12000 --validation-size 3000 --test-size 3000
```

Select a device explicitly if needed:

```bash
python train.py --device cpu
python train.py --device cuda
```

The best checkpoint is selected by validation accuracy. The test set is evaluated once after training using that checkpoint; do not use test results to tune hyperparameters. Accuracy is an empirical result and can vary with hardware, software versions, and settings.

## Evaluation metrics

`metrics.json` reports held-out test performance:

- **Accuracy** and **balanced accuracy**.
- **Macro and weighted precision, recall, and F1** for the two target classes.
- **ROC-AUC** and **average precision** from the positive-class probabilities.
- **Per-class precision, recall, F1, and support**.
- **Confusion matrix** with rows as true labels and columns as predictions.
- **Accuracy by delay**, with counts, to show how performance changes with retention interval.

The output also records the selected checkpoint's validation accuracy, seed, model configuration, and sample counts. ROC-AUC and average precision measure ranking behavior, not probability calibration. A balanced synthetic test set does not establish real-world generalization.

## Outputs

```text
artifacts/gru-recall/
├── best_model.pt
├── metrics.json
└── training_history.json
```

`training_history.json` includes train loss and validation loss/accuracy for each epoch. The checkpoint stores the model weights and configuration. Training output is not saved in the repository by default.

## Configuration

| Option | Default | Description |
| --- | ---: | --- |
| `--max-delay` | `12` | Maximum distractor steps before the query |
| `--epochs` | `12` | Maximum training epochs |
| `--train-size` | `8000` | Generated training examples per epoch |
| `--validation-size` | `2000` | Fixed validation examples |
| `--test-size` | `2000` | Fixed held-out test examples |
| `--batch-size` | `128` | Examples per batch |
| `--hidden-size` | `64` | GRU hidden-state width |
| `--num-layers` | `2` | Number of stacked GRU layers |
| `--learning-rate` | `0.001` | Adam learning rate |
| `--patience` | `4` | Stop after this many epochs without validation improvement |
| `--seed` | `17` | Random seed |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |

## Reproducibility and limitations

The seed controls data generation and model initialization, but exact reproducibility can still depend on device and library versions. For model comparisons, use the same generated data and run multiple seeds; report variation rather than a single best run. Compare against the LSTM sequence-recall project using matched data-generation settings before drawing conclusions about architecture differences.

This binary task is deliberately narrow. It does not show general memory, language understanding, or performance on real sequential data. Results depend on the synthetic task distribution; inspect per-delay and per-class metrics and avoid broader claims.
