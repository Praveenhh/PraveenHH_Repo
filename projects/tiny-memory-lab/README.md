# Tiny Memory Lab: LSTM Sequence Recall

A small PyTorch experiment for studying short-term sequence memory. An LSTM sees a binary value marked near the beginning of each sequence, processes randomly generated distractor values, then receives a query token. It must recall the marked bit after a variable delay.

This is a synthetic learning exercise—not a benchmark of human memory or a claim about production model performance. It trains from generated data and does not download datasets or pretrained weights.

## Task design

Each timestep has three input features:

1. the current binary value,
2. a marker bit that identifies the value to remember,
3. a query bit that marks the end of the sequence.

The target is the binary value from the marked first timestep. Delay length varies between examples, so the model must preserve the information while processing intervening distractors. The script reports held-out accuracy overall and by delay bucket.

## Requirements

Python 3.10+ and PyTorch. Install the project dependency:

```bash
python -m pip install -r requirements.txt
```

Install a PyTorch build suitable for your CPU/GPU from the [official PyTorch instructions](https://pytorch.org/get-started/locally/) if needed.

## Train and evaluate

From this directory:

```bash
python train_memory.py
```

Try a longer retention interval and more training:

```bash
python train_memory.py --max-delay 20 --epochs 15 --train-size 12000 --test-size 3000
```

Use a GPU when available:

```bash
python train_memory.py --device cuda
```

The script selects CUDA automatically when available unless `--device` is specified. Training generates examples on the fly, uses a fixed random seed by default, prints loss and test accuracy per epoch, and prints test accuracy grouped by delay at the end. Results vary with hardware, PyTorch version, and hyperparameters; no target score is promised.

## Options

| Option | Default | Description |
| --- | ---: | --- |
| `--max-delay` | `12` | Maximum number of distractor steps before the query |
| `--epochs` | `8` | Number of training epochs |
| `--train-size` | `8000` | Training examples generated per epoch |
| `--test-size` | `2000` | Held-out evaluation examples |
| `--batch-size` | `128` | Examples per batch |
| `--hidden-size` | `64` | LSTM hidden-state width |
| `--seed` | `7` | Random seed for repeatability |
| `--device` | auto | `auto`, `cpu`, or `cuda` |

## Model and evaluation

The model is a one-layer LSTM followed by a two-class linear classifier on the final hidden state. Variable-length sequences are packed so padding does not enter the recurrent computation. The test set is generated separately from the training batches. Per-delay accuracy helps reveal whether recall becomes harder as distractors increase.

This compact setup intentionally omits hyperparameter search, multiple seeds, confidence intervals, and baseline comparisons. For a careful experiment, run multiple seeds and compare against a simple baseline, such as a model that always predicts the majority class, then report mean and variability.

## Limitations

- Synthetic binary recall is deliberately narrow; results do not imply general memory, language understanding, or real-world sequence performance.
- The task reveals one behavior under one data-generation process. Change the task, delay distribution, or model before drawing broader conclusions.
- Accuracy is an empirical metric on generated examples, not a guarantee of reliability.
