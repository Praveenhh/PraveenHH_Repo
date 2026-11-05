# Transformer Text Classification

Fine-tune a pretrained DistilBERT model for binary sentiment classification on the IMDb movie-review dataset. The training script uses the Hugging Face `datasets` and `transformers` libraries, evaluates on a held-out split, and saves the model and tokenizer for later inference.

This is a compact learning project. Results depend on the chosen model, hardware, and training settings; run multiple seeds and compare baselines before making research or production claims.

## Setup

Python 3.10+ is recommended. Install PyTorch using the [official instructions](https://pytorch.org/get-started/locally/) for your hardware, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The dataset and pretrained model are downloaded from the Hugging Face Hub on first use. Check their licenses and model cards before using or redistributing them.

## Train and evaluate

```bash
python train.py --output-dir artifacts/distilbert-imdb
```

The script fine-tunes `distilbert/distilbert-base-uncased-finetuned-sst-2-english` on the IMDb training split and evaluates on the test split, reporting accuracy and weighted F1. To train on fewer examples for a quick experiment:

```bash
python train.py --max-train-samples 2000 --max-eval-samples 1000 --epochs 1
```

Choose a compatible device with `--device cpu`, `--device cuda`, or `--device auto` (the default). GPU availability and memory determine practical batch sizes and run time.

## Predict with a saved model

After training:

```bash
python predict.py --model-dir artifacts/distilbert-imdb "The story was moving and beautifully acted."
```

## Reproducibility

The training script accepts a seed and records the model, dataset split, training arguments, and evaluation metrics in `metrics.json` alongside the saved model. For more rigorous comparisons, run multiple seeds and report the mean and variation. The test split should be reserved for final evaluation rather than repeated tuning.

## Project contents

- `train.py` — dataset loading, tokenization, fine-tuning, evaluation, and artifact saving.
- `predict.py` — load a saved classifier and classify one or more review strings.
- `requirements.txt` — project dependencies.

## Limitations

IMDb sentiment labels are binary and reflect movie reviews; performance does not imply reliable sentiment analysis in other domains or languages. Text classifiers can inherit dataset biases and fail on sarcasm, mixed sentiment, or distribution shifts. Inspect predictions and evaluate on representative data before real use.
