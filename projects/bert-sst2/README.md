# BERT on SST-2

A small fine-tuning experiment using `bert-base-uncased` for binary sentiment classification on the GLUE SST-2 dataset. The script fine-tunes the pretrained encoder, evaluates on the validation split, and saves the model and tokenizer.

## Setup

Python 3.10+ is recommended. Install a PyTorch build for your hardware using the [official instructions](https://pytorch.org/get-started/locally/), then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The dataset and pretrained model are downloaded from the Hugging Face Hub on first use. Check the dataset and model cards for their terms and licenses.

## Fine-tune

```bash
python train.py --output-dir artifacts/bert-sst2
```

Run a smaller experiment to check the pipeline:

```bash
python train.py --max-train-samples 1000 --epochs 1
```

By default, the script trains on the SST-2 training split and evaluates on its validation split. It reports accuracy and weighted F1 and saves the final model, tokenizer, and `metrics.json` in the output directory.

## Predict

After training:

```bash
python predict.py --model-dir artifacts/bert-sst2 "The film was thoughtful and well acted."
```

## Reproducibility and limitations

Set `--seed` to control data shuffling and training seeds. Results can vary with hardware, library versions, and training settings. For comparisons, use multiple seeds and report variation. SST-2 is a binary English sentiment dataset; results do not imply performance on other domains, languages, or nuanced sentiment. Treat predictions as model outputs, not ground truth.
