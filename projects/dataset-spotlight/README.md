# 🔎 Dataset Spotlight

A small, practical command-line project for exploring a dataset from the Hugging Face Hub and creating a shareable Markdown snapshot. It shows the dataset's columns, features, split size, and a few example rows—without uploading your data anywhere.

> This is an independent example project added to this repository. The repository itself is a fork of [🤗 Datasets](https://github.com/huggingface/datasets); this tool does not change or replace the upstream library.

## ✨ What it does

- Loads a dataset and split with 🤗 Datasets.
- Prints a compact overview of its columns and feature types.
- Previews a configurable number of examples.
- Optionally saves the overview as a Markdown report.

## 🚀 Quick start

From this directory, install the dependency:

```bash
python -m pip install -r requirements.txt
```

Explore a dataset:

```bash
python dataset_spotlight.py ag_news --split test --examples 2
```

Save a report:

```bash
python dataset_spotlight.py rotten_tomatoes --split validation --examples 3 --output report.md
```

For a private or gated dataset, first authenticate with Hugging Face using the official [user access token instructions](https://huggingface.co/docs/huggingface_hub/quick-start#authentication).

## 🧰 Options

```text
positional: dataset       Dataset ID, such as ag_news
--split                  Split to inspect (default: train)
--examples               Number of rows to preview (default: 3)
--output                 Optional path for a Markdown report
```

Use only datasets you trust. Dataset loading may involve downloading files and, depending on the dataset, running loading code. Review a dataset's Hub page and use a pinned revision when reproducibility matters.

## 📄 Example

```text
Dataset: ag_news
Split: test
Rows: 7600
Columns: text, label
Features:
- text: Value('string')
- label: ClassLabel(names=['World', 'Sports', 'Business', 'Sci/Tech'])
```

Exact output depends on the dataset version and installed 🤗 Datasets release.
