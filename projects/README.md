# Projects

Independent projects in this repository, grouped by the kind of work they explore. Each directory contains its own implementation, dependencies, and project-specific notes.

## Generative AI and retrieval

| Project | Description | Entry point |
| --- | --- | --- |
| [Semantic Search](semantic-search/) | Search uploaded text and Markdown by semantic similarity; optionally draft answers using a local Ollama model. | [`app.py`](semantic-search/app.py) |
| [Agent Workflow](agent-workflow/) | Fixed researcher, writer, reviewer, and editor stages using a local Ollama model. | [`orchestrate.py`](agent-workflow/orchestrate.py) |
| [Everyday Chat](everyday-chat/) | Browser-based conversation with a locally run open-weight model. | [`app.py`](everyday-chat/app.py) |

## NLP and language models

| Project | Description | Entry point |
| --- | --- | --- |
| [Transformer Classification](transformer-classification/) | Fine-tune DistilBERT for binary sentiment classification on IMDb. | [`train.py`](transformer-classification/train.py) |
| [BERT SST-2](bert-sst2/) | Fine-tune BERT for sentiment classification on the GLUE SST-2 dataset. | [`train.py`](bert-sst2/train.py) |
| [GPT-2 Text Generation](gpt2-text-generation/) | Generate a continuation from a text prompt with pretrained GPT-2. | [`generate.py`](gpt2-text-generation/generate.py) |

## Deep learning and data tools

| Project | Description | Entry point |
| --- | --- | --- |
| [CNN Image Classification](cnn-cifar10/) | Train a compact CNN on CIFAR-10, select by validation accuracy, then report test and per-class metrics with plots. | [`train.py`](cnn-cifar10/train.py) |
| [Tiny Memory Lab](tiny-memory-lab/) | Synthetic sequence-recall experiment using an LSTM and variable distractor delays. | [`train_memory.py`](tiny-memory-lab/train_memory.py) |
| [Dataset Spotlight](dataset-spotlight/) | Inspect dataset features and examples; optionally save a Markdown report. | [`dataset_spotlight.py`](dataset-spotlight/dataset_spotlight.py) |

## Working with these projects

Projects are independent rather than a single installable package. Use the README and dependency file inside the relevant project directory. Pretrained model and dataset downloads may require internet access and additional disk space. Confirm licenses and hardware requirements before use.

The applications are examples and experiments; their availability in this repository does not imply production readiness or benchmark performance. Check implementation status and limitations in each project's documentation.

## Repository structure

```text
.
├── projects/
│   ├── agent-workflow/
│   ├── bert-sst2/
│   ├── cnn-cifar10/
│   ├── dataset-spotlight/
│   ├── everyday-chat/
│   ├── gpt2-text-generation/
│   ├── semantic-search/
│   ├── tiny-memory-lab/
│   └── transformer-classification/
├── src/                 # Upstream Hugging Face Datasets fork
├── docs/                # Upstream documentation
└── notebooks/           # Upstream notebooks
```
