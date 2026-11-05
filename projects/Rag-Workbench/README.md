# Praveen H. | Data Scientist · Applied AI

<p align="center">
  <b>4 years of data science experience</b> · Machine Learning · NLP · LLMs · RAG · AI Agents
</p>

I build practical data and AI systems—from classical machine learning and deep learning to retrieval-augmented generation and agent workflows. This repository is my project showcase alongside a fork of [🤗 Hugging Face Datasets](https://github.com/huggingface/datasets).

> The experience and skills above reflect my background. The demos below are independent portfolio projects; they are not official Hugging Face projects. Please see each project's README for implementation scope and limitations.

## Featured work

| Project | Focus | What it demonstrates |
| --- | --- | --- |
| [RAG Evaluation Workbench](projects/rag-workbench/) | RAG · NLP · FastAPI · Docker | A local document Q&A API with semantic retrieval, source references, and a small test suite. |
| [Everyday Chat](projects/everyday-chat/) | LLM · Transformers · Gradio | A local conversational demo using an open-weight instruction model. |
| [Dataset Spotlight](projects/dataset-spotlight/) | Data engineering · Hugging Face Datasets | Dataset metadata inspection and Markdown report generation. |

## Areas of experience

- **Generative AI:** RAG pipelines, prompt design, LLM applications, agent orchestration, and A2A concepts.
- **NLP and language models:** Transformers, BERT, GPT-2, text classification, and language processing.
- **Machine learning and deep learning:** model development, evaluation, and neural network architectures including CNNs, RNNs, LSTMs, and GRUs.
- **Applied data science:** dataset exploration, feature workflows, reproducible experiments, and communicating results.

## Repository map

```text
projects/
├── rag-workbench/       # Document Q&A API, Docker, tests
├── everyday-chat/       # Local open-weight chat demo
└── dataset-spotlight/   # Dataset inspection utility


# 🧭 RAG Evaluation Workbench

A small, runnable Retrieval-Augmented Generation (RAG) API for exploring the core pipeline: ingest text, split it into overlapping chunks, embed and retrieve relevant passages, then generate an answer grounded in the retrieved context. Answers include source references so retrieval can be inspected.

## What this demo shows

- FastAPI endpoints for adding text documents and asking questions.
- Sentence-Transformers embeddings and cosine-similarity retrieval.
- Configurable chunk size, overlap, and top-k retrieval.
- A local Hugging Face text-to-text generation model; no hosted LLM API key.
- Source snippets and similarity scores in each response.
- Unit/API tests using small fake models so CI does not download model weights.
- Docker support for a reproducible local demo.

## Run with Docker

```bash
docker compose up --build
