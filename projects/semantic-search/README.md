# Semantic Search & RAG

A small Streamlit app for searching your own text and Markdown notes by meaning rather than exact keyword matches. It splits uploaded files into chunks, embeds them with a sentence-transformer model, and ranks the most relevant passages. Optionally, it can send those passages to a local Ollama model to draft a grounded answer.

## Run locally

Python 3.10+ is recommended. Install PyTorch appropriate for your machine from the [official instructions](https://pytorch.org/get-started/locally/), then install the app dependencies:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The embedding model downloads from Hugging Face on first run. Upload `.txt` or `.md` files in the browser, enter a question, and inspect the ranked results and source excerpts.

## Optional local answer generation

Semantic search works without a language model. To also generate an answer from the retrieved passages, install and start [Ollama](https://ollama.com/), then pull a model:

```bash
ollama pull llama3.2:3b
```

Enable **Generate an answer with local Ollama** in the app. Defaults are `http://127.0.0.1:11434` and `llama3.2:3b`; override them with `OLLAMA_HOST` and `OLLAMA_MODEL` environment variables. Ollama runs separately from this app. If it is unavailable, the search results remain usable and the app reports the generation error.

## How retrieval works

1. Uploaded files are decoded and split into overlapping character chunks.
2. A sentence-transformer encodes each chunk and the query.
3. Cosine similarity ranks the chunks; the highest scoring passages are shown with filenames and similarity scores.
4. If Ollama is enabled, those retrieved passages and the question are passed to the configured local model to draft an answer.

The index exists only in the current Streamlit session; it is not persisted or shared. Similarity scores are ranking signals, not calibrated confidence values.

## Notes and limitations

- This app processes uploaded documents in memory. It does not save uploads to a database or send them to a hosted embedding service.
- When optional Ollama generation is used, retrieved text is sent to the Ollama endpoint you configured. Keep it on a trusted local endpoint if documents are sensitive.
- Retrieved documents are untrusted input. The answer prompt asks the model to treat them as evidence, not instructions, but prompt injection is not fully preventable.
- Generated answers can be incomplete or incorrect. Check the returned source passages before relying on an answer.
- This is a lightweight demo; it has no authentication, durable vector store, or production monitoring.

## Project files

- `app.py` — Streamlit interface, chunking, semantic retrieval, and optional Ollama answer generation.
- `requirements.txt` — Python dependencies.
- `sample_docs/` — small example notes to try first.
