# Agent Workflow

A small, sequential agent-orchestration example for turning a question and supplied notes into a reviewed short brief. Four role prompts share one local Ollama model:

1. **Researcher** extracts relevant points from the notes and flags missing evidence.
2. **Writer** drafts a concise answer using only the supplied notes and research summary.
3. **Reviewer** checks the draft for unsupported claims, omissions, and clarity.
4. **Editor** revises the draft using the review while staying within the evidence.

This demonstrates explicit hand-offs and a bounded workflow—not autonomous agents. Agents do not browse the web, execute code, or call tools. Every stage receives user-provided material and returns text. The orchestration trace lists stage names and status only; it does not expose private model reasoning.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally
- A model pulled in Ollama, for example `llama3.2:3b`

Pull a model once:

```bash
ollama pull llama3.2:3b
```

No Python packages beyond the standard library are required.

## Run

From this directory:

```bash
python orchestrate.py \
  --task "Summarize the main causes of urban heat islands" \
  --context "Dark surfaces absorb more solar energy. Limited vegetation reduces shade and evaporative cooling. Waste heat from buildings and vehicles also contributes."
```

Set a model or local Ollama endpoint if needed:

```bash
OLLAMA_MODEL=llama3.2:3b OLLAMA_HOST=http://127.0.0.1:11434 \
  python orchestrate.py --task "Write a short project summary" --context "Project notes go here."
```

The default host is local (`127.0.0.1`). A model must already be available in Ollama. Model output can vary and should be checked against the source notes.

## Test the orchestration without a model

```bash
python -m unittest discover -s tests -v
```

Tests use a fake language model and do not require Ollama or network access.

## Design notes

- The workflow is fixed and sequential; no agent can add a new stage or delegate dynamically.
- Agents share a configured model but have separate role instructions and inputs.
- The researcher is instructed to use only supplied notes and identify gaps, not invent citations.
- The reviewer checks evidence alignment and clarity. It is a useful second pass, not a guarantee of correctness.
- There are no tools, shell commands, web searches, database writes, or external service calls beyond the configured Ollama chat endpoint.

For production use, add authentication and rate limits if exposed as a service, logging with appropriate privacy controls, evaluation data, model/version tracking, and human review for consequential output.
