# GPT-2 Text Generation

A small command-line example that generates a continuation from the pretrained GPT-2 model using Hugging Face Transformers. It takes a prompt, samples new tokens, and prints only the generated continuation.

GPT-2 is a general-purpose next-token prediction model, not an instruction-tuned chatbot. It may continue a prompt in unexpected ways or produce inaccurate content. This project demonstrates text generation; it does not fine-tune the model.

## Setup

Python 3.10+ is recommended. Install a PyTorch build suitable for your hardware using the [official instructions](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

The model weights and tokenizer are downloaded from the Hugging Face Hub on first run. Review the [GPT-2 model card](https://huggingface.co/openai-community/gpt2) and license before use.

## Generate text

```bash
python generate.py "Once upon a time in a small seaside town"
```

Control generation length and sampling:

```bash
python generate.py "The future of renewable energy" --max-new-tokens 120 --temperature 0.8 --top-p 0.92 --seed 17
```

Use deterministic greedy decoding instead of sampling:

```bash
python generate.py "A good data scientist" --max-new-tokens 60 --no-sample
```

Choose a device explicitly if needed:

```bash
python generate.py "A walk through the forest" --device cpu
python generate.py "A walk through the forest" --device cuda
```

## Options

| Option | Default | Description |
| --- | ---: | --- |
| `--model-name` | `openai-community/gpt2` | Hugging Face causal language model ID |
| `--max-new-tokens` | `80` | Maximum number of generated tokens |
| `--temperature` | `0.8` | Sampling temperature; higher values increase variation |
| `--top-p` | `0.92` | Nucleus sampling probability threshold |
| `--seed` | `17` | Random seed for sampled generation |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--no-sample` | off | Use greedy decoding rather than sampling |

The script checks that the prompt and requested continuation fit within the selected model's context window. If they do not, shorten the prompt or reduce `--max-new-tokens`.

## Notes

- Outputs are generated text, not factual answers; verify claims independently.
- Generation quality and speed depend on the prompt, decoding settings, hardware, and model version.
- The pretrained GPT-2 model can produce biased or unsuitable text. Do not treat generated content as professional advice.
- This project loads a pretrained checkpoint and does not train or fine-tune it.
