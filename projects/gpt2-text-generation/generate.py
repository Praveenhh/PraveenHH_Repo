"""Generate a continuation with a pretrained GPT-2 causal language model."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

DEFAULT_MODEL = "openai-community/gpt2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="Text prompt to continue")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--no-sample", action="store_true", help="Use greedy decoding instead of sampling"
    )
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than 0")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in the range (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available in this PyTorch installation")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if context_limit is None:
        context_limit = getattr(model.config, "n_positions", None)
    if context_limit is not None and input_ids.shape[1] + args.max_new_tokens > context_limit:
        raise SystemExit(
            f"Prompt ({input_ids.shape[1]} tokens) plus requested continuation "
            f"({args.max_new_tokens} tokens) exceeds this model's context window "
            f"({context_limit} tokens). Shorten the prompt or reduce --max-new-tokens."
        )

    generation_options = {
        "input_ids": input_ids,
        "attention_mask": encoded["attention_mask"].to(device),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": not args.no_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if not args.no_sample:
        generation_options.update(temperature=args.temperature, top_p=args.top_p)

    with torch.inference_mode():
        output_ids = model.generate(**generation_options)

    continuation_ids = output_ids[0, input_ids.shape[1] :]
    continuation = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
    print(continuation or "[The model did not generate a continuation.] ")


if __name__ == "__main__":
    main()
