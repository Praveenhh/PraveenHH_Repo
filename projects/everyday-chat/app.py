"""A small local chat app powered by an open-weight Hugging Face model."""

from __future__ import annotations

import os

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("CHAT_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
SYSTEM_MESSAGE = (
    "You are a friendly, thoughtful assistant for everyday conversation. "
    "Reply naturally and clearly. Be concise by default, ask a follow-up when useful, "
    "and say when you are unsure instead of inventing facts."
)
MAX_HISTORY_MESSAGES = 8


def load_chat_model(model_id: str):
    """Load tokenizer and model once when the app starts."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


TOKENIZER, MODEL = load_chat_model(MODEL_ID)


def chat(message: str, history: list[dict], max_new_tokens: int = 256) -> str:
    """Generate a response using the latest turns from the current chat."""
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]

    for turn in history[-MAX_HISTORY_MESSAGES:]:
        role = turn.get("role")
        content = turn.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": message})
    input_ids = TOKENIZER.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(MODEL.device)

    with torch.inference_mode():
        output_ids = MODEL.generate(
            input_ids,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=TOKENIZER.eos_token_id,
        )

    new_tokens = output_ids[0, input_ids.shape[-1] :]
    answer = TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()
    return answer or "I couldn't come up with a response just now. Could you try asking another way?"


with gr.Blocks(theme=gr.themes.Soft(), title="Everyday Chat") as demo:
    gr.Markdown(
        "# 💬 Everyday Chat\n"
        "A simple, friendly conversation demo running with an open-weight model on your machine."
    )
    gr.ChatInterface(
        fn=chat,
        type="messages",
        chatbot=gr.Chatbot(height=460, type="messages"),
        textbox=gr.Textbox(
            placeholder="Say hello, ask a question, or start a conversation…",
            lines=2,
            max_lines=5,
        ),
        additional_inputs=[
            gr.Slider(
                minimum=64,
                maximum=512,
                value=256,
                step=32,
                label="Maximum reply length (tokens)",
            )
        ],
        examples=[
            "Hey! How's your day going?",
            "Can you suggest a simple weekend hobby?",
            "Help me plan a relaxed movie night.",
        ],
    )


if __name__ == "__main__":
    demo.launch()
