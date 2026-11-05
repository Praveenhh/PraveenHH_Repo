# 💬 Everyday Chat — Open-Weight Conversation Demo

A friendly, browser-based chatbot for ordinary conversation, powered by the open-weight [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) model. Ask a question, brainstorm, or just say hello—the app keeps a short conversation context so replies feel connected.

> This is a standalone showcase project in this repository. It is not part of the upstream 🤗 Datasets library and does not call a hosted chat API. The model weights are downloaded from Hugging Face when you first start the app, then inference runs on your machine.

## ✨ Highlights

- Clean Gradio chat interface with example prompts and a conversation reset.
- Short rolling chat history for contextual follow-up messages.
- Adjustable response length.
- Uses an open-weight model with an Apache-2.0 license.
- No API key required; no chat messages are sent to a third-party inference service by this app.

## 🖥️ Run locally

Python 3.10 or newer is recommended. Install the dependencies in a virtual environment:

```bash
python -m pip install -r requirements.txt
python app.py
```

Open the local URL printed by Gradio in your browser. The first launch downloads the model weights and may take a little while. Model inference needs compatible hardware and memory; performance depends on your machine.

By default, the app binds locally and does not enable Gradio's public sharing option. Do not expose it to the internet without adding appropriate access controls.

## ⚙️ Choose another compatible model

Set `CHAT_MODEL_ID` before starting the app to use another Hugging Face causal chat model that supports the tokenizer chat template:

```bash
CHAT_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct python app.py
```

Check the selected model's license, hardware requirements, and model-card guidance before using it.

## 🧠 How it works

1. The app loads a tokenizer and causal language model from Hugging Face.
2. It builds a chat prompt from a friendly assistant instruction, recent turns, and your message.
3. The model generates a reply locally, and Gradio displays it in the chat window.

The app keeps only the most recent eight conversation messages in the prompt to limit context growth. Use **Clear** in the chat interface to start over.

## 🔐 Privacy and limitations

- Chat content is processed locally by the app; model files are fetched from Hugging Face on first use.
- Avoid entering passwords, personal information, or other sensitive data.
- The model can produce incorrect, biased, or made-up information. Verify important claims.
- This is a demo, not a medical, legal, financial, or emergency advice service.

## 📜 Model and license

The default model is [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), distributed under Apache-2.0 according to its model card. Review the model card and license for the exact version you download. The code in this demo is also provided under Apache-2.0.
