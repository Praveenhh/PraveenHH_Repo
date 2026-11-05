"""Command-line runner for the local Ollama-backed agent workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

from workflow import AgentWorkflow


class OllamaChat:
    def __init__(self, host: str, model: str, timeout: int = 180) -> None:
        self.url = host.rstrip("/") + "/api/chat"
        self.model = model
        self.timeout = timeout

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {"temperature": 0.2},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"Could not reach Ollama at {self.url}. Start Ollama and verify the host. Details: {error}"
            ) from error
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise RuntimeError("Ollama returned an invalid chat response") from error

        try:
            content = result["message"]["content"]
        except (KeyError, TypeError) as error:
            raise RuntimeError("Ollama response did not contain message content") from error
        if not isinstance(content, str):
            raise RuntimeError("Ollama returned non-text message content")
        return content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="What the final brief should address")
    parser.add_argument("--context", required=True, help="Notes the agents may use as evidence")
    parser.add_argument("--show-trace", action="store_true", help="Print stage names/status after the answer")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = OllamaChat(
        host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
    )
    try:
        result = AgentWorkflow(model).run(args.task, args.context)
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(result.answer)
    if args.show_trace:
        print("\nWorkflow stages:")
        for stage in result.stages:
            print(f"- {stage.name}: {stage.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
