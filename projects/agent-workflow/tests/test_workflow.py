from __future__ import annotations

import unittest

from workflow import AgentWorkflow


class FakeModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return next(self.responses)


class AgentWorkflowTests(unittest.TestCase):
    def test_runs_fixed_stages_and_returns_editor_output(self) -> None:
        model = FakeModel(["research notes", "first draft", "review notes", "final brief"])

        result = AgentWorkflow(model).run("Summarize", "Source facts")

        self.assertEqual(result.answer, "final brief")
        self.assertEqual([stage.name for stage in result.stages], ["researcher", "writer", "reviewer", "editor"])
        self.assertEqual(len(model.calls), 4)
        self.assertIn("research notes", model.calls[1][1])
        self.assertIn("first draft", model.calls[2][1])
        self.assertIn("review notes", model.calls[3][1])

    def test_requires_context(self) -> None:
        model = FakeModel([])
        with self.assertRaisesRegex(ValueError, "Context must not be empty"):
            AgentWorkflow(model).run("Summarize", "  ")
        self.assertEqual(model.calls, [])

    def test_empty_agent_output_stops_workflow(self) -> None:
        model = FakeModel([" "])
        with self.assertRaisesRegex(RuntimeError, "researcher"):
            AgentWorkflow(model).run("Summarize", "Source facts")
        self.assertEqual(len(model.calls), 1)


if __name__ == "__main__":
    unittest.main()
