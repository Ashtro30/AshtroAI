"""Shared fixtures: isolated settings (temp dirs, offline hashing embeddings) and a scripted fake LLM."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.config import Settings  # noqa: E402
from agent.llm import LLMError  # noqa: E402
from agent.memory import MemoryManager  # noqa: E402
from agent.tools import ToolContext  # noqa: E402


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    data = tmp_path / "data"
    shutil.copytree(ROOT / "data", data)
    return Settings(
        embedding_backend="hash",
        data_dir=data,
        outputs_dir=tmp_path / "outputs",
        state_dir=tmp_path / "state",
        max_iterations=6,
        max_tool_retries=3,
        python_timeout=8,
        episodic_memory=True,
        memory_min_score=0.2,
    ).with_overrides()


@pytest.fixture()
def memory(settings: Settings) -> MemoryManager:
    settings.ensure_dirs()
    return MemoryManager(settings)


@pytest.fixture()
def ctx(settings: Settings, memory: MemoryManager) -> ToolContext:
    return ToolContext(settings=settings, memory=memory, session_id="test-session")


class ScriptedLLM:
    """Fake chat model that replays a list of replies (dicts are JSON-encoded).

    An ``Exception`` instance in the script is raised instead of returned.
    The last reply repeats forever so loops can be tested.
    """

    def __init__(self, script: list[Any]) -> None:
        self.script = script
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        item = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, str) else json.dumps(item)


def act(tool: str, args: dict[str, Any], reason: str = "need a tool") -> dict[str, Any]:
    return {"reason": reason, "plan": ["use tool", "answer"], "action": {"tool": tool, "args": args}, "verify": "", "final_answer": None}


def answer(text: str, verify: str = "checked against observations") -> dict[str, Any]:
    return {"reason": "done", "plan": [], "action": None, "verify": verify, "final_answer": text}


__all__ = ["ScriptedLLM", "act", "answer", "LLMError"]
