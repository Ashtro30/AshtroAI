"""Verifies the eval harness end-to-end with an *oracle* LLM that follows the ideal tool path.

The oracle answers from real tool observations (not hard-coded numbers), so this also proves that the
sample data, the real tools and the benchmark ground truth agree with each other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from eval import run_eval
from tests.conftest import act, answer

TASKS = json.loads((Path(run_eval.EVAL_DIR) / "benchmark_tasks.json").read_text())["tasks"]

Step = Callable[[list[str]], dict[str, Any]]


def _obs(messages: list[dict[str, str]]) -> list[str]:
    out = []
    for m in messages:
        if m["role"] == "user" and m["content"].startswith("OBSERVATION from"):
            body = m["content"].split(":\n", 1)[1].rsplit("\n\nNow verify", 1)[0]
            out.append(body)
    return out


CSV = "df = load_csv('quarterly_sales.csv')\n"
SCRIPTS: dict[int, list[Step]] = {
    1: [lambda o: act("search_documents", {"query": "remote work policy core hours"}), lambda o: answer(f"Handbook says:\n{o[-1]}")],
    2: [lambda o: act("search_documents", {"query": "Aurora Sensor Hub radio range battery capacity"}), lambda o: answer(f"Spec:\n{o[-1]}")],
    3: [lambda o: act("python_eval", {"code": "print(round(15000*(1+0.075/12)**144, 2))"}), lambda o: answer(f"Final value: {o[-1]}")],
    4: [
        lambda o: act("python_eval", {"code": "f=[0,1]\nwhile len(f)<10: f.append(f[-1]+f[-2])\np=[]\nn=2\nwhile len(p)<20:\n    if all(n%d for d in range(2,int(n**.5)+1)): p.append(n)\n    n+=1\nprint(sum(f), p[-1])"}),
        lambda o: answer(f"Results (sum, 20th prime): {o[-1]}"),
    ],
    5: [lambda o: act("python_eval", {"code": CSV + "t=df.groupby('region').revenue_usd.sum()\nprint(t.idxmax(), t.max())"}), lambda o: answer(f"Top region: {o[-1]}")],
    6: [
        lambda o: act("python_eval", {"code": CSV + "q=df.groupby('quarter').revenue_usd.sum()\nprint(q['Q1'], q['Q4'], round((q['Q4']-q['Q1'])/q['Q1']*100,1))"}),
        lambda o: answer(f"Q1, Q4, % change: {o[-1]}"),
    ],
    7: [
        lambda o: act("search_documents", {"query": "Aurora price volume discount"}),
        lambda o: act("python_eval", {"code": "print(round(137*249*(1-0.08),2))"}),
        lambda o: answer(f"Total: {o[-1]}"),
    ],
    8: [
        lambda o: act("search_documents", {"query": "Aurora battery capacity active sleep current duty cycle"}),
        lambda o: act("python_eval", {"code": "avg=0.02*12+0.98*0.05\nprint(round(4800/avg/24))"}),
        lambda o: answer(f"Battery life: {o[-1]} days"),
    ],
    9: [
        lambda o: act("search_documents", {"query": "incident severity levels SEV1 SEV2 SEV3"}),
        lambda o: act("generate_report", {"filename": "incident_summary.md", "content": "- SEV1: 15 min\n- SEV2: 1 hour\n- SEV3: next business day"}),
        lambda o: answer("Saved outputs/incident_summary.md"),
    ],
    10: [
        lambda o: act("search_documents", {"query": "password length key rotation"}),
        lambda o: act("generate_report", {"filename": "security_checklist.md", "content": "- [ ] Password >= 14 chars\n- [ ] Rotate API keys every 90 days"}),
        lambda o: answer("Saved outputs/security_checklist.md"),
    ],
    11: [
        lambda o: act("python_eval", {"code": CSV + "print(df.groupby('region').revenue_usd.sum().to_dict())"}),
        lambda o: act("generate_report", {"filename": "sales_analysis.md", "content": f"Revenue per region: {o[-1]}"}),
        lambda o: answer("Saved outputs/sales_analysis.md"),
    ],
    12: [
        lambda o: act("task_manager", {"action": "plan", "goal": "Launch v2.5", "steps": ["Freeze scope", "Run regression tests", "Build release candidate", "Security review", "Publish release notes", "Tag and ship"]}),
        lambda o: answer(f"Plan:\n1. a\n2. b\n3. c\n4. d\n5. e\n\n{o[-1]}"),
    ],
    13: [
        lambda o: act("search_documents", {"query": "SEV1 response procedure"}),
        lambda o: act("task_manager", {"action": "plan", "goal": "SEV1 response", "steps": ["Declare incident", "Assign roles", "Open bridge", "Roll back", "Update status page"]}),
        lambda o: act("task_manager", {"action": "update", "task_id": 1, "status": "in_progress"}),
        lambda o: answer(f"Done.\n{o[-1]}"),
    ],
    14: [
        lambda o: act("search_documents", {"query": "learning budget"}),
        lambda o: act("python_eval", {"code": "print(round(50000/12, 2))"}),
        lambda o: act("generate_report", {"filename": "benefits_note.md", "content": f"Monthly learning budget: INR {o[-1]}"}),
        lambda o: answer(f"INR {o[1]} per month; saved benefits_note.md"),
    ],
    15: [lambda o: answer("The capital of France is Paris.")],
}


class OracleLLM:
    """Plays the ideal script for whichever benchmark task is in the conversation."""

    def chat(self, messages: list[dict[str, str]]) -> str:
        prompts = {t["prompt"]: t["id"] for t in TASKS}
        task_id = next(prompts[m["content"]] for m in reversed(messages) if m["role"] == "user" and m["content"] in prompts)
        observations = _obs(messages)
        # step = number of tool observations seen so far (verify nudges / recoveries are not expected here)
        step = SCRIPTS[task_id][min(len(observations), len(SCRIPTS[task_id]) - 1)]
        return json.dumps(step(observations))


@pytest.fixture()
def eval_settings(settings):
    return settings.with_overrides(episodic_memory=False, max_iterations=8)


def test_oracle_passes_all_15_benchmark_tasks(eval_settings, memory):
    assert len(TASKS) == 15 and {t["id"] for t in TASKS} == set(range(1, 16))
    rows = run_eval.run_benchmark(TASKS, eval_settings, llm=OracleLLM(), memory=memory)
    failed = [(r["id"], r["failures"], r["used_tools"]) for r in rows if not r["strict_success"]]
    assert not failed, failed
    summary = run_eval.summarise(rows)
    assert summary["tool_selection_accuracy"] == 1.0 and summary["task_completion_rate"] == 1.0
    assert summary["p95_latency_s"] >= summary["median_latency_s"] >= 0


def test_scoring_detects_wrong_tools_and_wrong_answers(eval_settings, memory):
    class WrongLLM:
        def chat(self, messages):
            return json.dumps(answer("I think it is 7 days."))  # never uses a tool, wrong answer

    rows = run_eval.run_benchmark([TASKS[0], TASKS[14]], eval_settings, llm=WrongLLM(), memory=memory)
    assert rows[0]["tool_selection_correct"] is False and rows[0]["completed"] is False and not rows[0]["strict_success"]
    assert rows[1]["tool_selection_correct"] is True and rows[1]["completed"] is False  # no tools needed but answer wrong
    s = run_eval.summarise(rows)
    assert s["tool_selection_accuracy"] == 0.5 and s["task_completion_rate"] == 0.0


def test_helpers():
    assert run_eval.normalise("**36,790.86** and 1,234,567") == "36790.86 and 1234567"
    assert run_eval.percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)
    assert run_eval.percentile([], 95) == 0.0


def test_readme_table_update(tmp_path, monkeypatch):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{run_eval.README_START}\nold\n{run_eval.README_END}\noutro")
    monkeypatch.setattr(run_eval, "README", readme)
    summary = {"tool_selection_accuracy": 0.9333, "task_completion_rate": 0.8, "strict_success_rate": 0.8, "avg_latency_s": 12.3, "p95_latency_s": 30.1, "avg_iterations": 3.2}
    assert run_eval.update_readme({"ollama_chat/llama3.1": summary})
    text = readme.read_text()
    assert "| `ollama_chat/llama3.1` | 93% | 80% |" in text and "old" not in text and text.endswith("outro")


def test_llm_outage_never_counts_as_success(eval_settings, memory):
    from agent.llm import LLMError

    class Down:
        def chat(self, messages):
            raise LLMError("Cannot reach Ollama")

    (row,) = run_eval.run_benchmark([TASKS[14]], eval_settings, llm=Down(), memory=memory)  # the no-tool task
    assert row["status"] == "llm_error" and row["tool_selection_correct"] is False and not row["strict_success"]
