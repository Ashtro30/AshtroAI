"""Automated evaluation for Ashtro-AI.

Runs the 15 prompts in ``benchmark_tasks.json`` against the configured model and
reports **tool-selection accuracy**, **task-completion rate** and **latency**.

Usage::

    python -m eval.run_eval                       # use .env settings
    python -m eval.run_eval --model ollama_chat/qwen2.5 --update-readme
    python -m eval.run_eval --ids 1,5,9 --verbose

Scoring (per task)
------------------
* **Tool selection correct**: ``required <= used <= required | optional`` - every
  expected tool was called and nothing unexpected was (hallucinated tool names count
  as unexpected). Tasks with no expected tools must be answered without any tool call.
* **Completed**: run finished with status ``ok`` AND every ``answer_regex`` matches the
  final answer AND every ``observation_regex`` matches the tool outputs AND (if set)
  the expected file exists and matches every ``file_regex`` AND ``min_calls`` are met.
* **Strict success**: tool selection correct AND completed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.config import Settings, load_settings  # noqa: E402
from agent.graph import AshtroAgent, RunResult  # noqa: E402
from agent.llm import ChatModel  # noqa: E402
from agent.memory import MemoryManager  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
README = ROOT / "README.md"
README_START, README_END = "<!-- BENCHMARK:START -->", "<!-- BENCHMARK:END -->"


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def normalise(text: str) -> str:
    """Strip thousands separators (``36,790.86`` -> ``36790.86``) and markdown bold."""
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    return text.replace("**", "").replace("`", "")


def _all_match(patterns: list[str], text: str) -> list[str]:
    """Return the patterns that do NOT match."""
    return [p for p in patterns if not re.search(p, text, re.IGNORECASE)]


def score_task(task: dict[str, Any], result: RunResult, outputs_dir: Path) -> dict[str, Any]:
    """Score one run against its task definition."""
    used_calls = [c["tool"] for c in result.tool_calls]
    used = set(used_calls)
    required = set(task.get("expected_tools", []))
    optional = set(task.get("optional_tools", []))
    tool_ok = required <= used <= (required | optional) and result.status != "llm_error"  # an outage proves nothing
    tp = len(used & required)
    precision = tp / len(used) if used else (1.0 if not required else 0.0)
    recall = tp / len(required) if required else 1.0

    failures: list[str] = []
    if result.status != "ok":
        failures.append(f"status={result.status}")
    answer = normalise(result.answer)
    for p in _all_match(task.get("answer_regex", []), answer):
        failures.append(f"answer missing /{p}/")
    obs = normalise("\n".join(result.observations))
    for p in _all_match(task.get("observation_regex", []), obs):
        failures.append(f"tool output missing /{p}/")
    for tool, n in task.get("min_calls", {}).items():
        if used_calls.count(tool) < n:
            failures.append(f"{tool} called {used_calls.count(tool)}x (<{n})")
    if task.get("expected_file"):
        f = outputs_dir / task["expected_file"]
        if not f.exists() or f.stat().st_size == 0:
            failures.append(f"file {task['expected_file']} not created")
        else:
            content = normalise(f.read_text(encoding="utf-8", errors="replace"))
            for p in _all_match(task.get("file_regex", []), content):
                failures.append(f"file missing /{p}/")

    completed = not failures
    return {
        "id": task["id"],
        "category": task.get("category", ""),
        "prompt": task["prompt"],
        "expected_tools": sorted(required),
        "used_tools": used_calls,
        "tool_selection_correct": tool_ok,
        "tool_precision": round(precision, 3),
        "tool_recall": round(recall, 3),
        "completed": completed,
        "strict_success": tool_ok and completed,
        "failures": failures,
        "status": result.status,
        "iterations": result.iterations,
        "tool_errors": result.errors,
        "latency_s": round(result.latency_s, 2),
        "answer_preview": result.answer[:300],
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-task rows into headline metrics."""
    n = len(rows) or 1
    lat = [r["latency_s"] for r in rows]
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    return {
        "tasks": len(rows),
        "tool_selection_accuracy": round(sum(r["tool_selection_correct"] for r in rows) / n, 4),
        "task_completion_rate": round(sum(r["completed"] for r in rows) / n, 4),
        "strict_success_rate": round(sum(r["strict_success"] for r in rows) / n, 4),
        "avg_latency_s": round(statistics.fmean(lat), 2) if lat else 0.0,
        "median_latency_s": round(statistics.median(lat), 2) if lat else 0.0,
        "p95_latency_s": round(percentile(lat, 95), 2),
        "avg_iterations": round(statistics.fmean(r["iterations"] for r in rows), 2) if rows else 0.0,
        "tool_errors_recovered": sum(r["tool_errors"] for r in rows),
        "by_category": {
            c: {"n": len(v), "tool_selection": round(sum(x["tool_selection_correct"] for x in v) / len(v), 3),
                "completion": round(sum(x["completed"] for x in v) / len(v), 3)}
            for c, v in sorted(by_cat.items())
        },
    }  # fmt: skip


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
def run_benchmark(
    tasks: list[dict[str, Any]],
    settings: Settings,
    llm: Optional[ChatModel] = None,
    memory: Optional[MemoryManager] = None,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Execute ``tasks`` sequentially, each in its own session, and score them."""
    memory = memory or MemoryManager(settings)
    memory.ingest()  # index once up front so embedding time is not counted as latency
    agent = AshtroAgent(settings, llm=llm, memory=memory)
    rows: list[dict[str, Any]] = []
    stamp = int(time.time())
    for pos, task in enumerate(tasks, 1):
        if task.get("expected_file"):  # make sure a stale file can't satisfy the check
            (settings.outputs_dir / task["expected_file"]).unlink(missing_ok=True)
        print(f"[{pos:>2}/{len(tasks)}] #{task['id']:<2} {task['category']:<18} {task['prompt'][:70]}...", flush=True)
        result = agent.run(task["prompt"], session_id=f"eval-{task['id']}-{stamp}")
        row = score_task(task, result, settings.outputs_dir)
        rows.append(row)
        flag = "PASS" if row["strict_success"] else "FAIL"
        print(
            f"      {flag}  tools={row['used_tools'] or '-'}  steps={row['iterations']}  "
            f"{row['latency_s']:.1f}s" + (f"  ({'; '.join(row['failures'])})" if row["failures"] else ""),
            flush=True,
        )
        if verbose:
            print("      answer:", row["answer_preview"].replace("\n", " ")[:200], flush=True)
    return rows


def render_table(rows: list[dict[str, Any]]) -> str:
    head = f"{'#':>2}  {'category':<18} {'tools OK':<9} {'done':<5} {'steps':>5} {'errs':>4} {'latency':>8}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['id']:>2}  {r['category']:<18} {'yes' if r['tool_selection_correct'] else 'NO':<9} "
            f"{'yes' if r['completed'] else 'NO':<5} {r['iterations']:>5} {r['tool_errors']:>4} {r['latency_s']:>7.1f}s"
        )
    return "\n".join(lines)


def render_summary(model: str, s: dict[str, Any]) -> str:
    return (
        f"Model: {model}\n"
        f"Tool-selection accuracy : {s['tool_selection_accuracy']:.1%}\n"
        f"Task-completion rate    : {s['task_completion_rate']:.1%}\n"
        f"Strict success          : {s['strict_success_rate']:.1%}\n"
        f"Latency avg/median/p95  : {s['avg_latency_s']}s / {s['median_latency_s']}s / {s['p95_latency_s']}s\n"
        f"Avg reasoning steps     : {s['avg_iterations']}   tool errors recovered: {s['tool_errors_recovered']}"
    )


# --------------------------------------------------------------------------- #
# Persistence / README table
# --------------------------------------------------------------------------- #
def save_results(model: str, rows: list[dict[str, Any]], summary: dict[str, Any], record_history: bool = True) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-")
    path = RESULTS_DIR / f"{ts}_{slug}.json"
    payload = {"model": model, "timestamp": ts, "summary": summary, "tasks": rows}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    (RESULTS_DIR / "latest.md").write_text(
        f"# Latest benchmark - {model} ({ts})\n\n```\n{render_summary(model, summary)}\n```\n\n```\n{render_table(rows)}\n```\n",
        encoding="utf-8",
    )
    if not record_history:
        return path
    history_path = RESULTS_DIR / "history.json"
    history: dict[str, Any] = {}
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = {}
    history[model] = {"timestamp": ts, **summary}
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return path


def update_readme(history: dict[str, Any]) -> bool:
    """Rewrite the benchmark table between the README markers from ``history.json``."""
    if not README.exists():
        return False
    text = README.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        return False
    rows = [
        "| Model | Tool selection | Completion | Strict success | Avg latency | p95 latency | Avg steps |",
        "| ----- | -------------- | ---------- | -------------- | ----------- | ----------- | --------- |",
    ]
    for model, s in sorted(history.items()):
        rows.append(
            f"| `{model}` | {s['tool_selection_accuracy']:.0%} | {s['task_completion_rate']:.0%} | {s['strict_success_rate']:.0%} "
            f"| {s['avg_latency_s']}s | {s['p95_latency_s']}s | {s['avg_iterations']} |"
        )
    block = f"{README_START}\n" + "\n".join(rows) + f"\n{README_END}"
    new = re.sub(re.escape(README_START) + r".*?" + re.escape(README_END), lambda _m: block, text, flags=re.DOTALL)
    README.write_text(new, encoding="utf-8")
    return True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Ashtro-AI on the benchmark tasks.")
    parser.add_argument("--tasks", type=Path, default=EVAL_DIR / "benchmark_tasks.json")
    parser.add_argument("--ids", help="comma-separated task ids to run (default: all)")
    parser.add_argument("--model", help="override LLM_MODEL (LiteLLM name, e.g. ollama_chat/qwen2.5)")
    parser.add_argument("--api-base", help="override LLM_API_BASE")
    parser.add_argument("--max-iterations", type=int, help="override AGENT_MAX_ITERATIONS")
    parser.add_argument("--with-memory", action="store_true", help="enable episodic memory (off by default for fairness)")
    parser.add_argument("--update-readme", action="store_true", help="refresh the README benchmark table")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))["tasks"]
    if args.ids:
        wanted = {int(i) for i in args.ids.split(",")}
        tasks = [t for t in tasks if t["id"] in wanted]
    if not tasks:
        print("No tasks selected.", file=sys.stderr)
        return 2

    base = load_settings()
    # Isolated sandbox so benchmark runs never touch the user's outputs, tasks or memory.
    state = ROOT / ".ashtro_eval"
    outputs = state / "outputs"
    shutil.rmtree(outputs, ignore_errors=True)
    api_base = args.api_base
    if api_base is None:
        if not args.model:
            api_base = base.llm_api_base
        elif args.model.startswith("ollama"):
            api_base = base.llm_api_base or "http://localhost:11434"
    settings = base.with_overrides(
        llm_model=args.model,
        llm_api_base=api_base,
        max_iterations=args.max_iterations,
        episodic_memory=args.with_memory,
        state_dir=state,
        outputs_dir=outputs,
    )
    settings.ensure_dirs()

    print(f"Evaluating {len(tasks)} tasks with {settings.llm_model} (max_iterations={settings.max_iterations})\n")
    rows = run_benchmark(tasks, settings, verbose=args.verbose)
    summary = summarise(rows)
    print("\n" + render_table(rows) + "\n\n" + render_summary(settings.llm_model, summary))
    outage = [r["id"] for r in rows if r["status"] == "llm_error"]
    path = save_results(settings.llm_model, rows, summary, record_history=not outage)
    print(f"\nSaved {path.relative_to(ROOT)} and eval/results/latest.md")
    if outage:
        print(f"WARNING: the model was unreachable/failed on task(s) {outage}. Results are NOT recorded in history or the README table.")
        return 1
    if args.update_readme:
        history = json.loads((RESULTS_DIR / "history.json").read_text(encoding="utf-8"))
        print("README table updated." if update_readme(history) else "README markers not found; table not updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
