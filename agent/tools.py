"""Ashtro's four custom tools and the registry that executes them safely.

Tools
-----
1. ``search_documents`` - semantic RAG search over the local knowledge base.
2. ``python_eval``       - sandboxed Python evaluator for maths / data analysis.
3. ``generate_report``   - write Markdown (or txt/csv/json) files into ``outputs/``.
4. ``task_manager``      - plan, track and update a per-session task list.

Every tool is executed through :meth:`ToolRegistry.execute`, which validates and
coerces arguments, catches *all* exceptions, and returns a :class:`ToolResult`.
Failures never raise into the graph: they come back as ``ERROR: ... Hint: ...``
observations so the LLM can self-correct.
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from .config import Settings

if TYPE_CHECKING:  # pragma: no cover
    from .memory import MemoryManager

logger = logging.getLogger("ashtro.tools")

MAX_OUTPUT_CHARS = 4000


# --------------------------------------------------------------------------- #
# Core types
# --------------------------------------------------------------------------- #
@dataclass
class ToolResult:
    """Outcome of a tool call. ``output`` is exactly what the LLM will read."""

    ok: bool
    output: str
    meta: dict[str, Any] = field(default_factory=dict)


class ToolError(Exception):
    """An expected, user-actionable failure. ``hint`` tells the agent how to fix it."""

    def __init__(self, message: str, hint: Optional[str] = None) -> None:
        super().__init__(message)
        self.hint = hint


@dataclass
class ToolContext:
    """Per-call execution context injected into tool functions."""

    settings: Settings
    memory: "MemoryManager"
    session_id: str


@dataclass(frozen=True)
class Param:
    """Declarative parameter spec used for prompt docs, validation and coercion."""

    type: str  # "string" | "integer" | "boolean" | "array"
    description: str
    required: bool = True
    enum: Optional[tuple[str, ...]] = None
    default: Any = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: dict[str, Param]
    func: Callable[..., ToolResult]
    example: str = ""


def _fail(message: str, hint: Optional[str] = None, **meta: Any) -> ToolResult:
    text = f"ERROR: {message}"
    if hint:
        text += f"\nHint: {hint}"
    return ToolResult(ok=False, output=text, meta=meta)


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} characters]"


# --------------------------------------------------------------------------- #
# Tool 1: local RAG document search
# --------------------------------------------------------------------------- #
def search_documents(ctx: ToolContext, query: str, top_k: int = 4) -> ToolResult:
    """Semantic search over the indexed knowledge base in ``data/``."""
    query = query.strip()
    if not query:
        raise ToolError("'query' is empty.", "Pass a short natural-language question or keywords.")
    top_k = max(1, min(int(top_k), 8))
    hits = ctx.memory.search_knowledge(query, k=top_k)
    if not hits:
        return ToolResult(
            True,
            "No documents are indexed yet. Add .md/.txt files to the data/ folder and re-index.",
            {"hits": 0},
        )
    blocks = []
    for i, h in enumerate(hits, 1):
        label = h.source + (f" > {h.section}" if h.section else "")
        body = h.text.split("\n", 1)[1] if "\n" in h.text else h.text  # drop the breadcrumb line
        blocks.append(f"[{i}] {label} (relevance {h.score:.2f})\n{body.strip()}")
    note = ""
    if hits[0].score < 0.2:
        note = "\nNote: low relevance - the documents may not cover this topic. Try different keywords or say so."
    return ToolResult(True, "\n\n".join(blocks) + note, {"hits": len(hits), "top_score": hits[0].score})


# --------------------------------------------------------------------------- #
# Tool 2: sandboxed Python evaluator
# --------------------------------------------------------------------------- #
# Defence in depth, NOT a hardened security boundary: AST screening in the parent,
# a stripped-down builtins table + module allowlist in a fresh `python -I` child,
# POSIX rlimits (CPU, memory, no file writes), a minimal environment (no API keys)
# and a wall-clock timeout.  For untrusted multi-tenant use, run Ashtro in a container.
_ALLOWED_MODULES = {
    "math", "statistics", "json", "re", "datetime", "collections", "itertools",
    "functools", "decimal", "fractions", "random", "string", "numpy", "pandas",
}  # fmt: skip
_DENY_NAMES = {
    "exec", "eval", "compile", "open", "input", "breakpoint", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "help", "exit", "quit", "memoryview", "__import__",
}  # fmt: skip
_DENY_ATTRS = {
    "system", "popen", "spawn", "fork", "exec", "eval", "query", "tofile", "fromfile", "memmap",
    "to_csv", "to_pickle", "to_excel", "to_parquet", "to_sql", "to_hdf", "to_feather", "to_json",
    "to_clipboard", "to_xml", "to_html", "to_latex", "to_stata", "to_orc", "to_gbq", "savefig",
    "os", "sys", "subprocess", "importlib", "builtins", "modules", "ctypes", "socket", "shutil",
    "load", "loadtxt", "genfromtxt", "fromregex", "dump", "open",
}  # fmt: skip
_DENY_ATTR_PREFIXES = ("read_", "save")  # pandas/numpy file readers and writers

_RUNNER_SRC = r'''
import sys, json, io, ast, types, importlib, contextlib, traceback, warnings, os
payload = json.loads(sys.stdin.read())
code, data_dir = payload["code"], payload["data_dir"]
allowed, deny_attrs, deny_prefixes = set(payload["allowed"]), set(payload["deny_attrs"]), tuple(payload["deny_prefixes"])
timeout = int(payload["timeout"])
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (timeout + 2, timeout + 2))
    for lim, val in ((getattr(resource, "RLIMIT_AS", None), 4 * 1024**3), (getattr(resource, "RLIMIT_FSIZE", None), 0)):
        if lim is not None:
            try: resource.setrlimit(lim, (val, val))
            except (ValueError, OSError): pass
except ImportError:
    pass  # Windows: rely on the parent's wall-clock timeout

warnings.simplefilter("ignore")
_cache = {}

def _namespace(mod, subs=()):
    """Expose only public, non-module, non-denied attributes of a module."""
    d = {}
    for k in dir(mod):
        if k.startswith("_") or k in deny_attrs or k.startswith(deny_prefixes):
            continue
        try: v = getattr(mod, k)
        except Exception: continue
        if isinstance(v, types.ModuleType):
            if k in subs: d[k] = _namespace(v)
            continue
        d[k] = v
    return types.SimpleNamespace(**d)

_SUBS = {"numpy": ("linalg", "random", "fft"), "pandas": ()}
def _load(name):
    if name not in _cache:
        _cache[name] = _namespace(importlib.import_module(name), _SUBS.get(name, ()))
    return _cache[name]

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name not in allowed:
        raise ImportError(f"import of '{name}' is not allowed. Allowed modules: {sorted(allowed)}")
    return _load(name)

def load_csv(filename):
    """Load a CSV from the data folder into a pandas DataFrame."""
    import pandas
    root = os.path.realpath(data_dir)
    path = os.path.realpath(os.path.join(root, filename))
    if not path.startswith(root + os.sep) or not path.lower().endswith(".csv"):
        raise ValueError("load_csv only reads .csv files inside the data folder")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no such CSV in data folder: {filename}. Available: {sorted(f for f in os.listdir(root) if f.endswith('.csv'))}")
    return pandas.read_csv(path)

import builtins as _b
_names = ("abs all any bin bool bytes callable chr complex dict divmod enumerate filter float format frozenset hash hex int "
          "isinstance issubclass iter len list map max min next object oct ord pow print property range repr reversed round "
          "set slice sorted staticmethod classmethod str sum super tuple type zip "
          "Exception ValueError TypeError KeyError IndexError ZeroDivisionError ArithmeticError StopIteration "
          "AssertionError RuntimeError NotImplementedError OverflowError AttributeError NameError ImportError "
          "True False None __build_class__").split()
safe_builtins = {n: getattr(_b, n) for n in _names if hasattr(_b, n)}
safe_builtins["__import__"] = _safe_import

env = {"__builtins__": safe_builtins, "__name__": "__ashtro__", "load_csv": load_csv}
for m in ("math", "statistics", "json", "re", "datetime", "collections", "itertools"):
    env[m] = _load(m)

out = io.StringIO()
result = {"ok": True, "stdout": "", "error": ""}
try:
    tree = ast.parse(code, "<ashtro>", "exec")
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    with contextlib.redirect_stdout(out):
        exec(compile(tree, "<ashtro>", "exec"), env)
        if last is not None:
            value = eval(compile(last, "<ashtro>", "eval"), env)
            if value is not None:
                print(repr(value) if not hasattr(value, "to_string") else value.to_string())
except BaseException as exc:
    tb = traceback.extract_tb(exc.__traceback__)
    line = next((f.lineno for f in reversed(tb) if f.filename == "<ashtro>"), None)
    result["ok"] = False
    result["error"] = f"{type(exc).__name__}: {exc}" + (f" (line {line})" if line else "")
result["stdout"] = out.getvalue()
sys.__stdout__.write(json.dumps(result))
'''  # noqa: E501


def _screen_python(code: str) -> None:
    """Reject obviously dangerous constructs before anything is executed."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ToolError(f"SyntaxError: {exc.msg} (line {exc.lineno})", "Fix the Python syntax and resend the full code.")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in _ALLOWED_MODULES or "." in alias.name:
                    raise ToolError(
                        f"import of '{alias.name}' is not allowed.",
                        f"Allowed modules: {', '.join(sorted(_ALLOWED_MODULES))}. Use e.g. np.linalg instead of dotted imports.",
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in _ALLOWED_MODULES or "." in (node.module or ""):
                raise ToolError(
                    f"import from '{node.module}' is not allowed.",
                    f"Allowed modules: {', '.join(sorted(_ALLOWED_MODULES))}.",
                )
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in _DENY_ATTRS or node.attr.startswith(_DENY_ATTR_PREFIXES):
                raise ToolError(
                    f"access to attribute '.{node.attr}' is blocked in the sandbox.",
                    "Use plain computations; file/system access is not available. Use load_csv('name.csv') for data files.",
                )
        elif isinstance(node, ast.Name):
            if node.id in _DENY_NAMES or node.id.startswith("__"):
                raise ToolError(f"use of '{node.id}' is blocked in the sandbox.", "Remove it and compute the result directly.")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise ToolError("string literals containing '__' are blocked in the sandbox.", "Avoid dunder names in strings.")


def python_eval(ctx: ToolContext, code: str) -> ToolResult:
    """Run Python code in a restricted subprocess and return what it printed."""
    if not code.strip():
        raise ToolError("'code' is empty.", "Send Python code that print()s the values you need.")
    _screen_python(code)

    env = {k: v for k, v in os.environ.items() if k in {"PATH", "SYSTEMROOT", "TEMP", "TMP", "LANG", "LC_ALL"}}
    env["PYTHONIOENCODING"] = "utf-8"
    # Single-threaded BLAS keeps virtual-memory use small enough for the RLIMIT_AS cap.
    env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    payload = json.dumps(
        {
            "code": code,
            "data_dir": str(ctx.settings.data_dir),
            "allowed": sorted(_ALLOWED_MODULES),
            "deny_attrs": sorted(_DENY_ATTRS),
            "deny_prefixes": list(_DENY_ATTR_PREFIXES),
            "timeout": ctx.settings.python_timeout,
        }
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", _RUNNER_SRC],
            input=payload,
            capture_output=True,
            text=True,
            timeout=ctx.settings.python_timeout,
            env=env,
            cwd=str(ctx.settings.state_dir),
        )
    except subprocess.TimeoutExpired:
        raise ToolError(
            f"execution exceeded the {ctx.settings.python_timeout}s time limit.",
            "Remove infinite loops or reduce the problem size.",
        )
    if not proc.stdout.strip():
        detail = (proc.stderr or "no output").strip().splitlines()[-1:] or ["no output"]
        raise ToolError(f"the Python runner crashed ({detail[0][:200]}).", "Simplify the code and try again.")
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise ToolError("the Python runner returned unreadable output.", "Avoid writing directly to sys.stdout; use print().")

    stdout = _truncate(result.get("stdout", ""))
    if not result.get("ok"):
        error = result.get("error", "unknown error")
        hint = "Fix the bug and resend the full code."
        if "NameError" in error and re.search(r"'(np|pd)'", error):
            hint = "Add `import numpy as np` / `import pandas as pd` at the top."
        elif "NameError" in error and "load_csv" not in code:
            hint = "Define the variable before use, or import the module you need."
        msg = error + (f"\nOutput before the error:\n{stdout}" if stdout.strip() else "")
        return _fail(msg, hint, stdout=stdout)
    if not stdout.strip():
        return ToolResult(True, "(code ran successfully but printed nothing - use print() to show results)", {"stdout": ""})
    return ToolResult(True, stdout.rstrip(), {"stdout": stdout})


# --------------------------------------------------------------------------- #
# Tool 3: Markdown report / file generator
# --------------------------------------------------------------------------- #
_ALLOWED_EXTS = {".md", ".txt", ".csv", ".json"}
_MAX_FILE_BYTES = 500_000


def _safe_filename(name: str) -> str:
    base = Path(name.replace("\\", "/")).name.strip()  # drop any directory components
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    if not base:
        raise ToolError("'filename' is empty or invalid.", "Use something like 'weekly_summary.md'.")
    suffix = Path(base).suffix.lower()
    if not suffix:
        base += ".md"
    elif suffix not in _ALLOWED_EXTS:
        raise ToolError(
            f"extension '{suffix}' is not allowed.", f"Allowed extensions: {', '.join(sorted(_ALLOWED_EXTS))}."
        )
    return base


def generate_report(
    ctx: ToolContext, filename: str, content: str, title: str = "", mode: str = "write"
) -> ToolResult:
    """Write ``content`` to ``outputs/<filename>`` (Markdown gets a title + timestamp header)."""
    if not content.strip():
        raise ToolError("'content' is empty.", "Provide the full body of the report.")
    if len(content.encode("utf-8")) > _MAX_FILE_BYTES:
        raise ToolError("content is too large (limit 500 KB).", "Summarise or split the content.")
    safe = _safe_filename(filename)
    outputs = ctx.settings.outputs_dir
    outputs.mkdir(parents=True, exist_ok=True)
    path = (outputs / safe).resolve()
    if outputs.resolve() not in path.parents:  # defence in depth against traversal
        raise ToolError("resolved path escapes the outputs folder.", "Use a plain file name.")

    body = content.strip() + "\n"
    if path.suffix == ".md" and mode == "write":
        header = ""
        if not body.lstrip().startswith("#"):
            header += f"# {title.strip() or path.stem.replace('_', ' ').title()}\n\n"
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        header += f"_Generated by Ashtro-AI · {stamp}_\n\n"
        body = header + body
    try:
        if mode == "append" and path.exists():
            with path.open("a", encoding="utf-8") as fh:
                fh.write("\n" + body)
        else:
            path.write_text(body, encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"could not write the file: {exc}", "Check disk space and permissions.")
    size = path.stat().st_size
    rel = f"outputs/{safe}"
    return ToolResult(True, f"Saved {rel} ({size} bytes, mode={mode}).", {"path": str(path), "relpath": rel, "bytes": size})


# --------------------------------------------------------------------------- #
# Tool 4: task planner / manager
# --------------------------------------------------------------------------- #
_TASK_LOCK = threading.Lock()
_STATUSES = ("pending", "in_progress", "done", "blocked")
_PRIORITIES = ("low", "medium", "high")
_STATUS_MARK = {"pending": "[ ]", "in_progress": "[~]", "done": "[x]", "blocked": "[!]"}


def _task_path(settings: Settings, session_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", session_id)[:64] or "default"
    folder = settings.state_dir / "tasks"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{safe}.json"


def load_tasks(settings: Settings, session_id: str) -> dict[str, Any]:
    """Load the task board for ``session_id`` (empty board if none)."""
    path = _task_path(settings, session_id)
    if not path.exists():
        return {"goal": "", "tasks": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Corrupt task file %s; starting fresh", path)
        return {"goal": "", "tasks": []}


def _save_tasks(settings: Settings, session_id: str, board: dict[str, Any]) -> None:
    path = _task_path(settings, session_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(board, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)  # atomic


def update_task_status(settings: Settings, session_id: str, task_id: int, status: str) -> bool:
    """UI helper: set a task's status. Returns False if the task does not exist."""
    with _TASK_LOCK:
        board = load_tasks(settings, session_id)
        for task in board["tasks"]:
            if task["id"] == task_id and status in _STATUSES:
                task["status"] = status
                task["updated"] = _now()
                _save_tasks(settings, session_id, board)
                return True
    return False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _render_board(board: dict[str, Any]) -> str:
    tasks = board["tasks"]
    if not tasks:
        return "The task list is empty."
    done = sum(1 for t in tasks if t["status"] == "done")
    lines = [f"Goal: {board.get('goal') or '(none)'}", f"Progress: {done}/{len(tasks)} done"]
    for t in tasks:
        note = f" - {t['notes']}" if t.get("notes") else ""
        lines.append(f"{_STATUS_MARK[t['status']]} {t['id']}. {t['title']} ({t['status']}, {t['priority']} priority){note}")
    return "\n".join(lines)


def task_manager(
    ctx: ToolContext,
    action: str,
    goal: str = "",
    steps: Optional[list[str]] = None,
    title: str = "",
    task_id: Optional[int] = None,
    status: str = "",
    notes: str = "",
    priority: str = "medium",
) -> ToolResult:
    """Create and manage a task plan for the current session."""
    with _TASK_LOCK:
        board = load_tasks(ctx.settings, ctx.session_id)
        tasks: list[dict[str, Any]] = board["tasks"]

        if action == "plan":
            if not steps:
                raise ToolError("action 'plan' needs 'steps' (a list of task titles).", "Provide goal and steps=[...].")
            board = {"goal": goal.strip(), "tasks": []}
            for i, step in enumerate(steps, 1):
                board["tasks"].append(
                    {"id": i, "title": step.strip(), "status": "pending", "priority": priority if priority in _PRIORITIES else "medium",
                     "notes": "", "created": _now(), "updated": _now()}
                )  # fmt: skip
            _save_tasks(ctx.settings, ctx.session_id, board)
            return ToolResult(True, f"Created {len(steps)} tasks.\n{_render_board(board)}", {"tasks": len(steps)})

        if action == "add":
            if not title.strip():
                raise ToolError("action 'add' needs 'title'.", "Provide a task title.")
            new_id = max((t["id"] for t in tasks), default=0) + 1
            tasks.append(
                {"id": new_id, "title": title.strip(), "status": "pending", "priority": priority if priority in _PRIORITIES else "medium",
                 "notes": notes, "created": _now(), "updated": _now()}
            )  # fmt: skip
            _save_tasks(ctx.settings, ctx.session_id, board)
            return ToolResult(True, f"Added task {new_id}.\n{_render_board(board)}")

        if action == "update":
            if task_id is None:
                raise ToolError("action 'update' needs 'task_id'.", "Call action 'list' to see the task ids.")
            target = next((t for t in tasks if t["id"] == task_id), None)
            if target is None:
                ids = ", ".join(str(t["id"]) for t in tasks) or "none"
                raise ToolError(f"task {task_id} does not exist (existing ids: {ids}).", "Call action 'list' or 'plan' first.")
            if status:
                if status not in _STATUSES:
                    raise ToolError(f"invalid status '{status}'.", f"Use one of: {', '.join(_STATUSES)}.")
                target["status"] = status
            if notes:
                target["notes"] = notes
            if title.strip():
                target["title"] = title.strip()
            target["updated"] = _now()
            _save_tasks(ctx.settings, ctx.session_id, board)
            return ToolResult(True, f"Task {task_id} is now {target['status']}.\n{_render_board(board)}")

        if action == "next":
            order = {"high": 0, "medium": 1, "low": 2}
            pending = sorted((t for t in tasks if t["status"] in ("pending", "in_progress")), key=lambda t: (t["status"] != "in_progress", order.get(t["priority"], 1), t["id"]))
            if not pending:
                return ToolResult(True, "No open tasks - everything is done (or the list is empty).")
            t = pending[0]
            return ToolResult(True, f"Next task: {t['id']}. {t['title']} ({t['status']}, {t['priority']} priority)")

        if action == "list":
            return ToolResult(True, _render_board(board))

        if action == "clear":
            _save_tasks(ctx.settings, ctx.session_id, {"goal": "", "tasks": []})
            return ToolResult(True, "Task list cleared.")

    raise ToolError(f"unknown action '{action}'.", "Valid actions: plan, add, update, next, list, clear.")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def _coerce(tool: str, name: str, param: Param, value: Any) -> Any:
    """Coerce an LLM-supplied value to the declared type or raise ToolError."""
    try:
        if param.type == "string":
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False)
            return str(value)
        if param.type == "integer":
            if isinstance(value, bool):
                raise ValueError
            return int(float(value)) if not isinstance(value, int) else value
        if param.type == "boolean":
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "yes"}
        if param.type == "array":
            if isinstance(value, str):
                stripped = value.strip()
                if stripped.startswith("["):
                    value = json.loads(stripped)
                else:  # newline / bullet separated text
                    value = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip() for ln in stripped.splitlines() if ln.strip()]
            if not isinstance(value, list):
                raise ValueError
            return [str(v) for v in value]
    except (ValueError, TypeError, json.JSONDecodeError):
        raise ToolError(
            f"argument '{name}' for {tool} must be of type {param.type}, got {type(value).__name__}: {str(value)[:60]!r}.",
            "Fix the argument type.",
        )
    return value


class ToolRegistry:
    """Holds tool specs, renders their docs for the prompt, and executes them safely."""

    def __init__(self, specs: Optional[list[ToolSpec]] = None) -> None:
        self._tools: dict[str, ToolSpec] = {s.name: s for s in (specs or default_tool_specs())}

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> str:
        """Human/LLM-readable tool documentation for the system prompt."""
        blocks = []
        for spec in self._tools.values():
            lines = [f"## {spec.name}", spec.description]
            for pname, p in spec.params.items():
                req = "required" if p.required else f"optional, default={p.default!r}"
                enum = f" one of {list(p.enum)}" if p.enum else ""
                lines.append(f"  - {pname} ({p.type}{enum}, {req}): {p.description}")
            if spec.example:
                lines.append(f"  Example: {spec.example}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def execute(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Validate, run and time a tool call. Never raises."""
        start = time.perf_counter()
        result = self._execute(name, args, ctx)
        result.meta["duration_ms"] = int((time.perf_counter() - start) * 1000)
        return result

    # ------------------------------------------------------------------
    def _execute(self, name: str, args: Any, ctx: ToolContext) -> ToolResult:
        spec = self._tools.get(name) or self._tools.get(str(name).strip().lower())
        if spec is None:
            return _fail(f"unknown tool '{name}'.", f"Available tools: {', '.join(self._tools)}.")
        if not isinstance(args, dict):
            return _fail(f"'args' for {spec.name} must be a JSON object.", f"Parameters: {', '.join(spec.params)}.")

        unknown = [k for k in args if k not in spec.params]
        if unknown:
            return _fail(
                f"unexpected argument(s) for {spec.name}: {', '.join(unknown)}.",
                f"Valid parameters: {', '.join(spec.params)}.",
            )
        try:
            clean: dict[str, Any] = {}
            for pname, p in spec.params.items():
                if pname in args and args[pname] is not None:
                    value = _coerce(spec.name, pname, p, args[pname])
                    if p.enum and isinstance(value, str) and value not in p.enum:
                        raise ToolError(f"'{pname}' must be one of {list(p.enum)}, got '{value}'.", "Pick a valid value.")
                    clean[pname] = value
                elif p.required:
                    raise ToolError(
                        f"missing required argument '{pname}' for {spec.name}.",
                        f"Parameters: {', '.join(spec.params)}. Example: {spec.example}" if spec.example else None,
                    )
            if "ctx" in inspect.signature(spec.func).parameters:
                clean["ctx"] = ctx
            return spec.func(**clean)
        except ToolError as exc:
            return _fail(str(exc), exc.hint)
        except Exception as exc:  # noqa: BLE001 - tools must never crash the agent
            logger.exception("Tool %s crashed", spec.name)
            return _fail(f"{type(exc).__name__}: {exc}", "Check the arguments and try again, or use a different approach.")


def default_tool_specs() -> list[ToolSpec]:
    """The four built-in tools."""
    return [
        ToolSpec(
            name="search_documents",
            description="Semantic search over the local knowledge base (company handbook, product specs, runbooks, policies, notes). Returns the most relevant passages with their source file.",
            params={
                "query": Param("string", "Natural-language question or keywords to search for."),
                "top_k": Param("integer", "How many passages to return (1-8).", required=False, default=4),
            },
            func=lambda ctx, query, top_k=4: search_documents(ctx, query, top_k),
            example='{"tool":"search_documents","args":{"query":"remote work policy"}}',
        ),
        ToolSpec(
            name="python_eval",
            description=(
                "Run Python in a sandbox for maths, statistics and data analysis. Only print() output is returned "
                "(a trailing bare expression is echoed). Allowed imports: math, statistics, json, re, datetime, collections, "
                "itertools, functools, decimal, fractions, random, numpy, pandas. math/statistics/json/re/datetime/collections/"
                "itertools are pre-imported. load_csv('file.csv') loads a CSV from the data folder as a pandas DataFrame. "
                "No file, network or system access."
            ),
            params={"code": Param("string", "Complete Python code. Must print() the results.")},
            func=lambda ctx, code: python_eval(ctx, code),
            example='{"tool":"python_eval","args":{"code":"df = load_csv(\'quarterly_sales.csv\')\\nprint(df.groupby(\'region\').revenue_usd.sum())"}}',
        ),
        ToolSpec(
            name="generate_report",
            description="Save a Markdown (or .txt/.csv/.json) file into the outputs folder. Use it to deliver reports, summaries and notes. Returns the saved path.",
            params={
                "filename": Param("string", "File name such as 'summary.md' (no directories; .md is added if missing)."),
                "content": Param("string", "Full file body (Markdown for .md)."),
                "title": Param("string", "Optional document title added as a heading if the content has none.", required=False, default=""),
                "mode": Param("string", "'write' (overwrite) or 'append'.", required=False, enum=("write", "append"), default="write"),
            },
            func=lambda ctx, filename, content, title="", mode="write": generate_report(ctx, filename, content, title, mode),
            example='{"tool":"generate_report","args":{"filename":"summary.md","title":"Summary","content":"## Findings\\n- ..."}}',
        ),
        ToolSpec(
            name="task_manager",
            description=(
                "Plan and track tasks for the current session. Actions: plan (create a task list from a goal + steps), "
                "add, update (change status/notes), next, list, clear."
            ),
            params={
                "action": Param("string", "What to do.", enum=("plan", "add", "update", "next", "list", "clear")),
                "goal": Param("string", "Overall goal (for 'plan').", required=False, default=""),
                "steps": Param("array", "Ordered task titles (for 'plan').", required=False),
                "title": Param("string", "Task title (for 'add').", required=False, default=""),
                "task_id": Param("integer", "Task id (for 'update').", required=False),
                "status": Param("string", "New status (for 'update').", required=False, enum=_STATUSES),
                "notes": Param("string", "Optional notes.", required=False, default=""),
                "priority": Param("string", "low | medium | high.", required=False, enum=_PRIORITIES, default="medium"),
            },
            func=lambda ctx, **kw: task_manager(ctx, **kw),
            example='{"tool":"task_manager","args":{"action":"plan","goal":"Ship v2","steps":["Write tests","Tag release"]}}',
        ),
    ]
