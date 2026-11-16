"""LangGraph orchestration for Ashtro.

Graph topology (cyclic ReAct with recovery and a hard iteration guard)::

    START -> recall -> reason -> router --+--> tools --+--> reason        (success: loop)
                         ^                |            +--> recover -----+ (tool error)
                         |                +--> recover -> reason          (bad format / missing verify)
                         |                +--> guard --> finalize         (max_iterations / error limit)
                         +----------------+--> finalize -> END            (final answer)

* ``recall``   - loads short-term history + similar episodic memories.
* ``reason``   - one LLM step producing a Reason/Plan/Act/Verify JSON decision.
* ``router``   - deterministic conditional routing (and the ``max_iterations`` guard).
* ``tools``    - executes the chosen tool through the safe registry.
* ``recover``  - turns tool failures / malformed output into corrective guidance.
* ``guard``    - forces a best-effort final answer when the budget is exhausted.
* ``finalize`` - persists short-term + episodic memory and emits the final event.

Every node appends UI-friendly events to ``trace``; :meth:`AshtroAgent.stream` yields
them live (``stream_mode="updates"``) so the Streamlit app can render each step.
"""

from __future__ import annotations

import json
import logging
import operator
import time
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Iterator, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import Settings, load_settings
from .llm import ChatModel, LLMError, build_llm, extract_json
from .memory import MemoryManager, format_episodes_for_prompt
from .prompts import (
    GUARD_WRAPUP,
    OBSERVATION_TEMPLATE,
    RECOVER_FORMAT_ERROR,
    RECOVER_TOOL_ERROR,
    VERIFY_NUDGE,
    build_system_prompt,
)
from .tools import ToolContext, ToolRegistry

logger = logging.getLogger("ashtro.graph")

_NULLISH = {"", "null", "none", "n/a", "nil", "undefined"}


# --------------------------------------------------------------------------- #
# Structured LLM decision (Reason -> Plan -> Act -> Verify)
# --------------------------------------------------------------------------- #
class DecisionError(ValueError):
    """The model reply could not be turned into a valid decision."""


class ToolCall(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    """One parsed reasoning step. Tolerant of the common ways small models drift."""

    model_config = ConfigDict(extra="ignore")

    reason: str = ""
    plan: list[str] = Field(default_factory=list)
    action: Optional[ToolCall] = None
    verify: str = ""
    final_answer: Optional[str] = None

    @field_validator("reason", "verify", mode="before")
    @classmethod
    def _to_text(cls, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (list, dict)):
            return json.dumps(v, ensure_ascii=False)
        return str(v).strip()

    @field_validator("plan", mode="before")
    @classmethod
    def _to_plan(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [ln.strip(" -*•\t") for ln in v.splitlines() if ln.strip()]
        if isinstance(v, list):
            return [str(x if not isinstance(x, dict) else json.dumps(x, ensure_ascii=False)).strip() for x in v if str(x).strip()]
        return [str(v)]

    @field_validator("action", mode="before")
    @classmethod
    def _to_action(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and v.strip().lower() in _NULLISH):
            return None
        if isinstance(v, str):
            return {"tool": v.strip(), "args": {}}
        if isinstance(v, dict):
            tool = v.get("tool") or v.get("name") or v.get("tool_name") or v.get("function")
            args = next((v[k] for k in ("args", "arguments", "parameters", "input") if k in v), {})
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {"_raw": args}
            if not tool or str(tool).strip().lower() in _NULLISH:
                return None
            return {"tool": str(tool).strip(), "args": args or {}}
        return v

    @field_validator("final_answer", mode="before")
    @classmethod
    def _to_answer(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, (dict, list)):
            return json.dumps(v, indent=2, ensure_ascii=False)
        text = str(v).strip()
        return None if text.lower() in _NULLISH else text

    @model_validator(mode="after")
    def _action_wins(self) -> "Decision":
        # If the model returns both, run the tool first - it has not seen the result yet.
        if self.action is not None and self.final_answer is not None:
            self.final_answer = None
        return self

    def to_json(self) -> str:
        return self.model_dump_json()


def parse_decision(raw: str) -> Decision:
    """Parse raw model text into a :class:`Decision` or raise :class:`DecisionError`."""
    try:
        obj = extract_json(raw, required_any=("reason", "plan", "action", "final_answer", "verify"))
    except ValueError as exc:
        raise DecisionError(str(exc)) from exc
    try:
        decision = Decision.model_validate(obj)
    except Exception as exc:  # noqa: BLE001 - pydantic ValidationError and friends
        raise DecisionError(f"schema mismatch: {str(exc).splitlines()[0]}") from exc
    if decision.action is None and decision.final_answer is None:
        raise DecisionError('both "action" and "final_answer" are null - exactly one must be set')
    return decision


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #
class AgentState(TypedDict, total=False):
    # inputs
    session_id: str
    task: str
    # accumulated (reducer = list concat)
    messages: Annotated[list[dict[str, str]], operator.add]
    trace: Annotated[list[dict[str, Any]], operator.add]
    tool_calls: Annotated[list[dict[str, Any]], operator.add]
    observations: Annotated[list[str], operator.add]
    files: Annotated[list[str], operator.add]
    # overwritten each step
    memory_context: str
    iteration: int
    decision: Optional[dict[str, Any]]
    parse_error: Optional[str]
    pending_error: Optional[str]
    recover_reason: Optional[str]  # "format" | "tool" | "verify"
    last_tool: Optional[str]
    consecutive_errors: int
    total_errors: int
    verify_nudged: bool
    route: str
    status: str  # ok | max_iterations | error_limit | llm_error
    final_answer: Optional[str]


_REDUCER_KEYS = {"messages", "trace", "tool_calls", "observations", "files"}


def _ev(kind: str, step: int, **data: Any) -> dict[str, Any]:
    """Create a trace event."""
    return {"type": kind, "step": step, "ts": time.time(), **data}


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"... [+{len(text) - limit} chars]"


def _with_user_suffix(messages: list[dict[str, str]], text: str) -> list[dict[str, str]]:
    """Append a user turn without creating two consecutive user messages."""
    msgs = [dict(m) for m in messages]
    if msgs and msgs[-1]["role"] == "user":
        msgs[-1]["content"] += "\n\n" + text
    else:
        msgs.append({"role": "user", "content": text})
    return msgs


@dataclass
class RunResult:
    """Final outcome of one agent run."""

    answer: str
    status: str
    iterations: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    errors: int = 0
    latency_s: float = 0.0
    session_id: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class AshtroAgent:
    """Autonomous agent built on a LangGraph ``StateGraph``."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm: Optional[ChatModel] = None,
        memory: Optional[MemoryManager] = None,
        registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.settings.ensure_dirs()
        self.llm: ChatModel = llm or build_llm(self.settings)
        self.memory = memory or MemoryManager(self.settings)
        self.registry = registry or ToolRegistry()
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ build
    def _build_graph(self) -> Any:
        g = StateGraph(AgentState)
        g.add_node("recall", self._recall_node)
        g.add_node("reason", self._reason_node)
        g.add_node("router", self._router_node)
        g.add_node("tools", self._tools_node)
        g.add_node("recover", self._recover_node)
        g.add_node("guard", self._guard_node)
        g.add_node("finalize", self._finalize_node)

        g.add_edge(START, "recall")
        g.add_edge("recall", "reason")
        g.add_edge("reason", "router")
        g.add_conditional_edges(
            "router",
            lambda s: s["route"],
            {"tools": "tools", "recover": "recover", "guard": "guard", "finalize": "finalize"},
        )
        g.add_conditional_edges(
            "tools",
            self._after_tools,
            {"reason": "reason", "recover": "recover", "guard": "guard"},
        )
        g.add_conditional_edges("recover", self._after_recover, {"reason": "reason", "guard": "guard"})
        g.add_edge("guard", "finalize")
        g.add_edge("finalize", END)
        return g.compile()

    # ------------------------------------------------------------------ nodes
    def _recall_node(self, state: AgentState) -> dict[str, Any]:
        """Load short-term history and similar past episodes."""
        session_id = state["session_id"]
        task = state["task"]
        history = self.memory.short_term.messages(session_id)
        context = ""
        episodes_shown: list[str] = []
        if self.settings.episodic_memory:
            try:
                hits = self.memory.recall(task, k=3, min_score=self.settings.memory_min_score)
                context = format_episodes_for_prompt(hits)
                episodes_shown = [" ".join(h.text.split())[:160] for h in hits]
            except Exception as exc:  # noqa: BLE001 - memory must never block the agent
                logger.warning("Episodic recall failed: %s", exc)
        event = _ev("memory", 0, episodes=episodes_shown, history_turns=len(history) // 2)
        return {
            "messages": history + [{"role": "user", "content": task}],
            "memory_context": context,
            "trace": [event],
        }

    def _reason_node(self, state: AgentState) -> dict[str, Any]:
        """One LLM step: parse a Reason/Plan/Act/Verify decision."""
        step = state.get("iteration", 0) + 1
        system = build_system_prompt(self.registry.describe(), state.get("memory_context", ""), self.settings.max_iterations)
        started = time.perf_counter()
        try:
            raw = self.llm.chat([{"role": "system", "content": system}] + state["messages"])
        except LLMError as exc:
            return {
                "iteration": step,
                "decision": None,
                "parse_error": None,
                "status": "llm_error",
                "final_answer": f"I couldn't reach the language model.\n\n**Details:** {exc}",
                "trace": [_ev("error", step, message=str(exc), fatal=True)],
            }
        latency = time.perf_counter() - started

        try:
            decision = parse_decision(raw)
        except DecisionError as exc:
            return {
                "iteration": step,
                "decision": None,
                "parse_error": str(exc),
                "messages": [{"role": "assistant", "content": raw.strip()[:2000] or "(empty response)"}],
                "trace": [_ev("error", step, message=f"Malformed model output: {exc}", raw=_clip(raw, 300), latency=latency)],
            }

        events = [_ev("reason", step, text=decision.reason, latency=latency)]
        if decision.plan:
            events.append(_ev("plan", step, steps=decision.plan))
        if decision.action:
            events.append(_ev("action", step, tool=decision.action.tool, args=decision.action.args))
        elif decision.verify:
            events.append(_ev("verify", step, text=decision.verify))
        return {
            "iteration": step,
            "decision": decision.model_dump(),
            "parse_error": None,
            "messages": [{"role": "assistant", "content": decision.to_json()}],
            "trace": events,
        }

    def _router_node(self, state: AgentState) -> dict[str, Any]:
        """Deterministic routing after each reasoning step."""
        step = state["iteration"]
        decision = state.get("decision")
        errors = state.get("consecutive_errors", 0)
        update: dict[str, Any] = {}

        if state.get("status") == "llm_error":
            route = "finalize"
        elif state.get("parse_error") or decision is None:
            errors += 1
            update.update(consecutive_errors=errors, total_errors=state.get("total_errors", 0) + 1, recover_reason="format")
            route = "guard" if errors >= self.settings.max_tool_retries or step >= self.settings.max_iterations else "recover"
            if route == "guard":
                update["status"] = "error_limit" if errors >= self.settings.max_tool_retries else "max_iterations"
        elif decision.get("action"):
            route = "tools"
        else:  # final answer
            if not decision.get("verify") and not state.get("verify_nudged") and step < self.settings.max_iterations:
                update.update(verify_nudged=True, recover_reason="verify")
                route = "recover"
            else:
                route = "finalize"
        update["route"] = route
        update["trace"] = [_ev("route", step, route=route)]
        return update

    def _tools_node(self, state: AgentState) -> dict[str, Any]:
        """Execute the chosen tool through the safe registry."""
        step = state["iteration"]
        action = state["decision"]["action"]  # type: ignore[index]
        name, args = action["tool"], action["args"]
        ctx = ToolContext(self.settings, self.memory, state["session_id"])
        result = self.registry.execute(name, args, ctx)

        call = {"tool": name, "args": args, "ok": result.ok, "duration_ms": result.meta.get("duration_ms", 0), "step": step}
        out: dict[str, Any] = {
            "tool_calls": [call],
            "observations": [f"[{name}] {result.output}"],
            "last_tool": name,
            "trace": [
                _ev("observation", step, tool=name, ok=result.ok, output=_clip(result.output, 2500), duration_ms=call["duration_ms"])
            ],
        }
        if result.meta.get("relpath"):
            out["files"] = [result.meta["path"]]
        if result.ok:
            out.update(
                consecutive_errors=0,
                recover_reason=None,
                pending_error=None,
                messages=[{"role": "user", "content": OBSERVATION_TEMPLATE.format(tool=name, content=_clip(result.output, 6000))}],
            )
        else:
            out.update(
                consecutive_errors=state.get("consecutive_errors", 0) + 1,
                total_errors=state.get("total_errors", 0) + 1,
                recover_reason="tool",
                pending_error=result.output,
            )
        return out

    def _after_tools(self, state: AgentState) -> str:
        if state.get("recover_reason") == "tool":
            if state.get("consecutive_errors", 0) >= self.settings.max_tool_retries:
                return "guard"
            return "guard" if state["iteration"] >= self.settings.max_iterations else "recover"
        return "guard" if state["iteration"] >= self.settings.max_iterations else "reason"

    def _recover_node(self, state: AgentState) -> dict[str, Any]:
        """Convert a failure into corrective guidance for the next reasoning step."""
        step = state["iteration"]
        reason = state.get("recover_reason") or "format"
        attempt = state.get("consecutive_errors", 1)
        limit = self.settings.max_tool_retries
        if reason == "tool":
            text = RECOVER_TOOL_ERROR.format(
                tool=state.get("last_tool", "tool"), attempt=attempt, limit=limit, error=_clip(state.get("pending_error") or "unknown error", 3000)
            )
            summary = f"Tool failed (attempt {attempt}/{limit}); asking the model to diagnose and retry differently."
        elif reason == "verify":
            text = VERIFY_NUDGE
            summary = "Final answer had no verification step; asking the model to verify before finishing."
        else:
            text = RECOVER_FORMAT_ERROR.format(error=state.get("parse_error") or "unknown format error")
            summary = f"Malformed reply (attempt {attempt}/{limit}); asking the model to resend valid JSON."
        return {
            "messages": [{"role": "user", "content": text}],
            "recover_reason": None,
            "trace": [_ev("recover", step, reason=reason, message=summary)],
        }

    def _after_recover(self, state: AgentState) -> str:
        return "guard" if state["iteration"] >= self.settings.max_iterations else "reason"

    def _guard_node(self, state: AgentState) -> dict[str, Any]:
        """Budget exhausted: force a best-effort answer from what has been gathered."""
        step = state["iteration"]
        errors = state.get("consecutive_errors", 0)
        status = state.get("status") if state.get("status") not in (None, "ok") else (
            "error_limit" if errors >= self.settings.max_tool_retries else "max_iterations"
        )
        why = "tool-call retries" if status == "error_limit" else f"reasoning steps (limit {self.settings.max_iterations})"
        system = build_system_prompt(self.registry.describe(), state.get("memory_context", ""), self.settings.max_iterations)
        answer: Optional[str] = None
        try:
            raw = self.llm.chat(
                [{"role": "system", "content": system}] + _with_user_suffix(state["messages"], GUARD_WRAPUP.format(reason=why))
            )
            answer = parse_decision(raw).final_answer
        except (LLMError, DecisionError) as exc:
            logger.warning("Guard wrap-up failed: %s", exc)
        if not answer:
            gathered = "\n".join(state.get("observations", [])[-3:]) or "(no tool results were obtained)"
            answer = (
                f"I stopped early because I ran out of {why}. Here is what I gathered so far:\n\n"
                f"```\n{_clip(gathered, 1500)}\n```\n\nPlease rephrase or narrow the request and I'll try again."
            )
        else:
            answer = f"⚠️ _I hit my {why} limit, so this answer may be incomplete._\n\n{answer}"
        return {
            "status": status,
            "final_answer": answer,
            "trace": [_ev("guard", step, status=status, message=f"Stopped: out of {why}. Producing best-effort answer.")],
        }

    def _finalize_node(self, state: AgentState) -> dict[str, Any]:
        """Persist memory and emit the final event."""
        step = state.get("iteration", 0)
        decision = state.get("decision") or {}
        answer = state.get("final_answer") or decision.get("final_answer") or "(no answer produced)"
        status = state.get("status") or "ok"
        if status != "llm_error":
            self.memory.short_term.add_turn(state["session_id"], state["task"], answer)
        if status == "ok" and self.settings.episodic_memory:
            try:
                tools_used = [c["tool"] for c in state.get("tool_calls", []) if c.get("ok")]
                self.memory.remember_episode(state["session_id"], state["task"], answer, tools_used)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not store episode: %s", exc)
        return {
            "final_answer": answer,
            "status": status,
            "trace": [_ev("final", step, text=answer, status=status)],
        }

    # -------------------------------------------------------------- public API
    def stream(self, task: str, session_id: Optional[str] = None) -> Iterator[dict[str, Any]]:
        """Run the graph, yielding trace events live; the last event has ``type == "result"``."""
        session_id = session_id or uuid.uuid4().hex[:8]
        started = time.perf_counter()
        initial: AgentState = {
            "session_id": session_id,
            "task": task.strip(),
            "messages": [],
            "trace": [],
            "tool_calls": [],
            "observations": [],
            "files": [],
            "iteration": 0,
            "consecutive_errors": 0,
            "total_errors": 0,
            "verify_nudged": False,
            "status": "ok",
        }
        acc: dict[str, Any] = dict(initial)
        limit = self.settings.max_iterations * 5 + 15  # reason+router+tools+recover per loop, with slack
        try:
            for chunk in self.graph.stream(initial, config={"recursion_limit": limit}, stream_mode="updates"):
                for _node, update in chunk.items():
                    if not update:
                        continue
                    for key, value in update.items():
                        acc[key] = (acc.get(key, []) + value) if key in _REDUCER_KEYS else value
                    for event in update.get("trace", []):
                        yield event
        except Exception as exc:  # noqa: BLE001 - surface graph-level failures as a result, not a crash
            logger.exception("Graph execution failed")
            acc["status"] = "error"
            acc["final_answer"] = f"Something went wrong while running the agent: {type(exc).__name__}: {exc}"
            yield _ev("error", acc.get("iteration", 0), message=acc["final_answer"], fatal=True)

        result = RunResult(
            answer=acc.get("final_answer") or "(no answer produced)",
            status=acc.get("status", "ok"),
            iterations=acc.get("iteration", 0),
            tool_calls=acc.get("tool_calls", []),
            observations=acc.get("observations", []),
            files=list(dict.fromkeys(acc.get("files", []))),
            errors=acc.get("total_errors", 0),
            latency_s=time.perf_counter() - started,
            session_id=session_id,
            trace=acc.get("trace", []),
        )
        yield {"type": "result", "step": result.iterations, "ts": time.time(), "result": result}

    def run(self, task: str, session_id: Optional[str] = None) -> RunResult:
        """Blocking convenience wrapper around :meth:`stream`."""
        result: Optional[RunResult] = None
        for event in self.stream(task, session_id):
            if event["type"] == "result":
                result = event["result"]
        assert result is not None
        return result
