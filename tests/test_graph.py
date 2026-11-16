"""Graph behaviour tests using a scripted fake LLM (no network, no model)."""

from __future__ import annotations

import json

from agent.graph import AshtroAgent, Decision, DecisionError, parse_decision
from agent.llm import LLMError, extract_json
from tests.conftest import ScriptedLLM, act, answer

import pytest


def make_agent(settings, memory, script) -> tuple[AshtroAgent, ScriptedLLM]:
    llm = ScriptedLLM(script)
    return AshtroAgent(settings, llm=llm, memory=memory), llm


# ---------------------------------------------------------------- parsing
def test_extract_json_handles_fences_think_and_prose():
    assert extract_json('<think>{"x":1}</think>```json\n{"reason":"r"}\n```') == {"reason": "r"}
    assert extract_json('Sure! {"final_answer": "hi", "action": null} done', ("final_answer",))["final_answer"] == "hi"
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_parse_decision_is_tolerant():
    d = parse_decision(json.dumps({"reason": "r", "plan": "- a\n- b", "action": {"name": "python_eval", "arguments": '{"code": "1"}'}, "final_answer": "ignored"}))
    assert d.plan == ["a", "b"] and d.action.tool == "python_eval" and d.action.args == {"code": "1"}
    assert d.final_answer is None  # action wins over a premature answer
    assert parse_decision('{"action": "none", "final_answer": "ok"}').final_answer == "ok"
    with pytest.raises(DecisionError):
        parse_decision('{"reason": "x", "action": null, "final_answer": null}')
    with pytest.raises(DecisionError):
        parse_decision("garbage")
    assert isinstance(Decision(), Decision)


# ------------------------------------------------------------------ flows
def test_direct_answer_without_tools(settings, memory):
    agent, llm = make_agent(settings, memory, [answer("Paris")])
    res = agent.run("capital of France?", "s1")
    assert res.status == "ok" and res.answer == "Paris" and res.tool_calls == [] and res.iterations == 1
    assert len(llm.calls) == 1
    assert llm.calls[0][0]["role"] == "system" and "Reason -> Plan -> Act -> Verify" in llm.calls[0][0]["content"]


def test_tool_then_answer_streams_events_in_order(settings, memory):
    agent, _ = make_agent(
        settings,
        memory,
        [act("python_eval", {"code": "print(21*2)"}), answer("The answer is 42")],
    )
    events = list(agent.stream("what is 21*2", "s2"))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "memory" and kinds[-1] == "result"
    assert kinds.index("action") < kinds.index("observation") < kinds.index("final") < kinds.index("result")
    obs = next(e for e in events if e["type"] == "observation")
    assert obs["ok"] and obs["output"] == "42"
    res = events[-1]["result"]
    assert res.status == "ok" and [c["tool"] for c in res.tool_calls] == ["python_eval"] and res.iterations == 2


def test_tool_error_triggers_recovery_then_success(settings, memory):
    agent, llm = make_agent(
        settings,
        memory,
        [
            act("python_eval", {"code": "print(1/0)"}),
            act("python_eval", {"code": "print(10/2)"}),
            answer("5.0"),
        ],
    )
    events = list(agent.stream("divide", "s3"))
    kinds = [e["type"] for e in events]
    assert "recover" in kinds
    res = events[-1]["result"]
    assert res.status == "ok" and res.errors == 1 and res.answer == "5.0"
    # the corrective guidance reached the model
    assert "FAILED" in llm.calls[1][-1]["content"] and "ZeroDivisionError" in llm.calls[1][-1]["content"]


def test_unknown_tool_is_recoverable(settings, memory):
    agent, _ = make_agent(settings, memory, [act("web_browse", {}), answer("sorry, done")])
    res = agent.run("x", "s4")
    assert res.status == "ok" and res.errors == 1 and "Available tools" in res.observations[0]


def test_repeated_tool_errors_hit_error_limit(settings, memory):
    agent, _ = make_agent(settings, memory, [act("python_eval", {"code": "print(1/0)"}), answer("giving up")])
    # script repeats last item; first call errors forever until guard, whose wrap-up gets answer()
    agent, _ = make_agent(settings, memory, [act("python_eval", {"code": "print(1/0)"})] * 3 + [answer("I could not compute this")])
    res = agent.run("loop", "s5")
    assert res.status == "error_limit" and res.errors == 3
    assert "could not compute" in res.answer and "limit" in res.answer


def test_max_iterations_guard_stops_infinite_loop(settings, memory):
    s = settings.with_overrides(max_iterations=3)
    agent, llm = make_agent(s, memory, [act("task_manager", {"action": "list"})] * 3 + [answer("partial result")])
    res = agent.run("never ends", "s6")
    assert res.status == "max_iterations" and res.iterations == 3
    assert len(res.tool_calls) == 3 and "partial result" in res.answer
    assert len(llm.calls) == 4  # 3 reasoning steps + 1 wrap-up


def test_guard_fallback_when_wrapup_fails(settings, memory):
    s = settings.with_overrides(max_iterations=2)
    agent, _ = make_agent(s, memory, [act("task_manager", {"action": "list"}), act("task_manager", {"action": "list"}), LLMError("down")])
    res = agent.run("x", "s7")
    assert res.status == "max_iterations" and "ran out of" in res.answer


def test_malformed_json_recovers(settings, memory):
    agent, llm = make_agent(settings, memory, ["I think the answer is 4", answer("4")])
    events = list(agent.stream("2+2", "s8"))
    assert any(e["type"] == "error" for e in events) and any(e["type"] == "recover" for e in events)
    assert events[-1]["result"].answer == "4"
    assert "not a valid response" in llm.calls[1][-1]["content"]


def test_missing_verify_is_nudged_once(settings, memory):
    no_verify = {"reason": "r", "plan": [], "action": None, "verify": "", "final_answer": "hello"}
    agent, llm = make_agent(settings, memory, [no_verify, answer("hello", verify="rechecked")])
    res = agent.run("hi", "s9")
    assert res.status == "ok" and len(llm.calls) == 2
    # a model that never verifies is not looped forever
    agent, llm = make_agent(settings, memory, [no_verify])
    assert agent.run("hi", "s10").answer == "hello" and len(llm.calls) == 2


def test_llm_failure_is_reported_not_raised(settings, memory):
    agent, _ = make_agent(settings, memory, [LLMError("Cannot reach Ollama")])
    res = agent.run("hi", "s11")
    assert res.status == "llm_error" and "Cannot reach Ollama" in res.answer


def test_memory_short_and_long_term_are_used(settings, memory):
    agent, llm = make_agent(settings, memory, [answer("Remote is 3 days a week.")])
    agent.run("How many remote days does the handbook allow?", "chat")
    agent2, llm2 = make_agent(settings, memory, [answer("Same as before.")])
    events = list(agent2.stream("How many remote days does the handbook allow per week?", "chat"))
    mem_event = events[0]
    assert mem_event["type"] == "memory" and mem_event["history_turns"] == 1 and mem_event["episodes"]
    first_call = llm2.calls[0]
    assert "Long-term memory" in first_call[0]["content"]
    assert [m["role"] for m in first_call[1:]] == ["user", "assistant", "user"]


def test_report_file_is_tracked_in_result(settings, memory):
    agent, _ = make_agent(
        settings,
        memory,
        [act("generate_report", {"filename": "r.md", "content": "hello"}), answer("saved outputs/r.md")],
    )
    res = agent.run("write r.md", "s12")
    assert res.files and res.files[0].endswith("r.md")
    assert (settings.outputs_dir / "r.md").exists()
