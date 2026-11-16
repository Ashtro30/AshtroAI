"""Ashtro-AI - Streamlit front-end with a live "Agent Thought Process" trace.

Run with::

    streamlit run app.py

Every LangGraph step (memory recall, reasoning, plan, tool call, observation,
verification, error recovery) is streamed into the chat as it happens.
"""

from __future__ import annotations

import html
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import streamlit as st

from agent import AshtroAgent, Settings, load_settings
from agent.graph import RunResult
from agent.llm import build_llm
from agent.memory import MemoryManager
from agent.tools import load_tasks, update_task_status

st.set_page_config(page_title="Ashtro-AI", page_icon="🪐", layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- #
# Dark theme
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--bg:#0b0d14;--panel:#12151f;--panel2:#171b28;--border:#252a3b;--text:#e6e8f0;--muted:#8b92a8;
--violet:#8b5cf6;--cyan:#22d3ee;--green:#34d399;--amber:#fbbf24;--red:#f87171;--pink:#f472b6;--blue:#60a5fa;}
html,body,[class*="css"],.stApp{font-family:'Inter',system-ui,sans-serif;}
.stApp{background:radial-gradient(1100px 560px at 8% -8%,rgba(139,92,246,.16),transparent 60%),
radial-gradient(900px 500px at 100% 0%,rgba(34,211,238,.10),transparent 55%),var(--bg);}
.stApp,[data-testid="stMarkdownContainer"],[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li{color:var(--text);}
h1,h2,h3,h4,h5,h6{color:var(--text);}
header[data-testid="stHeader"]{background:transparent;}
/* Self-sufficient dark styling: stays dark even if the app is launched outside the project root
   (where .streamlit/config.toml would not be picked up). */
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h5,[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{color:var(--text)!important;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{color:var(--muted)!important;}
[data-baseweb="input"],[data-baseweb="base-input"],[data-baseweb="select"]>div,.stTextInput input,.stTextArea textarea{background:var(--panel2)!important;color:var(--text)!important;border-color:var(--border)!important;}
[data-testid="stBottom"],[data-testid="stBottom"]>div,[data-testid="stBottomBlockContainer"]{background:transparent!important;}
[data-testid="stCode"] pre,[data-testid="stCode"]{background:var(--panel2)!important;border-radius:10px;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;}[data-testid="stMetricValue"]{color:var(--text)!important;}
[data-testid="stFileUploaderDropzone"]{background:var(--panel2);border:1px dashed var(--border);}
[data-testid="stAlert"]{background:rgba(23,27,40,.8);}
.empty h3{color:var(--text);}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0e1018,#0b0d14);border-right:1px solid var(--border);}
.block-container{padding-top:1.4rem;max-width:1180px;}
.hero{padding:.2rem 0 1rem 0;}
.hero h1{font-size:2.5rem;font-weight:700;letter-spacing:-.02em;margin:0;
background:linear-gradient(90deg,#c4b5fd 0%,#8b5cf6 35%,#22d3ee 100%);-webkit-background-clip:text;background-clip:text;color:transparent;}
.hero p{color:var(--muted);margin:.15rem 0 .7rem 0;font-size:.98rem;}
.pill{display:inline-flex;align-items:center;gap:.4rem;font-size:.74rem;padding:.2rem .65rem;margin:0 .35rem .35rem 0;border-radius:999px;
border:1px solid var(--border);background:rgba(23,27,40,.7);color:var(--muted);font-family:'JetBrains Mono',monospace;}
.pill b{color:var(--text);font-weight:500;}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 8px var(--green);display:inline-block;}
.dot.warn{background:var(--amber);box-shadow:0 0 8px var(--amber);}
/* chat */
[data-testid="stChatMessage"]{background:rgba(18,21,31,.72);border:1px solid var(--border);border-radius:14px;padding:.9rem 1rem;margin-bottom:.7rem;
backdrop-filter:blur(6px);}
[data-testid="stChatInput"] textarea{font-family:'Inter',sans-serif;}
[data-testid="stChatInput"]>div{border:1px solid var(--border);border-radius:14px;background:var(--panel);}
[data-testid="stChatInput"]>div:focus-within{border-color:var(--violet);box-shadow:0 0 0 3px rgba(139,92,246,.18);}
[data-testid="stExpander"],[data-testid="stStatus"]{background:rgba(10,12,18,.65);border:1px solid var(--border)!important;border-radius:12px;}
.stButton>button,.stDownloadButton>button{background:var(--panel2);border:1px solid var(--border);color:var(--text);border-radius:10px;transition:all .18s ease;}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--violet);color:#fff;box-shadow:0 0 0 3px rgba(139,92,246,.16),0 6px 20px rgba(139,92,246,.18);transform:translateY(-1px);}
.stTabs [data-baseweb="tab-list"]{gap:.25rem;border-bottom:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{border-radius:10px 10px 0 0;color:var(--muted);padding:.5rem 1rem;}
.stTabs [aria-selected="true"]{color:#fff;background:rgba(139,92,246,.12);}
[data-testid="stMetric"]{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:.7rem .9rem;}
/* trace */
.trace-step{border-left:2px solid var(--border);padding:.3rem 0 .45rem .95rem;margin:.05rem 0 .05rem .35rem;animation:fadeIn .32s ease;}
.trace-step.reason{border-color:#6d49d9}.trace-step.plan{border-color:#3b82f6}.trace-step.act{border-color:#d9a406}
.trace-step.obs{border-color:#10b981}.trace-step.verify{border-color:#34d399}.trace-step.recover,.trace-step.error{border-color:#ef4444}
.trace-step.memory{border-color:#db2777}.trace-step.guard{border-color:#f59e0b}
.trace-head{display:flex;align-items:center;gap:.5rem;margin-bottom:.2rem;}
.badge{font-size:.66rem;font-weight:700;letter-spacing:.07em;padding:.12rem .55rem;border-radius:999px;border:1px solid;font-family:'JetBrains Mono',monospace;}
.badge.reason{color:#c4b5fd;border-color:#6d49d9;background:rgba(139,92,246,.12)}
.badge.plan{color:#93c5fd;border-color:#2563eb;background:rgba(59,130,246,.12)}
.badge.act{color:#fcd34d;border-color:#b45309;background:rgba(245,158,11,.12)}
.badge.obs{color:#6ee7b7;border-color:#059669;background:rgba(16,185,129,.12)}
.badge.verify{color:#86efac;border-color:#16a34a;background:rgba(34,197,94,.12)}
.badge.recover,.badge.error{color:#fca5a5;border-color:#dc2626;background:rgba(239,68,68,.12)}
.badge.memory{color:#f9a8d4;border-color:#be185d;background:rgba(236,72,153,.12)}
.badge.guard{color:#fcd34d;border-color:#b45309;background:rgba(245,158,11,.12)}
.meta{color:var(--muted);font-size:.72rem;font-family:'JetBrains Mono',monospace;}
.tool{font-family:'JetBrains Mono',monospace;font-size:.82rem;color:var(--cyan);}
.txt{color:#cfd3e1;font-size:.9rem;line-height:1.5;}
.plan-list{margin:.1rem 0 0 0;padding-left:1.15rem;}
.plan-list li{color:#b9c3dc;font-size:.86rem;line-height:1.45;}
.code{background:#07080d;border:1px solid var(--border);border-radius:8px;padding:.5rem .7rem;margin-top:.3rem;font-family:'JetBrains Mono',monospace;
font-size:.76rem;color:#c9d1e6;white-space:pre-wrap;word-break:break-word;max-height:240px;overflow:auto;}
.code.bad{border-color:#7f1d1d;color:#fecaca;}
.route{color:#555d75;font-size:.7rem;margin:0 0 0 1.3rem;font-family:'JetBrains Mono',monospace;}
.pulse{display:inline-flex;align-items:center;gap:.55rem;color:var(--muted);font-size:.84rem;margin:.3rem 0 0 .4rem;}
.pulse i{width:9px;height:9px;border-radius:50%;background:var(--cyan);box-shadow:0 0 12px var(--cyan);animation:pulse 1.1s ease-in-out infinite;}
.chips{margin:.5rem 0 .2rem 0;}
.chip{display:inline-block;font-size:.74rem;color:var(--muted);border:1px solid var(--border);background:var(--panel2);padding:.15rem .6rem;margin:0 .35rem .3rem 0;border-radius:8px;font-family:'JetBrains Mono',monospace;}
.chip b{color:var(--text);font-weight:500}.chip.ok b{color:var(--green)}.chip.bad b{color:var(--red)}
.empty{border:1px dashed var(--border);border-radius:16px;padding:1.4rem 1.6rem;background:rgba(18,21,31,.5);margin-bottom:1rem;}
.empty h3{margin:.1rem 0 .3rem 0;font-weight:600}.empty p{color:var(--muted);margin:0;font-size:.92rem}
.task{border:1px solid var(--border);background:var(--panel);border-radius:12px;padding:.6rem .8rem;margin-bottom:.45rem;font-size:.88rem;}
.task small{color:var(--muted)}
.col-title{font-size:.72rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin-bottom:.5rem;font-weight:600}
@keyframes fadeIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
@keyframes pulse{0%,100%{opacity:.35;transform:scale(.8)}50%{opacity:1;transform:scale(1.15)}}
::-webkit-scrollbar{width:8px;height:8px}::-webkit-scrollbar-thumb{background:#2a3042;border-radius:8px}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

EXAMPLES = [
    ("📚", "According to the company handbook, how many remote days per week are allowed, and what are the core hours?"),
    ("🧮", "Using quarterly_sales.csv, which region had the highest total revenue and by how much did Q4 beat Q1?"),
    ("📝", "Summarize the incident severity levels from the runbook and save the summary as incident_summary.md"),
    ("🔋", "Estimate the Aurora Sensor Hub battery life in days at the default duty cycle."),
    ("🗂️", "Make a task plan for responding to a SEV1 incident, then mark the first task as in progress."),
    ("💬", "What can you do?"),
]

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def esc(value: Any) -> str:
    """HTML-escape and make newlines survive Streamlit's Markdown pass."""
    return html.escape(str(value), quote=False).replace("\n", "&#10;")


def code_block(text: str, limit: int = 1500, bad: bool = False) -> str:
    text = text if len(text) <= limit else text[:limit] + " …"
    return f'<div class="code{" bad" if bad else ""}">{esc(text)}</div>'


def render_event(ev: dict[str, Any], t0: float) -> Optional[str]:
    """Convert one trace event to a single-line HTML snippet (None = not shown)."""
    kind, step = ev["type"], ev.get("step", 0)
    when = f'+{max(ev["ts"] - t0, 0):.1f}s'

    def head(label: str, cls: str, extra: str = "") -> str:
        step_html = f'<span class="meta">step {step}</span>' if step else ""
        return f'<div class="trace-head"><span class="badge {cls}">{label}</span>{step_html}{extra}<span class="meta">{when}</span></div>'

    def wrap(cls: str, inner: str) -> str:
        return f'<div class="trace-step {cls}">{inner}</div>'

    if kind == "memory":
        n_ep, n_turns = len(ev.get("episodes", [])), ev.get("history_turns", 0)
        body = f'<div class="txt">Recalled {n_ep} similar past episode(s) and {n_turns} earlier turn(s) from this session.</div>'
        for line in ev.get("episodes", []):
            body += f'<div class="meta">• {esc(line)}</div>'
        return wrap("memory", head("MEMORY", "memory") + body)
    if kind == "reason":
        lat = f'<span class="meta">llm {ev["latency"]:.1f}s</span>' if "latency" in ev else ""
        return wrap("reason", head("REASON", "reason", lat) + f'<div class="txt">{esc(ev.get("text") or "…")}</div>')
    if kind == "plan":
        items = "".join(f"<li>{esc(s)}</li>" for s in ev.get("steps", []))
        return wrap("plan", head("PLAN", "plan") + f'<ol class="plan-list">{items}</ol>')
    if kind == "action":
        import json

        args = json.dumps(ev.get("args", {}), indent=2, ensure_ascii=False)
        return wrap("act", head("ACT", "act", f'<span class="tool">⚙ {esc(ev["tool"])}</span>') + code_block(args, 1200))
    if kind == "observation":
        ok = ev.get("ok", True)
        meta = f'<span class="meta">{ev.get("duration_ms", 0)} ms</span>'
        label, cls = ("OBSERVE", "obs") if ok else ("TOOL ERROR", "error")
        return wrap(cls, head(label, cls, f'<span class="tool">{esc(ev["tool"])}</span>{meta}') + code_block(ev.get("output", ""), 1400, bad=not ok))
    if kind == "verify":
        return wrap("verify", head("VERIFY", "verify") + f'<div class="txt">{esc(ev.get("text", ""))}</div>')
    if kind == "recover":
        return wrap("recover", head("RECOVER", "recover") + f'<div class="txt">{esc(ev.get("message", ""))}</div>')
    if kind == "guard":
        return wrap("guard", head("GUARD", "guard") + f'<div class="txt">{esc(ev.get("message", ""))}</div>')
    if kind == "error":
        return wrap("error", head("ERROR", "error") + f'<div class="txt">{esc(ev.get("message", ""))}</div>')
    if kind == "route":
        return f'<div class="route">↳ route → <b>{esc(ev.get("route", ""))}</b></div>'
    return None  # "final" and "result" are shown as the chat answer


def trace_html(events: list[dict[str, Any]]) -> str:
    if not events:
        return ""
    t0 = events[0]["ts"]
    return "".join(filter(None, (render_event(e, t0) for e in events)))


def live_label(ev: dict[str, Any]) -> str:
    labels = {
        "memory": "🧠 Recalling memory…",
        "reason": "🧠 Reasoning…",
        "plan": "🗺️ Planning…",
        "action": f"⚙️ Calling {ev.get('tool', 'tool')}…",
        "observation": "🔎 Verifying result…",
        "verify": "✅ Verifying answer…",
        "recover": "🛠️ Recovering from an error…",
        "guard": "⏱️ Wrapping up…",
        "error": "⚠️ Handling an error…",
    }
    return labels.get(ev["type"], "🧠 Thinking…")


def pulse_html(ev: dict[str, Any]) -> str:
    if ev["type"] == "action":
        text = f'Running <span class="tool">{esc(ev.get("tool"))}</span>…'
    elif ev["type"] in ("guard", "verify"):
        text = "Composing the final answer…"
    else:
        text = "Thinking about the next step…"
    return f'<div class="pulse"><i></i><span>{text}</span></div>'


def chips_html(meta: dict[str, Any]) -> str:
    status = meta.get("status", "ok")
    ok = status == "ok"
    chips = [
        f'<span class="chip"><b>{meta.get("steps", 0)}</b> reasoning steps</span>',
        f'<span class="chip"><b>{meta.get("tools", 0)}</b> tool calls</span>',
        f'<span class="chip {"bad" if meta.get("errors") else ""}"><b>{meta.get("errors", 0)}</b> errors recovered</span>',
        f'<span class="chip"><b>{meta.get("latency", 0):.1f}s</b></span>',
        f'<span class="chip {"ok" if ok else "bad"}"><b>{esc(status)}</b></span>',
    ]
    return f'<div class="chips">{"".join(chips)}</div>'


def typewriter(slot: Any, text: str, max_seconds: float = 1.2) -> None:
    """Reveal ``text`` progressively for a lively feel (capped so long answers stay snappy)."""
    words = re.split(r"(\s+)", text)
    if len(text) > 2500 or len(words) < 6:
        slot.markdown(text)
        return
    delay = min(0.02, max_seconds / max(len(words), 1))
    shown = ""
    for w in words:
        shown += w
        slot.markdown(shown + " ▌")
        time.sleep(delay)
    slot.markdown(text)


def file_downloads(paths: list[str], key_prefix: str) -> None:
    for i, p in enumerate(paths):
        path = Path(p)
        if path.exists():
            st.download_button(f"⬇ {path.name}", path.read_bytes(), file_name=path.name, key=f"{key_prefix}-{i}-{path.name}")


@st.cache_resource(show_spinner="Loading embedding model and indexing the knowledge base…")
def get_memory(_settings: Settings, cache_key: str) -> MemoryManager:
    memory = MemoryManager(_settings)
    memory.ingest()
    return memory


# --------------------------------------------------------------------------- #
# State, settings, sidebar
# --------------------------------------------------------------------------- #
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("session_id", uuid.uuid4().hex[:8])
ss.setdefault("pending_prompt", None)

base = load_settings()

with st.sidebar:
    st.markdown("### 🪐 Ashtro-AI")
    st.caption("Autonomous · Contextual · Adaptive")
    st.markdown("##### Model")
    model = st.text_input("LiteLLM model", value=base.llm_model, key="s_model", help="e.g. ollama_chat/llama3.1, ollama_chat/qwen2.5, openai/gpt-4o-mini, anthropic/claude-sonnet-5-5")
    api_base = st.text_input("API base (blank for cloud)", value=base.llm_api_base or "", key="s_api_base")
    temperature = st.slider("Temperature", 0.0, 1.0, float(base.temperature), 0.05, key="s_temp")
    max_iter = st.slider("Max iterations", 2, 20, int(base.max_iterations), key="s_iter", help="Hard cap on reasoning steps (the guard node stops the loop here).")
    episodic = st.toggle("Episodic memory", value=base.episodic_memory, key="s_epi", help="Recall and store summaries of past tasks in the vector store.")

    settings = base.with_overrides(
        llm_model=model.strip() or base.llm_model,
        llm_api_base=api_base.strip() or None,
        temperature=temperature,
        max_iterations=max_iter,
        episodic_memory=episodic,
    )

    if st.button("🔌 Test connection", key="b_ping"):
        client = build_llm(settings)
        with st.spinner("Pinging model…"):
            ok, msg, secs = client.ping() if hasattr(client, "ping") else (True, "n/a", 0.0)
        (st.success if ok else st.error)(f"{'Connected' if ok else 'Failed'} in {secs:.1f}s — {msg}")

    st.divider()
    st.markdown("##### Session")
    st.code(ss.session_id, language=None)
    if st.button("✨ New session", key="b_new"):
        memory_for_clear = get_memory(base, f"{base.embedding_backend}|{base.embedding_model}|{base.data_dir}|{base.state_dir}")
        memory_for_clear.short_term.clear(ss.session_id)
        ss.messages, ss.session_id = [], uuid.uuid4().hex[:8]
        st.rerun()
    stats_slot = st.empty()

memory = get_memory(base, f"{base.embedding_backend}|{base.embedding_model}|{base.data_dir}|{base.state_dir}")


def render_stats() -> None:
    with stats_slot.container():
        c1, c2 = st.columns(2)
        c1.metric("Doc chunks", memory.knowledge_count())
        c2.metric("Episodes", memory.episode_count())


# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
backend = memory.backend_id
fallback = memory.embedder.fallback_reason is not None
st.markdown(
    '<div class="hero"><h1>ASHTRO</h1><p>An autonomous assistant that reasons, plans, acts and verifies — and shows its work.</p>'
    f'<span class="pill"><span class="dot"></span>model <b>{esc(settings.llm_model)}</b></span>'
    f'<span class="pill"><span class="dot{" warn" if fallback else ""}"></span>embeddings <b>{esc(backend)}</b></span>'
    f'<span class="pill">max steps <b>{settings.max_iterations}</b></span>'
    f'<span class="pill">session <b>{esc(ss.session_id)}</b></span></div>',
    unsafe_allow_html=True,
)
if fallback:
    st.warning("sentence-transformers could not be loaded, so Ashtro is using offline hashing embeddings (lexical retrieval). Install/download the model for semantic search.", icon="⚠️")

prompt: Optional[str] = st.chat_input("Ask Ashtro to research, calculate, plan or write a report…") or ss.pop("pending_prompt", None)

tab_chat, tab_kb, tab_out, tab_tasks = st.tabs(["💬 Chat", "📚 Knowledge & Memory", "📁 Outputs", "✅ Tasks"])


# --------------------------------------------------------------------------- #
# Chat tab
# --------------------------------------------------------------------------- #
def render_message(idx: int, msg: dict[str, Any]) -> None:
    with st.chat_message(msg["role"], avatar="🧑‍🚀" if msg["role"] == "user" else "🪐"):
        if msg["role"] == "assistant" and msg.get("events"):
            n = sum(1 for e in msg["events"] if e["type"] in ("reason", "observation", "recover"))
            with st.expander(f"🧠 Agent Thought Process · {n} events", expanded=False):
                st.markdown(trace_html(msg["events"]), unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg.get("meta"):
            st.markdown(chips_html(msg["meta"]), unsafe_allow_html=True)
            file_downloads(msg["meta"].get("files", []), f"hist-{idx}")


def handle_prompt(user_prompt: str) -> None:
    ss.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="🧑‍🚀"):
        st.markdown(user_prompt)

    agent = AshtroAgent(settings, memory=memory)
    events: list[dict[str, Any]] = []
    result: Optional[RunResult] = None

    with st.chat_message("assistant", avatar="🪐"):
        status = st.status("🧠 Agent Thought Process — starting…", expanded=True)
        t0 = time.time()
        with status:
            slot = st.empty()
            for ev in agent.stream(user_prompt, ss.session_id):
                slot.empty()
                if ev["type"] == "result":
                    result = ev["result"]
                    break
                events.append(ev)
                snippet = render_event(ev, t0)
                if snippet:
                    st.markdown(snippet, unsafe_allow_html=True)
                status.update(label=f"🧠 Agent Thought Process — {live_label(ev)}", state="running", expanded=True)  # expanded must be re-sent or the box collapses
                slot = st.empty()
                slot.markdown(pulse_html(ev), unsafe_allow_html=True)
            slot.empty()

        assert result is not None
        meta = {
            "steps": result.iterations,
            "tools": len(result.tool_calls),
            "errors": result.errors,
            "latency": result.latency_s,
            "status": result.status,
            "files": result.files,
        }
        failed = result.status in ("llm_error", "error")
        status.update(
            label=f"🧠 Agent Thought Process · {result.iterations} steps · {len(result.tool_calls)} tool calls · {result.latency_s:.1f}s",
            state="error" if failed else "complete",
            expanded=False,
        )
        answer_slot = st.empty()
        typewriter(answer_slot, result.answer)
        st.markdown(chips_html(meta), unsafe_allow_html=True)
        file_downloads(result.files, f"live-{len(ss.messages)}")

    ss.messages.append({"role": "assistant", "content": result.answer, "events": events, "meta": meta})


with tab_chat:
    if not ss.messages and not prompt:
        st.markdown(
            '<div class="empty"><h3>What should we work on?</h3><p>Ashtro searches your documents, runs sandboxed Python, writes reports and plans tasks — '
            "streaming every reasoning step so you can follow along. Pick a starter or type below.</p></div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, (icon, text) in enumerate(EXAMPLES):
            if cols[i % 2].button(f"{icon}  {text}", key=f"ex-{i}"):
                ss.pending_prompt = text
                st.rerun()
    for i, m in enumerate(ss.messages):
        render_message(i, m)
    if prompt:
        handle_prompt(prompt)


# --------------------------------------------------------------------------- #
# Knowledge & memory tab
# --------------------------------------------------------------------------- #
with tab_kb:
    sources = memory.knowledge_sources()
    c1, c2, c3 = st.columns(3)
    c1.metric("Indexed documents", len(sources))
    c2.metric("Chunks", memory.knowledge_count())
    c3.metric("Episodic memories", memory.episode_count())

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Add documents")
        uploads = st.file_uploader("Markdown or text files", type=["md", "txt"], accept_multiple_files=True, key="uploader")
        if uploads and st.button("➕ Add to knowledge base", key="b_add"):
            for up in uploads:
                safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(up.name).name)
                (settings.data_dir / safe).write_bytes(up.getvalue())
            report = memory.ingest()
            st.success(f"Indexed: {report.summary()}")
            st.rerun()
    with right:
        st.markdown("#### Index")
        if st.button("🔄 Re-index data folder", key="b_reindex"):
            report = memory.ingest(force=True)
            st.success(report.summary())
            st.rerun()
        st.caption(f"Embedding backend: `{backend}`")

    if sources:
        st.dataframe([{"source": s, "chunks": n} for s, n in sources.items()], hide_index=True)

    st.markdown("#### Try semantic retrieval")
    query = st.text_input("Search the knowledge base", placeholder="e.g. how fast must critical vulnerabilities be fixed?", key="kb_query")
    if query.strip():
        for hit in memory.search_knowledge(query.strip(), k=4):
            label = f"{hit.source} › {hit.section or 'document'} — relevance {hit.score:.2f}"
            with st.expander(label):
                st.progress(min(max(hit.score, 0.0), 1.0))
                st.markdown(hit.text.split("\n", 1)[-1])

    st.markdown("#### Episodic memory")
    episodes = memory.episodes(8)
    if not episodes:
        st.caption("Nothing stored yet — successful tasks are summarised here automatically.")
    for ep in episodes:
        with st.expander(f"{ep['ts'][:19].replace('T', ' ')} · session {ep['session']}"):
            st.text(ep["text"])
    if episodes and st.button("🗑️ Clear episodic memory", key="b_clear_ep"):
        memory.clear_episodes()
        st.rerun()


# --------------------------------------------------------------------------- #
# Outputs tab
# --------------------------------------------------------------------------- #
with tab_out:
    files = sorted(
        (p for p in settings.outputs_dir.glob("*") if p.is_file() and not p.name.startswith(".")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        st.info("No generated files yet. Ask Ashtro to “save the summary as report.md”.")
    else:
        names = [f.name for f in files]
        choice = st.selectbox("Generated files", names, key="out_choice")
        path = settings.outputs_dir / choice
        a, b = st.columns([1, 1])
        a.download_button("⬇ Download", path.read_bytes(), file_name=choice, key="out_dl")
        if b.button("🗑️ Delete", key="out_del"):
            path.unlink(missing_ok=True)
            st.rerun()
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".md":
            st.markdown(text)
        else:
            st.code(text, language="json" if path.suffix == ".json" else None)


# --------------------------------------------------------------------------- #
# Tasks tab
# --------------------------------------------------------------------------- #
with tab_tasks:
    board = load_tasks(settings, ss.session_id)
    tasks = board["tasks"]
    if not tasks:
        st.info("No tasks for this session yet. Try: “Plan the launch of firmware v2.5 as a task list.”")
    else:
        done = sum(1 for t in tasks if t["status"] == "done")
        st.markdown(f"**Goal:** {esc(board.get('goal') or '—')}")
        st.progress(done / len(tasks), text=f"{done}/{len(tasks)} done")
        columns = {"pending": "To do", "in_progress": "In progress", "blocked": "Blocked", "done": "Done"}
        cols = st.columns(len(columns))
        moves = {"pending": [("▶", "in_progress")], "in_progress": [("✓", "done"), ("⏸", "blocked")], "blocked": [("↺", "pending")], "done": [("↺", "pending")]}
        for col, (status_key, title) in zip(cols, columns.items()):
            with col:
                items = [t for t in tasks if t["status"] == status_key]
                st.markdown(f'<div class="col-title">{title} · {len(items)}</div>', unsafe_allow_html=True)
                for t in items:
                    st.markdown(
                        f'<div class="task">{esc(t["title"])}<br><small>#{t["id"]} · {esc(t["priority"])} priority</small></div>',
                        unsafe_allow_html=True,
                    )
                    btn_cols = st.columns(len(moves[status_key]))
                    for bc, (icon, new_status) in zip(btn_cols, moves[status_key]):
                        if bc.button(icon, key=f"mv-{t['id']}-{new_status}", help=f"Move to {new_status}"):
                            update_task_status(settings, ss.session_id, t["id"], new_status)
                            st.rerun()

render_stats()
