"""Memory layer: short-term session state + long-term vector memory.

* :class:`ShortTermMemory` - rolling window of recent turns per session, replayed to
  the LLM so follow-up questions work.
* :class:`MemoryManager` - persistent ChromaDB store with two collections:

  - ``knowledge``: chunked local documents (``data/``) for RAG
  - ``episodes``: summaries of past successful interactions (episodic memory)

Embeddings come from ``sentence-transformers`` by default.  If the model cannot be
loaded (offline machine, missing weights) the manager transparently falls back
to a deterministic hashing embedder so the app keeps working; collections are
namespaced by embedding backend so vectors from different models never mix.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import zlib
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Iterable, Optional

from .config import Settings

logger = logging.getLogger("ashtro.memory")

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}
_HASH_DIM = 2048
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how i in is it its of on or that the this to was what when which who will with you your".split()
)


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    """A retrievable slice of a source document."""

    id: str
    text: str
    source: str
    section: str
    index: int


@dataclass
class MemoryHit:
    """A single retrieval result."""

    id: str
    text: str
    score: float  # cosine similarity in [0, 1] (higher is better)
    source: str = ""
    section: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestReport:
    """Summary of an indexing run."""

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    chunks_total: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)

    def summary(self) -> str:
        return (
            f"{len(self.added)} added, {len(self.updated)} updated, "
            f"{len(self.removed)} removed, {len(self.unchanged)} unchanged - {self.chunks_total} chunks indexed"
        )


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
class Embedder:
    """Lazy embedding backend with an offline-safe fallback."""

    def __init__(self, backend: str, model_name: str, allow_fallback: bool = True) -> None:
        self.requested_backend = backend
        self.model_name = model_name
        self.allow_fallback = allow_fallback
        self._model: Any = None
        self._resolved: Optional[str] = None
        self._lock = threading.Lock()
        self.fallback_reason: Optional[str] = None

    @property
    def backend_id(self) -> str:
        """Stable identifier of the *actually used* backend (used to name collections)."""
        self._resolve()
        assert self._resolved is not None
        return self._resolved

    def _resolve(self) -> None:
        with self._lock:
            if self._resolved is not None:
                return
            if self.requested_backend == "hash":
                self._resolved = f"hash{_HASH_DIM}"
                return
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                self._resolved = "st-" + re.sub(r"[^a-z0-9]+", "-", self.model_name.lower().split("/")[-1]).strip("-")
            except Exception as exc:  # noqa: BLE001 - any load failure triggers fallback
                if not self.allow_fallback:
                    raise RuntimeError(f"Could not load embedding model '{self.model_name}': {exc}") from exc
                self.fallback_reason = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "sentence-transformers unavailable (%s). Falling back to hashing embeddings - "
                    "retrieval will be lexical rather than semantic.",
                    self.fallback_reason,
                )
                self._resolved = f"hash{_HASH_DIM}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` into L2-normalised vectors."""
        self._resolve()
        if not texts:
            return []
        if self._model is not None:
            vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            return [v.tolist() for v in vectors]
        return [self._hash_embed(t) for t in texts]

    @staticmethod
    def _stem(token: str) -> str:
        """Very light plural stripping so 'passwords' matches 'password'."""
        if len(token) > 4 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
        return token

    @staticmethod
    def _hash_embed(text: str) -> list[float]:
        """Deterministic bag-of-words feature hashing (unigrams + bigrams, sublinear tf)."""
        tokens = [Embedder._stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]
        features: dict[int, float] = {}

        def add(token: str, weight: float) -> None:
            h = zlib.crc32(token.encode("utf-8"))
            idx = h % _HASH_DIM
            features[idx] = features.get(idx, 0.0) + weight  # unsigned: collisions only add small positive noise

        for tok in tokens:
            add(tok, 1.0)
        for a, b in zip(tokens, tokens[1:]):
            add(f"{a}_{b}", 0.6)
        vec = [0.0] * _HASH_DIM
        for idx, value in features.items():
            vec[idx] = math.log1p(value)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def chunk_markdown(text: str, source: str, max_chars: int = 900, overlap: int = 120) -> list[Chunk]:
    """Split Markdown/plain text into heading-aware chunks.

    Each chunk is prefixed with its heading breadcrumb (``Doc > Section``) so the
    embedding captures context even for short passages.
    """
    sections: list[tuple[str, list[str]]] = []
    crumbs: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((" > ".join(crumbs), [body]))
        buffer.clear()

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            crumbs[:] = crumbs[: level - 1] + [match.group(2)]
        else:
            buffer.append(line)
    flush()
    if not sections and text.strip():
        sections = [("", [text.strip()])]

    chunks: list[Chunk] = []
    for section, bodies in sections:
        paragraphs = [p.strip() for body in bodies for p in re.split(r"\n\s*\n", body) if p.strip()]
        current = ""
        pieces: list[str] = []
        for para in paragraphs:
            if current and len(current) + len(para) + 2 > max_chars:
                pieces.append(current)
                current = (current[-overlap:] + "\n\n" + para) if overlap else para
            else:
                current = f"{current}\n\n{para}" if current else para
            while len(current) > max_chars * 1.6:  # very long paragraph: hard split
                pieces.append(current[:max_chars])
                current = current[max_chars - overlap:]
        if current.strip():
            pieces.append(current)
        for piece in pieces:
            prefix = f"[{source}] {section}\n" if section else f"[{source}]\n"
            body = prefix + piece.strip()
            cid = hashlib.sha1(f"{source}|{len(chunks)}|{body}".encode("utf-8")).hexdigest()[:20]
            chunks.append(Chunk(id=cid, text=body, source=source, section=section, index=len(chunks)))
    return chunks


# --------------------------------------------------------------------------- #
# Short-term memory
# --------------------------------------------------------------------------- #
class ShortTermMemory:
    """Per-session rolling conversation window."""

    def __init__(self, max_turns: int = 6) -> None:
        self.max_turns = max_turns
        self._turns: dict[str, Deque[tuple[str, str]]] = {}
        self._lock = threading.Lock()

    def add_turn(self, session_id: str, user: str, assistant: str) -> None:
        with self._lock:
            dq = self._turns.setdefault(session_id, deque(maxlen=self.max_turns))
            dq.append((user, assistant))

    def messages(self, session_id: str) -> list[dict[str, str]]:
        """Prior turns as chat messages. Assistant turns are re-encoded in the JSON
        protocol so the model keeps following the output format."""
        with self._lock:
            turns = list(self._turns.get(session_id, ()))
        out: list[dict[str, str]] = []
        for user, assistant in turns:
            out.append({"role": "user", "content": user})
            payload = {
                "reason": "(earlier turn)",
                "plan": [],
                "action": None,
                "verify": "(earlier turn)",
                "final_answer": assistant[:1500],
            }
            out.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)})
        return out

    def turn_count(self, session_id: str) -> int:
        with self._lock:
            return len(self._turns.get(session_id, ()))

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._turns.pop(session_id, None)


# --------------------------------------------------------------------------- #
# Long-term vector memory
# --------------------------------------------------------------------------- #
def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")[:30] or "default"


class MemoryManager:
    """Owns the ChromaDB client, embedder, knowledge index and episodic memory."""

    def __init__(self, settings: Settings, embedder: Optional[Embedder] = None) -> None:
        self.settings = settings
        self.embedder = embedder or Embedder(settings.embedding_backend, settings.embedding_model)
        self.short_term = ShortTermMemory(settings.short_term_turns)
        self._client: Any = None
        self._kb: Any = None
        self._episodes: Any = None
        self._lock = threading.RLock()

    # ---------------------------------------------------------------- setup
    def _ensure(self) -> None:
        with self._lock:
            if self._client is not None:
                return
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            path = self.settings.state_dir / "chroma"
            path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))
            suffix = _slug(self.embedder.backend_id)
            self._kb = self._client.get_or_create_collection(f"ashtro_kb_{suffix}", metadata={"hnsw:space": "cosine"})
            self._episodes = self._client.get_or_create_collection(f"ashtro_ep_{suffix}", metadata={"hnsw:space": "cosine"})

    @property
    def backend_id(self) -> str:
        return self.embedder.backend_id

    # ------------------------------------------------------------ ingestion
    def _list_files(self) -> list[Path]:
        root = self.settings.data_dir
        if not root.exists():
            return []
        return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)

    def ingest(self, force: bool = False) -> IngestReport:
        """Index ``data_dir``. Idempotent: unchanged files are skipped, edited files are
        re-chunked and stale chunks removed, deleted files are dropped from the index."""
        self._ensure()
        report = IngestReport()
        with self._lock:
            existing = self._kb.get(include=["metadatas"])
            by_source: dict[str, dict[str, Any]] = {}
            for cid, meta in zip(existing.get("ids", []), existing.get("metadatas", []) or []):
                src = (meta or {}).get("source", "")
                entry = by_source.setdefault(src, {"ids": [], "hash": (meta or {}).get("file_hash")})
                entry["ids"].append(cid)

            on_disk: set[str] = set()
            for path in self._list_files():
                source = path.relative_to(self.settings.data_dir).as_posix()
                on_disk.add(source)
                try:
                    raw = path.read_bytes()
                    text = raw.decode("utf-8", errors="replace")
                except OSError as exc:
                    logger.warning("Skipping unreadable file %s: %s", path, exc)
                    continue
                file_hash = hashlib.sha1(raw).hexdigest()
                prior = by_source.get(source)
                if prior and prior["hash"] == file_hash and not force:
                    report.unchanged.append(source)
                    continue
                if prior:
                    self._kb.delete(ids=prior["ids"])
                    report.updated.append(source)
                else:
                    report.added.append(source)
                chunks = chunk_markdown(text, source)
                self._add_chunks(chunks, file_hash)

            for source, entry in by_source.items():
                if source not in on_disk:
                    self._kb.delete(ids=entry["ids"])
                    report.removed.append(source)
            report.chunks_total = self._kb.count()
        logger.info("Knowledge index: %s", report.summary())
        return report

    def _add_chunks(self, chunks: list[Chunk], file_hash: str, batch: int = 64) -> None:
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            self._kb.add(
                ids=[c.id for c in part],
                documents=[c.text for c in part],
                embeddings=self.embedder.embed([c.text for c in part]),
                metadatas=[
                    {"source": c.source, "section": c.section, "chunk": c.index, "file_hash": file_hash} for c in part
                ],
            )

    # ------------------------------------------------------------ knowledge
    def search_knowledge(self, query: str, k: int = 4) -> list[MemoryHit]:
        """Semantic search over the document index (auto-indexes on first use)."""
        self._ensure()
        with self._lock:
            if self._kb.count() == 0:
                self.ingest()
            count = self._kb.count()
            if count == 0:
                return []
            res = self._kb.query(
                query_embeddings=self.embedder.embed([query]),
                n_results=min(max(k, 1), count),
                include=["documents", "metadatas", "distances"],
            )
        return self._to_hits(res)

    def knowledge_sources(self) -> dict[str, int]:
        """Mapping ``source -> chunk count``."""
        self._ensure()
        with self._lock:
            metas = self._kb.get(include=["metadatas"]).get("metadatas", []) or []
        counts: dict[str, int] = {}
        for meta in metas:
            src = (meta or {}).get("source", "?")
            counts[src] = counts.get(src, 0) + 1
        return dict(sorted(counts.items()))

    def knowledge_count(self) -> int:
        self._ensure()
        return int(self._kb.count())

    # ------------------------------------------------------------- episodic
    def remember_episode(self, session_id: str, task: str, outcome: str, tools: Iterable[str]) -> None:
        """Persist a summary of a successful interaction for future recall."""
        self._ensure()
        tool_list = sorted(set(tools))
        text = f"Task: {task.strip()[:500]}\nOutcome: {outcome.strip()[:700]}\nTools: {', '.join(tool_list) or 'none'}"
        ts = datetime.now(timezone.utc)
        eid = hashlib.sha1(f"{session_id}|{ts.isoformat()}|{task}".encode("utf-8")).hexdigest()[:20]
        with self._lock:
            self._episodes.add(
                ids=[eid],
                documents=[text],
                embeddings=self.embedder.embed([text]),
                metadatas=[{"session_id": session_id, "ts": ts.isoformat(timespec="seconds"), "task": task[:200]}],
            )

    def recall(self, query: str, k: int = 3, min_score: Optional[float] = None) -> list[MemoryHit]:
        """Retrieve episodes similar to ``query`` above ``min_score``."""
        self._ensure()
        threshold = self.settings.memory_min_score if min_score is None else min_score
        with self._lock:
            count = self._episodes.count()
            if count == 0:
                return []
            res = self._episodes.query(
                query_embeddings=self.embedder.embed([query]),
                n_results=min(k, count),
                include=["documents", "metadatas", "distances"],
            )
        return [h for h in self._to_hits(res) if h.score >= threshold]

    def episodes(self, limit: int = 20) -> list[dict[str, Any]]:
        """Most recent episodes (for the UI)."""
        self._ensure()
        with self._lock:
            res = self._episodes.get(include=["documents", "metadatas"])
        rows = [
            {"text": doc, "ts": (meta or {}).get("ts", ""), "session": (meta or {}).get("session_id", "")}
            for doc, meta in zip(res.get("documents", []) or [], res.get("metadatas", []) or [])
        ]
        rows.sort(key=lambda r: r["ts"], reverse=True)
        return rows[:limit]

    def episode_count(self) -> int:
        self._ensure()
        return int(self._episodes.count())

    def clear_episodes(self) -> None:
        """Delete all episodic memories."""
        self._ensure()
        with self._lock:
            name = self._episodes.name
            self._client.delete_collection(name)
            self._episodes = self._client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _to_hits(res: dict[str, Any]) -> list[MemoryHit]:
        hits: list[MemoryHit] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            meta = meta or {}
            hits.append(
                MemoryHit(
                    id=cid,
                    text=doc,
                    score=max(0.0, min(1.0, 1.0 - float(dist))),
                    source=meta.get("source", ""),
                    section=meta.get("section", ""),
                    metadata=dict(meta),
                )
            )
        return hits


def format_episodes_for_prompt(hits: list[MemoryHit]) -> str:
    """Render recalled episodes as a compact bullet list for the system prompt."""
    lines = []
    for h in hits:
        when = str(h.metadata.get("ts", ""))[:10]
        body = " ".join(h.text.split())[:420]
        lines.append(f"- [{when}] {body}")
    return "\n".join(lines)
