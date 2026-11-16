"""Tests for chunking, ingestion, retrieval and episodic/short-term memory."""

from __future__ import annotations

from agent.memory import Embedder, ShortTermMemory, chunk_markdown


def test_chunking_keeps_headings_and_respects_size():
    text = "# Doc\n\n## A\n" + "para one. " * 40 + "\n\n" + "para two. " * 40 + "\n\n## B\nshort"
    chunks = chunk_markdown(text, "doc.md", max_chars=400)
    assert len(chunks) >= 3
    assert all(c.text.startswith("[doc.md]") for c in chunks)
    assert any("Doc > A" in c.text for c in chunks) and any("Doc > B" in c.text for c in chunks)
    assert all(len(c.text) < 400 * 1.7 + 80 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_hash_embedder_is_deterministic_and_normalised():
    e = Embedder("hash", "unused")
    a, b = e.embed(["remote work policy"])[0], e.embed(["remote work policy"])[0]
    assert a == b and abs(sum(x * x for x in a) - 1.0) < 1e-6
    assert e.backend_id.startswith("hash")


def test_ingest_is_idempotent_and_detects_changes(memory, settings):
    first = memory.ingest()
    assert len(first.added) == 6 and first.chunks_total > 6
    second = memory.ingest()
    assert not second.changed and len(second.unchanged) == 6

    (settings.data_dir / "company_handbook.md").write_text("# Handbook\n\n## Remote\nRemote is 5 days.")
    (settings.data_dir / "new_doc.md").write_text("# New\n\nHello world.")
    (settings.data_dir / "ml_best_practices.md").unlink()
    third = memory.ingest()
    assert third.updated == ["company_handbook.md"] and third.added == ["new_doc.md"] and third.removed == ["ml_best_practices.md"]
    assert "ml_best_practices.md" not in memory.knowledge_sources()


def test_semantic_search_ranks_relevant_source_first(memory):
    memory.ingest()
    top = memory.search_knowledge("SEV1 response time minutes first response", k=3)
    assert top and top[0].source == "incident_runbook.md"
    assert 0.0 <= top[0].score <= 1.0
    top = memory.search_knowledge("battery capacity mAh and LoRa range", k=2)
    assert top[0].source == "product_specs.md"


def test_search_auto_ingests_when_empty(memory):
    assert memory.knowledge_count() == 0
    assert memory.search_knowledge("password length", k=1)[0].source == "security_policy.md"


def test_episodic_memory_roundtrip(memory):
    assert memory.recall("anything") == []
    memory.remember_episode("s1", "Summarise the incident severity levels", "SEV1 15 min, SEV2 1 hour", ["search_documents"])
    memory.remember_episode("s1", "Compute compound interest on 15000", "about 36,000", ["python_eval"])
    hits = memory.recall("incident severity levels summary", k=2, min_score=0.1)
    assert hits and "severity" in hits[0].text.lower()
    assert memory.episode_count() == 2 and len(memory.episodes()) == 2
    memory.clear_episodes()
    assert memory.episode_count() == 0


def test_short_term_memory_window_and_json_protocol():
    stm = ShortTermMemory(max_turns=2)
    for i in range(3):
        stm.add_turn("s", f"q{i}", f"a{i}")
    msgs = stm.messages("s")
    assert [m["role"] for m in msgs] == ["user", "assistant"] * 2
    assert msgs[0]["content"] == "q1"
    assert '"final_answer": "a2"' in msgs[-1]["content"]
    stm.clear("s")
    assert stm.messages("s") == []
