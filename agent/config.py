"""Central configuration for Ashtro-AI.

All runtime behaviour is driven by environment variables (optionally loaded from
a ``.env`` file) so the same code runs against a local Ollama model or a cloud
provider without modification.  ``Settings`` is an immutable dataclass; use
``Settings.with_overrides`` to derive per-run variants (the Streamlit sidebar and
the evaluation harness both do this).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

_OLLAMA_DEFAULT_BASE = "http://localhost:11434"


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return a stripped env var, treating empty strings as unset."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    raw = _env_str(name)
    if raw is None:
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings."""

    # --- LLM (via LiteLLM) -------------------------------------------------
    llm_model: str = "ollama_chat/llama3.1"
    llm_api_base: Optional[str] = _OLLAMA_DEFAULT_BASE
    llm_api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2048
    llm_timeout: int = 120
    llm_retries: int = 2
    json_mode: bool = True

    # --- Agent loop ----------------------------------------------------------
    max_iterations: int = 8
    max_tool_retries: int = 3

    # --- Memory / RAG --------------------------------------------------------
    embedding_backend: str = "sentence-transformers"  # or "hash" (offline fallback)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 4
    memory_min_score: float = 0.35
    episodic_memory: bool = True
    short_term_turns: int = 6

    # --- Tools ---------------------------------------------------------------
    python_timeout: int = 10

    # --- Paths ---------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    state_dir: Path = PROJECT_ROOT / ".ashtro"

    def with_overrides(self, **overrides: Any) -> "Settings":
        """Return a copy with the given fields replaced (``None`` values are kept)."""
        clean = {k: v for k, v in overrides.items() if v is not None or k == "llm_api_base"}
        return replace(self, **clean)

    def ensure_dirs(self) -> None:
        """Create all working directories if they do not yet exist."""
        for directory in (self.data_dir, self.outputs_dir, self.state_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def is_local_model(self) -> bool:
        return self.llm_model.startswith("ollama")


def load_settings(env_file: Optional[Path] = None) -> Settings:
    """Build ``Settings`` from the environment (and ``.env`` if present)."""
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)

    model = _env_str("LLM_MODEL", "ollama_chat/llama3.1") or "ollama_chat/llama3.1"
    default_base = _OLLAMA_DEFAULT_BASE if model.startswith("ollama") else None

    settings = Settings(
        llm_model=model,
        llm_api_base=_env_str("LLM_API_BASE", default_base),
        llm_api_key=_env_str("LLM_API_KEY"),
        temperature=_env_float("LLM_TEMPERATURE", 0.1),
        max_tokens=_env_int("LLM_MAX_TOKENS", 2048),
        llm_timeout=_env_int("LLM_TIMEOUT", 120),
        llm_retries=_env_int("LLM_RETRIES", 2),
        json_mode=_env_bool("LLM_JSON_MODE", True),
        max_iterations=_env_int("AGENT_MAX_ITERATIONS", 8),
        max_tool_retries=_env_int("AGENT_MAX_TOOL_RETRIES", 3),
        embedding_backend=(_env_str("EMBEDDING_BACKEND", "sentence-transformers") or "").lower(),
        embedding_model=_env_str("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        or "sentence-transformers/all-MiniLM-L6-v2",
        top_k=_env_int("RAG_TOP_K", 4),
        memory_min_score=_env_float("MEMORY_MIN_SCORE", 0.35),
        episodic_memory=_env_bool("ENABLE_EPISODIC_MEMORY", True),
        short_term_turns=_env_int("SHORT_TERM_TURNS", 6),
        python_timeout=_env_int("PYTHON_TOOL_TIMEOUT", 10),
        data_dir=_env_path("ASHTRO_DATA_DIR", PROJECT_ROOT / "data"),
        outputs_dir=_env_path("ASHTRO_OUTPUTS_DIR", PROJECT_ROOT / "outputs"),
        state_dir=_env_path("ASHTRO_STATE_DIR", PROJECT_ROOT / ".ashtro"),
    )
    settings.ensure_dirs()
    return settings
