"""LLM transport layer.

Ashtro talks to models exclusively through `LiteLLM <https://docs.litellm.ai>`_,
which gives one interface for local Ollama models (``ollama_chat/llama3.1``,
``ollama_chat/qwen2.5``) and cloud APIs (``openai/...``, ``anthropic/...``, ...).

The agent does **not** rely on provider-specific native tool calling (which is
patchy on small local models).  Instead it uses a strict JSON protocol that any
instruction-following model can follow; :func:`extract_json` parses it
defensively.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Protocol

from .config import Settings

logger = logging.getLogger("ashtro.llm")

Message = dict[str, str]


class LLMError(RuntimeError):
    """Raised when the model backend cannot produce a response."""


class ChatModel(Protocol):
    """Minimal interface the graph needs; trivially fakeable in tests."""

    def chat(self, messages: list[Message]) -> str:  # pragma: no cover - protocol
        ...


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str, required_any: tuple[str, ...] = ()) -> dict[str, Any]:
    """Extract the first JSON object from ``text``.

    Handles ``<think>`` blocks emitted by reasoning models, Markdown code fences
    and prose before/after the object.  If ``required_any`` is given, only
    objects containing at least one of those keys are accepted (this stops a
    nested ``{"tool": ...}`` being mistaken for the whole decision when the outer
    object is malformed).

    Raises:
        ValueError: if no suitable JSON object is found.
    """
    cleaned = _THINK_RE.sub("", text or "").strip()
    candidates: list[str] = [m.group(1) for m in _FENCE_RE.finditer(cleaned)]
    candidates.append(cleaned)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        for match in re.finditer(r"\{", candidate):
            try:
                obj, _ = decoder.raw_decode(candidate[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and (not required_any or any(k in obj for k in required_any)):
                return obj
    raise ValueError("no valid JSON object found in model output")


class LiteLLMClient:
    """Thin, defensive wrapper around ``litellm.completion``."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # ------------------------------------------------------------------ chat
    def chat(self, messages: list[Message]) -> str:
        """Send ``messages`` and return the assistant text.

        Raises:
            LLMError: with a human-readable explanation (connection refused,
                missing API key, unknown model, ...).
        """
        try:
            return self._complete(messages, json_mode=self.settings.json_mode)
        except Exception as exc:  # noqa: BLE001 - we normalise everything below
            text = str(exc).lower()
            # Some providers reject response_format; retry once without it.
            if self.settings.json_mode and any(k in text for k in ("response_format", "json_object", "json mode")):
                logger.warning("Provider rejected JSON mode; retrying without it")
                try:
                    return self._complete(messages, json_mode=False)
                except Exception as exc2:  # noqa: BLE001
                    raise LLMError(self._explain(exc2)) from exc2
            raise LLMError(self._explain(exc)) from exc

    def _complete(self, messages: list[Message], json_mode: bool) -> str:
        import litellm  # imported lazily: it is slow to import

        litellm.drop_params = True  # silently drop params a provider does not support
        litellm.suppress_debug_info = True

        kwargs: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "timeout": self.settings.llm_timeout,
            "num_retries": self.settings.llm_retries,
        }
        if self.settings.llm_api_base:
            kwargs["api_base"] = self.settings.llm_api_base
        if self.settings.llm_api_key:
            kwargs["api_key"] = self.settings.llm_api_key
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = litellm.completion(**kwargs)
        content = response.choices[0].message.content  # type: ignore[union-attr]
        if not content or not str(content).strip():
            raise LLMError("the model returned an empty response")
        return str(content)

    # ------------------------------------------------------------------ ping
    def ping(self) -> tuple[bool, str, float]:
        """Cheap connectivity check used by the UI. Returns ``(ok, message, seconds)``."""
        start = time.perf_counter()
        try:
            reply = self._complete(
                [{"role": "user", "content": 'Reply with the JSON object {"status": "ok"}.'}],
                json_mode=self.settings.json_mode,
            )
            return True, reply.strip()[:120], time.perf_counter() - start
        except Exception as exc:  # noqa: BLE001
            return False, self._explain(exc), time.perf_counter() - start

    # --------------------------------------------------------------- helpers
    def _explain(self, exc: BaseException) -> str:
        """Turn provider exceptions into actionable guidance."""
        raw = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        lowered = str(exc).lower()
        model = self.settings.llm_model
        if self.settings.is_local_model and any(
            k in lowered for k in ("connection", "refused", "connect", "unreachable")
        ):
            return (
                f"Cannot reach Ollama at {self.settings.llm_api_base}. Start it with `ollama serve` "
                f"and pull the model with `ollama pull {model.split('/', 1)[-1]}`. ({raw})"
            )
        if self.settings.is_local_model and ("not found" in lowered or "404" in lowered):
            return f"Ollama does not have model '{model}'. Run `ollama pull {model.split('/', 1)[-1]}`. ({raw})"
        if any(k in lowered for k in ("api key", "api_key", "authentication", "unauthorized", "401")):
            return f"Authentication failed for '{model}'. Check the provider API key in your .env file. ({raw})"
        if "timeout" in lowered or "timed out" in lowered:
            return f"Request to '{model}' timed out after {self.settings.llm_timeout}s. ({raw})"
        return f"LLM call to '{model}' failed: {raw}"


def build_llm(settings: Settings) -> ChatModel:
    """Factory used by the agent (kept separate so tests can monkeypatch it)."""
    return LiteLLMClient(settings)
