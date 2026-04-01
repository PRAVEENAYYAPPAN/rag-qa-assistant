"""
LLM service – unified abstraction over OpenAI, Groq, and local providers.

The provider is selected via the LLM_PROVIDER env var:
  - "openai"  → OpenAI ChatCompletion (gpt-4o, gpt-3.5-turbo, …)
  - "groq"    → Groq Cloud (llama3-8b-8192, mixtral-8x7b, …)
  - "local"   → Ollama-compatible local server on localhost:11434
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

log = get_logger(__name__)
settings = get_settings()

# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Q&A assistant with access to a curated knowledge base.
Your task is to answer the user's question accurately and concisely using ONLY the provided context.

Guidelines:
- Base your answer strictly on the provided context passages.
- If the context does not contain enough information, say "I don't have enough information in my knowledge base to answer this."
- Be precise, factual, and avoid hallucinating.
- Cite the source when you use a specific passage (e.g. "According to [source]…").
- Structure long answers with bullet points or numbered lists where appropriate.
"""

RAG_PROMPT_TEMPLATE = """{system}

## Retrieved Context

{context}

---

## Question
{question}

## Answer
"""


def _build_context(passages: list[dict[str, Any]]) -> str:
    """Format retrieved passages into a numbered context block."""
    parts = []
    for i, p in enumerate(passages, 1):
        source = p.get("source", "unknown")
        content = p.get("content", "")
        score = p.get("rerank_score", p.get("score", 0.0))
        parts.append(f"[{i}] (source: {source}, relevance: {score:.3f})\n{content}")
    return "\n\n".join(parts)


# ── Provider implementations ──────────────────────────────────────────────────

class _OpenAIProvider:
    def __init__(self) -> None:
        import openai
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.LLM_MODEL

    def generate(self, prompt: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        return resp.choices[0].message.content or "", elapsed


class _GroqProvider:
    def __init__(self) -> None:
        from groq import Groq
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.LLM_MODEL

    def generate(self, prompt: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        return resp.choices[0].message.content or "", elapsed


class _LocalProvider:
    """Ollama-compatible local REST API."""

    def __init__(self) -> None:
        import httpx
        self.client = httpx.Client(base_url="http://localhost:11434", timeout=120.0)
        self.model = settings.LLM_MODEL

    def generate(self, prompt: str) -> tuple[str, float]:
        import httpx
        t0 = time.perf_counter()
        resp = self.client.post(
            "/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": settings.LLM_TEMPERATURE,
                    "num_predict": settings.LLM_MAX_TOKENS,
                },
            },
        )
        elapsed = (time.perf_counter() - t0) * 1000
        data = resp.json()
        return data.get("response", ""), elapsed


@lru_cache(maxsize=1)
def _get_provider():
    p = settings.LLM_PROVIDER
    log.info("Initialising LLM provider: {} / {}", p, settings.LLM_MODEL)
    if p == "openai":
        return _OpenAIProvider()
    elif p == "groq":
        return _GroqProvider()
    elif p == "local":
        return _LocalProvider()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {p}")


# ── Public service ────────────────────────────────────────────────────────────

class LLMService:
    """High-level RAG answer-generation service."""

    def __init__(self) -> None:
        self._provider = _get_provider()

    def answer(
        self,
        question: str,
        passages: list[dict[str, Any]],
    ) -> tuple[str, float]:
        """
        Generate an answer given a question and retrieved passages.

        Returns:
            (answer_text, generation_time_ms)
        """
        context = _build_context(passages)
        prompt = RAG_PROMPT_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            context=context,
            question=question,
        )
        answer, elapsed = self._provider.generate(prompt)
        log.debug(
            "LLM answer generated — {:.1f} ms, {} chars",
            elapsed, len(answer),
        )
        return answer.strip(), elapsed

    @property
    def model_name(self) -> str:
        return f"{settings.LLM_PROVIDER}/{settings.LLM_MODEL}"


# Module-level singleton
_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
