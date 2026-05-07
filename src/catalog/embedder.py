"""
Embedding pipeline.

Converts CatalogMarket records into text strings and calls the embeddings
API (OpenAI or Gemini via OpenAI-compatible endpoint) in batches to produce
dense vectors for the FAISS index.

Provider selection:
  - If OPENAI_BASE_URL is set in env (e.g., Gemini's compat endpoint), uses
    that with the matching default model (`gemini-embedding-001`).
  - Otherwise falls back to native OpenAI (`text-embedding-3-small`).

Both providers respect the `dimensions` parameter for output truncation,
so a 512-dim FAISS index works with either provider as long as the model
chosen supports the requested dimensionality.
"""
import os
import time

from openai import OpenAI

from src.catalog.models import CatalogMarket
from src.utils.logging import get_logger

logger = get_logger(__name__)

_SKIP_SUBTITLES = {"", "yes", "no"}


def build_embedding_inputs(catalog: list[CatalogMarket]) -> list[str]:
    """
    Produce one text string per market for embedding.
    Format: "{title} -- {subtitle}" or just "{title}" when subtitle is trivial.
    """
    inputs: list[str] = []
    for m in catalog:
        subtitle = m.subtitle.strip()
        if subtitle.lower() in _SKIP_SUBTITLES:
            inputs.append(m.title)
        else:
            inputs.append(f"{m.title} -- {subtitle}")
    return inputs


def _default_model() -> str:
    """Pick a sensible default model based on which provider is configured.
    Gemini compat endpoint → gemini-embedding-001, else OpenAI default."""
    base = os.getenv("OPENAI_BASE_URL", "")
    if "generativelanguage.googleapis.com" in base:
        return "gemini-embedding-001"
    return "text-embedding-3-small"


def embed_texts(
    texts: list[str],
    model: str | None = None,
    dimensions: int = 512,
    batch_size: int = 100,
    client: OpenAI | None = None,
    rate_limit_delay_s: float = 0.0,
    retries_on_429: int = 5,
) -> list[list[float]]:
    """
    Embed a list of texts in batches of `batch_size`.
    Returns vectors in the same order as the input list.

    Both OpenAI and Gemini (via OpenAI-compat endpoint) respect `dimensions`,
    so the same 512-dim FAISS index works with either provider.

    `rate_limit_delay_s`: optional throttle between batches (Gemini free
    tier: 15 RPM, set to ~4.0 for safety at batch_size=100).
    `retries_on_429`: per-batch retry budget on rate-limit errors.
    """
    if client is None:
        client = OpenAI()
    if model is None:
        model = _default_model()
    embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        backoff = 6.0
        last_error: Exception | None = None
        for attempt in range(retries_on_429):
            try:
                response = client.embeddings.create(
                    model=model, input=batch, dimensions=dimensions,
                )
                break
            except Exception as e:
                msg = str(e).lower()
                if "429" in msg or "rate" in msg or "quota" in msg or "exhaust" in msg:
                    last_error = e
                    logger.warning("Embedding 429 — backing off",
                                   extra={"sleep_s": backoff, "attempt": attempt + 1})
                    time.sleep(backoff)
                    backoff *= 1.7
                    continue
                raise
        else:
            raise RuntimeError(f"Embedding failed after retries: {last_error}")

        # Sort by index when present (OpenAI). Gemini's compat endpoint
        # returns items in input order without populating .index — preserve
        # the original ordering when sorting isn't safe.
        try:
            batch_vecs = [
                item.embedding
                for item in sorted(response.data, key=lambda x: x.index)
            ]
        except TypeError:
            batch_vecs = [item.embedding for item in response.data]
        embeddings.extend(batch_vecs)
        logger.info(
            "Embedded batch",
            extra={
                "batch_start": i,
                "batch_end": i + len(batch),
                "total": len(texts),
                "model": model,
            },
        )
        if rate_limit_delay_s > 0 and i + batch_size < len(texts):
            time.sleep(rate_limit_delay_s)

    return embeddings
