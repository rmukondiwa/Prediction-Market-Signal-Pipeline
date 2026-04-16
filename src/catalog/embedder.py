"""
Embedding pipeline.

Converts CatalogMarket records into text strings and calls the OpenAI
embeddings API in batches to produce dense vectors for the FAISS index.
"""

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


def embed_texts(
    texts: list[str],
    model: str = "text-embedding-3-small",
    dimensions: int = 512,
    batch_size: int = 100,
) -> list[list[float]]:
    """
    Embed a list of texts via OpenAI in batches of `batch_size`.
    Returns vectors in the same order as the input list.
    Total cost for 5k markets at 512 dims: ~$0.005.
    """
    client = OpenAI()
    embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=model,
            input=batch,
            dimensions=dimensions,
        )
        # Sort by index to guarantee ordering matches input
        batch_vecs = [
            item.embedding
            for item in sorted(response.data, key=lambda x: x.index)
        ]
        embeddings.extend(batch_vecs)
        logger.info(
            "Embedded batch",
            extra={
                "batch_start": i,
                "batch_end": i + len(batch),
                "total": len(texts),
            },
        )

    return embeddings
