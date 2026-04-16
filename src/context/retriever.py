"""
Vector retrieval stage.

Embeds the focus market's title and searches the FAISS index for the
top-k most similar markets. Returns CandidateMarket objects ready for reranking.
"""

from openai import OpenAI

from src.catalog.models import CatalogMarket
from src.catalog.vector_store import VectorStore
from src.context.models import CandidateMarket
from src.utils.logging import get_logger

logger = get_logger(__name__)

_EMBED_MODEL = "text-embedding-3-small"
_EMBED_DIM = 512
_SKIP_SUBTITLES = {"", "yes", "no"}


def retrieve_candidates(
    ticker: str,
    catalog: list[CatalogMarket],
    vector_store: VectorStore,
    k: int = 30,
) -> list[CandidateMarket]:
    """
    Find the focus market, embed its title, search the vector store for top-k
    similar markets, and return them as CandidateMarket objects (excluding self).

    If the ticker is not in the catalog (new or stale market), embeds the
    ticker string directly as a fallback.
    """
    client = OpenAI()

    focus: CatalogMarket | None = next(
        (m for m in catalog if m.ticker == ticker), None
    )

    if focus is None:
        logger.warning(
            "Ticker not found in catalog, embedding ticker string directly",
            extra={"ticker": ticker},
        )
        query_text = ticker
    else:
        subtitle = focus.subtitle.strip()
        query_text = (
            f"{focus.title} -- {subtitle}"
            if subtitle.lower() not in _SKIP_SUBTITLES
            else focus.title
        )

    response = client.embeddings.create(
        model=_EMBED_MODEL,
        input=[query_text],
        dimensions=_EMBED_DIM,
    )
    query_embedding = response.data[0].embedding

    # Fetch k+1 so we have room to drop the focus market itself
    raw_results = vector_store.search(query_embedding, k=k + 1)

    candidates: list[CandidateMarket] = []
    for r in raw_results:
        if r.get("ticker") == ticker:
            continue
        try:
            candidates.append(
                CandidateMarket(
                    ticker=r["ticker"],
                    event_ticker=r.get("event_ticker", ""),
                    title=r.get("title", ""),
                    subtitle=r.get("subtitle", ""),
                    category=r.get("category", ""),
                    status=r.get("status", "open"),
                    yes_bid=r.get("yes_bid", 0),
                    yes_ask=r.get("yes_ask", 0),
                    implied_probability=r.get("implied_probability", 0.0),
                    similarity_score=r["score"],
                )
            )
        except Exception as exc:
            logger.warning(
                "Skipping malformed candidate",
                extra={"ticker": r.get("ticker"), "error": str(exc)},
            )
        if len(candidates) >= k:
            break

    logger.info(
        "Retrieved candidates",
        extra={"focus": ticker, "candidates": len(candidates)},
    )
    return candidates
