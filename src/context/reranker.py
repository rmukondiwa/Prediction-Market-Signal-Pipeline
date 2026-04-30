"""
LLM reranking stage.

Takes the top-k vector search candidates and asks Groq (Llama 3.3 70B) to
score each one for causal/conditional relevance to the focus market.
Markets scoring below the threshold are dropped.

Candidate order is randomised before building the prompt to neutralise
the LLM's positional bias toward items listed first.
"""

import json
import random
from typing import TYPE_CHECKING

from openai import OpenAI

from src.catalog.models import CatalogMarket
from src.context.models import CandidateMarket, ContextMarket
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.backtest.cache import InferenceCache

logger = get_logger(__name__)

_MODEL = "gpt-4o-mini"

# Bumped on prompt changes — included in cache key so old entries naturally expire.
PROMPT_VERSION = "v1"

_SYSTEM_PROMPT = """\
You are a prediction market analyst specialising in cross-market causal inference.
Your task is to identify which candidate markets are causally or conditionally
related to the focus market — not just topically similar.

Look for:
- Direct causal links (event A causes event B)
- Shared underlying drivers (both affected by the same force)
- Conditional relationships (if A occurs, B becomes more or less likely)
- Temporal dependencies (A must happen before B can happen)

Do NOT score markets purely on keyword overlap. A market about "Iran military
activity" is causally related to "oil prices" even if they share no words.\
"""

_USER_TEMPLATE = """\
Focus market:
  Ticker: {ticker}
  Title: {title}
  Subtitle: {subtitle}
  Implied probability: {implied_prob:.1%}

Candidate markets to evaluate:
{candidates_block}

For each candidate, return a JSON object with this exact structure:
{{
  "rankings": [
    {{
      "ticker": "<ticker>",
      "relevance_score": <0-10 float>,
      "relationship": "<one sentence explaining the causal or conditional link>"
    }}
  ]
}}

Scoring guide:
  9-10  Strong direct causal link
  7-8   Shared underlying driver or strong conditional dependency
  5-6   Moderate conditional relationship
  3-4   Weak or indirect relationship
  0-2   No meaningful causal link (keyword match only)

Include ALL candidates in your response, even those scored 0.\
"""


def rerank(
    focus: CatalogMarket,
    candidates: list[CandidateMarket],
    score_threshold: float = 6.0,
    cache: "InferenceCache | None" = None,
    seed: int | None = None,
) -> list[ContextMarket]:
    """
    Score each candidate for causal relevance to the focus market via the LLM.
    Returns only those at or above score_threshold, sorted by relevance_score desc.
    Returns empty list if no candidates pass the threshold.

    cache: optional InferenceCache. When provided, response is cached by
           prompt+model+temperature+PROMPT_VERSION; temperature is forced to 0
           for reproducibility. Live callers pass None and behaviour is unchanged.
    seed:  optional RNG seed for the candidate shuffle. Live callers leave None
           (random); backtest callers pass a seed derived from the snapshot
           timestamp so the same input gives the same shuffle every replay.
    """
    if not candidates:
        logger.info("No candidates to rerank", extra={"focus": focus.ticker})
        return []

    # Randomise order to neutralise LLM positional bias
    shuffled = candidates.copy()
    if seed is not None:
        random.Random(seed).shuffle(shuffled)
    else:
        random.shuffle(shuffled)

    candidates_block = "\n".join(
        f"  [{i+1}] {c.ticker} | {c.title}"
        + (f" -- {c.subtitle}" if c.subtitle.strip().lower() not in {"", "yes", "no"} else "")
        + f" | implied: {c.implied_probability:.1%}"
        for i, c in enumerate(shuffled)
    )

    prompt = _USER_TEMPLATE.format(
        ticker=focus.ticker,
        title=focus.title,
        subtitle=focus.subtitle or "N/A",
        implied_prob=focus.implied_probability,
        candidates_block=candidates_block,
    )

    full_prompt_for_cache = f"SYSTEM:\n{_SYSTEM_PROMPT}\n\nUSER:\n{prompt}"
    response_payload: dict | None = None

    if cache is not None:
        cached = cache.get(full_prompt_for_cache, _MODEL, 0.0, PROMPT_VERSION)
        if cached is not None:
            response_payload = cached
            logger.info("Reranker cache hit", extra={"focus": focus.ticker})

    if response_payload is None:
        client = OpenAI()
        logger.info(
            "Calling reranker LLM",
            extra={"focus": focus.ticker, "candidates": len(shuffled),
                   "cached": False, "model": _MODEL},
        )
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=2000,
            temperature=0.0 if cache is not None else 1.0,
        )
        try:
            response_payload = json.loads(response.choices[0].message.content)
        except Exception as exc:
            logger.warning(
                "Failed to parse reranker response",
                extra={"focus": focus.ticker, "error": str(exc)},
            )
            return []
        if cache is not None:
            cache.put(full_prompt_for_cache, _MODEL, 0.0, PROMPT_VERSION, response_payload)

    rankings = response_payload.get("rankings", []) if response_payload else []

    # Build a lookup from the shuffled candidates
    candidate_map = {c.ticker: c for c in shuffled}

    context_markets: list[ContextMarket] = []
    for item in rankings:
        ticker = item.get("ticker")
        relevance_score = float(item.get("relevance_score", 0))
        relationship = item.get("relationship", "")

        if relevance_score < score_threshold:
            continue
        if ticker not in candidate_map:
            continue

        c = candidate_map[ticker]
        context_markets.append(
            ContextMarket(
                ticker=c.ticker,
                event_ticker=c.event_ticker,
                title=c.title,
                subtitle=c.subtitle,
                category=c.category,
                status=c.status,
                yes_bid=c.yes_bid,
                yes_ask=c.yes_ask,
                implied_probability=c.implied_probability,
                similarity_score=c.similarity_score,
                relevance_score=relevance_score,
                relationship=relationship,
            )
        )

    context_markets.sort(key=lambda x: x.relevance_score, reverse=True)

    logger.info(
        "Reranking complete",
        extra={
            "focus": focus.ticker,
            "passed": len(context_markets),
            "dropped": len(shuffled) - len(context_markets),
        },
    )
    return context_markets
