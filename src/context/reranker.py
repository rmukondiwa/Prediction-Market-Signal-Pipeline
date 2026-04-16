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

from groq import Groq

from src.catalog.models import CatalogMarket
from src.context.models import CandidateMarket, ContextMarket
from src.utils.logging import get_logger

logger = get_logger(__name__)

_GROQ_MODEL = "llama-3.3-70b-versatile"

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
) -> list[ContextMarket]:
    """
    Score each candidate for causal relevance to the focus market via Groq LLM.
    Returns only those at or above score_threshold, sorted by relevance_score desc.
    Returns empty list if no candidates pass the threshold.
    """
    if not candidates:
        logger.info("No candidates to rerank", extra={"focus": focus.ticker})
        return []

    client = Groq()

    # Randomise order to neutralise LLM positional bias
    shuffled = candidates.copy()
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

    logger.info(
        "Calling Groq reranker",
        extra={"focus": focus.ticker, "candidates": len(shuffled)},
    )

    response = client.chat.completions.create(
        model=_GROQ_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=2000,
    )

    try:
        rankings = json.loads(response.choices[0].message.content).get("rankings", [])
    except Exception as exc:
        logger.warning(
            "Failed to parse reranker response",
            extra={"focus": focus.ticker, "error": str(exc)},
        )
        return []

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
