"""
Structured inference engine.

Hot-path stage 4: takes a focus market snapshot + causal context markets,
sends a structured reasoning prompt to Groq, and returns a fully populated
InferenceReport with mispricings and quarter-Kelly position sizes.
"""

import json
from typing import TYPE_CHECKING

from openai import OpenAI

from src.context.models import ContextMarket
from src.inference.models import (
    DerivedProbability,
    Edge,
    InferenceReport,
    Mispricing,
)
from src.insight.models import MarketSnapshot
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.backtest.cache import InferenceCache

logger = get_logger(__name__)

_MODEL = "gpt-4o"

# Bumped on prompt changes — included in cache key so old entries naturally expire.
PROMPT_VERSION = "v1"

_SYSTEM_PROMPT = """\
You are a quantitative prediction market analyst specialising in cross-market inference.
You receive a focus market snapshot and a set of causally related context markets.
Your job is to reason across all of them to surface mispricings and actionable edges.

Be precise and quantitative. Every probability estimate must be a specific number,
not a range or qualitative description. Every mispricing must cite specific prices.\
"""

_USER_TEMPLATE = """\
FOCUS MARKET
  Ticker:            {ticker}
  Title:             {title}
  Implied prob:      {implied_prob:.1%}  ({yes_bid}¢ bid / {yes_ask}¢ ask)
  Volume:            {volume:,} contracts
  Open interest:     {open_interest:,} contracts
  Timestamp:         {timestamp}

CONTEXT MARKETS
{context_block}

Perform four types of analysis and return a single JSON object with this exact structure:
{{
  "consistency_analysis": "<paragraph: do any prices violate probability axioms? e.g. P(by June) > P(by December) is impossible>",
  "derived_probabilities": [
    {{
      "description": "<e.g. P(oil spike | Iran strike)>",
      "value": <0.0-1.0>,
      "reasoning": "<how you derived it from the priced markets>"
    }}
  ],
  "detected_mispricings": [
    {{
      "ticker": "<ticker>",
      "title": "<title>",
      "direction": "<overpriced or underpriced>",
      "current_implied_prob": <0.0-1.0>,
      "estimated_fair_prob": <0.0-1.0>,
      "reasoning": "<specific quantitative reasoning>"
    }}
  ],
  "suggested_edges": [
    {{
      "ticker": "<ticker>",
      "title": "<title>",
      "side": "<yes or no>",
      "confidence": "<low, medium, or high>",
      "thesis": "<why this is an edge>",
      "kelly_fraction": <computed by caller, set to 0.0>
    }}
  ]
}}

Analysis types to cover:
1. CONSISTENCY — Do prices violate probability axioms? P(A by June) must be <= P(A by December).
   Flag any violations with specific tickers and prices.
2. CONDITIONAL PROBABILITY — Derive implied conditional probabilities. E.g. if P(Iran strike)=0.30
   and P(oil spike AND Iran strike)=0.22, then P(oil spike | Iran strike)=0.73.
3. CROSS-MARKET DIVERGENCE — Do correlated markets agree? If rate cut is at 70% but bank stocks
   haven't moved, that's a divergence worth flagging.
4. STALE PRICING — Has one market stopped updating while related markets moved significantly?

For detected_mispricings and suggested_edges: only include markets where the edge is specific
and quantifiable. Do not include vague or speculative entries.\
"""


def _build_context_block(context: list[ContextMarket]) -> str:
    if not context:
        return "  (none — single-market analysis mode)"
    lines = []
    for c in context:
        lines.append(
            f"  {c.ticker} | {c.title}"
            + (f" -- {c.subtitle}" if c.subtitle.strip().lower() not in {"", "yes", "no"} else "")
        )
        lines.append(f"    implied: {c.implied_probability:.1%} | relevance: {c.relevance_score:.1f}/10")
        lines.append(f"    relationship: {c.relationship}")
    return "\n".join(lines)


def _compute_kelly_fraction(mispricing: Mispricing) -> float:
    """
    Quarter-Kelly position sizing.
    Returns a fraction clamped to [0.0, 0.25]. Returns 0.0 if no edge.
    """
    p = mispricing.estimated_fair_prob
    q = mispricing.current_implied_prob

    if q <= 0 or q >= 1:
        return 0.0

    # For "underpriced" we're buying yes; for "overpriced" we're buying no
    if mispricing.direction == "underpriced":
        edge = p - q
        payout = 1.0 / q - 1.0
    else:
        edge = (1 - p) - (1 - q)
        payout = 1.0 / (1 - q) - 1.0

    if payout <= 0 or edge <= 0:
        return 0.0

    kelly = (p * payout - (1 - p)) / payout
    quarter_kelly = kelly * 0.25
    return max(0.0, min(0.25, quarter_kelly))


def run_inference(
    snapshot: MarketSnapshot,
    context: list[ContextMarket],
    cache: "InferenceCache | None" = None,
) -> InferenceReport:
    """
    Run cross-market inference against the focus snapshot and its context markets.
    Returns a fully populated InferenceReport with quarter-Kelly position sizes.
    Falls back to single-market analysis when context is empty.

    cache: optional InferenceCache. Live callers pass None (no caching);
           backtest callers pass a cache for reproducibility. When the cache is
           in use, temperature is forced to 0 and the response is cached by
           prompt+model+temperature+PROMPT_VERSION.
    """
    prompt = _USER_TEMPLATE.format(
        ticker=snapshot.market,
        title=snapshot.event,
        implied_prob=snapshot.implied_probability,
        yes_bid=snapshot.yes_bid,
        yes_ask=snapshot.yes_ask,
        volume=snapshot.volume,
        open_interest=snapshot.open_interest,
        timestamp=snapshot.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
        context_block=_build_context_block(context),
    )

    full_prompt_for_cache = f"SYSTEM:\n{_SYSTEM_PROMPT}\n\nUSER:\n{prompt}"
    raw: dict | None = None

    if cache is not None:
        cached = cache.get(full_prompt_for_cache, _MODEL, 0.0, PROMPT_VERSION)
        if cached is not None:
            raw = cached
            logger.info("Inference cache hit", extra={"market": snapshot.market})

    if raw is None:
        client = OpenAI()
        logger.info(
            "Calling inference LLM",
            extra={"market": snapshot.market, "context_markets": len(context),
                   "cached": False, "model": _MODEL},
        )
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=4000,
            temperature=0.0 if cache is not None else 1.0,
        )
        try:
            raw = json.loads(response.choices[0].message.content)
        except Exception as exc:
            logger.warning(
                "Failed to parse inference response",
                extra={"market": snapshot.market, "error": str(exc)},
            )
            raise
        if cache is not None:
            cache.put(full_prompt_for_cache, _MODEL, 0.0, PROMPT_VERSION, raw)

    derived_probabilities = [
        DerivedProbability(**item)
        for item in raw.get("derived_probabilities", [])
    ]

    detected_mispricings = [
        Mispricing(**item)
        for item in raw.get("detected_mispricings", [])
    ]

    # Compute quarter-Kelly for each mispricing and build Edge objects
    mispricing_map = {m.ticker: m for m in detected_mispricings}
    suggested_edges: list[Edge] = []
    for item in raw.get("suggested_edges", []):
        ticker = item.get("ticker", "")
        mispricing = mispricing_map.get(ticker)
        kelly = _compute_kelly_fraction(mispricing) if mispricing else 0.0

        suggested_edges.append(
            Edge(
                ticker=ticker,
                title=item.get("title", ""),
                side=item.get("side", "yes"),
                confidence=item.get("confidence", "low"),
                thesis=item.get("thesis", ""),
                kelly_fraction=kelly,
            )
        )

    report = InferenceReport(
        focus_market=snapshot,
        context_markets=context,
        consistency_analysis=raw.get("consistency_analysis", ""),
        derived_probabilities=derived_probabilities,
        detected_mispricings=detected_mispricings,
        suggested_edges=suggested_edges,
    )

    logger.info(
        "Inference complete",
        extra={
            "market": snapshot.market,
            "mispricings": len(detected_mispricings),
            "edges": len(suggested_edges),
        },
    )
    return report
