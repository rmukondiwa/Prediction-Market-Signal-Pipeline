"""
Insight Generator: calls the OpenAI API to produce a natural-language market
insight summary and follow-up actions from a structured MarketSnapshot.

This is the only LLM-driven component in the pipeline.
All other steps (ingestion, parsing, normalization, extraction) are deterministic.
"""

import json
from typing import Any

from openai import OpenAI

from src.insight.models import InsightReport, LLMInsight, MarketSnapshot
from src.utils.logging import get_logger

logger = get_logger(__name__)

_MODEL = "gpt-5-mini"

_SYSTEM_PROMPT = """\
You are a quantitative prediction market analyst.
You receive structured market data from a prediction exchange and produce concise,
actionable market insights for a research and monitoring workflow.
Be precise, factual, and grounded in the data provided.
Do not speculate beyond what the data supports.\
"""

_USER_TEMPLATE = """\
Here is the latest normalized market snapshot from {source}:

Event:              {event}
Market:             {market}
Outcome:            {outcome}
Quoted Price:       {quoted_price}¢
Implied Probability:{implied_probability:.1%}
Yes Bid / Ask:      {yes_bid}¢ / {yes_ask}¢
Volume:             {volume:,} contracts
Open Interest:      {open_interest:,} contracts
Timestamp:          {timestamp}

Generate the following:

1. insight_summary — A concise paragraph (3–5 sentences) addressing:
   - What this market appears to be pricing
   - Any notable signal or aspect worth monitoring
   - Why this information matters from a research or risk perspective

2. follow_up_actions — Exactly 3 specific actions a larger automated monitoring
   system could take next (e.g. compare snapshots, flag changes, enrich data).\
"""


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively add additionalProperties: false to all object nodes."""
    schema = dict(schema)
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        if "properties" in schema:
            schema["properties"] = {
                k: _strict_schema(v) for k, v in schema["properties"].items()
            }
    if "items" in schema:
        schema["items"] = _strict_schema(schema["items"])
    if "$defs" in schema:
        schema["$defs"] = {k: _strict_schema(v) for k, v in schema["$defs"].items()}
    return schema


def generate_insight(snapshot: MarketSnapshot) -> InsightReport:
    """
    Call OpenAI to generate an insight summary and follow-up actions.
    Returns a fully populated InsightReport.
    """
    client = OpenAI()

    prompt = _USER_TEMPLATE.format(
        source=snapshot.source,
        event=snapshot.event,
        market=snapshot.market,
        outcome=snapshot.outcome,
        quoted_price=snapshot.quoted_price,
        implied_probability=snapshot.implied_probability,
        yes_bid=snapshot.yes_bid,
        yes_ask=snapshot.yes_ask,
        volume=snapshot.volume,
        open_interest=snapshot.open_interest,
        timestamp=snapshot.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
    )

    logger.info("Calling OpenAI for market insight", extra={"market": snapshot.market})

    response = client.responses.create(
        model=_MODEL,
        input=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "LLMInsight",
                "strict": True,
                "schema": _strict_schema(LLMInsight.model_json_schema()),
            }
        },
    )

    llm_insight = LLMInsight(**json.loads(response.output_text))

    return InsightReport(
        structured_data=snapshot,
        insight_summary=llm_insight.insight_summary,
        follow_up_actions=llm_insight.follow_up_actions,
    )
