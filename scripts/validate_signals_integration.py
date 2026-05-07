"""
Integration smoke test for all 5 LLM signal models.

For each of N test markets:
  1. Retrieve k context markets via FAISS (real Gemini-embedded vectors)
  2. Construct a realistic `InferenceReport` from cached metadata
     (LLM-derived fields like mispricings/edges populated minimally —
     this is integration smoke, not edge measurement)
  3. Run all 5 signals on the report
  4. Record outputs (edges emitted, sides, kelly fractions)

Goals:
  - Confirm each signal runs without crashing on realistic input
  - Capture which signals emit edges, which stay silent
  - Surface any prefilter logic that's broken
  - Provide a baseline for future end-to-end edge validation

NOT a P&L test — that requires real LLM-derived mispricings on resolved
markets, which is a separate ~$5/run pipeline.

Usage:
    python -m scripts.validate_signals_integration --n 20
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.catalog.vector_store import VectorStore
from src.context.models import ContextMarket
from src.inference.models import (
    DerivedProbability,
    Edge,
    InferenceReport,
    Mispricing,
)
from src.insight.models import MarketSnapshot
from src.signals.bayesian_base_rate import BayesianBaseRateSignal
from src.signals.calibrated_llm import CalibratedLLMSignal
from src.signals.coherence_regression import CoherenceRegressionSignal
from src.signals.consistency_arb import ConsistencyArbSignal
from src.signals.models import HistoricalContext
from src.signals.raw_llm import RawLLMSignal


def load_resolved() -> list[dict]:
    """Resolved markets are now in experiments/decay/data/."""
    path = Path("experiments/decay/data/expanded_universe_meta.json")
    if not path.exists():
        return []
    rows = json.load(open(path))
    return [m for m in rows if m.get("status") == "finalized" and m.get("result") in ("yes", "no")]


def cents_to_dollars(d: dict, k: str, default: float = 0.5) -> float:
    """Catalog prices are in cents — normalize to dollars."""
    v = d.get(k)
    if v is None or v == "":
        return default
    try:
        f = float(v)
        return f / 100.0 if f > 1.0 else f
    except (TypeError, ValueError):
        return default


def make_market_snapshot(m: dict) -> MarketSnapshot:
    """Build a MarketSnapshot from a catalog/metadata dict.

    The schema uses cents (int) for yes_bid/yes_ask and quoted_price.
    We pull from raw catalog fields, fall back to last_price_dollars when
    the order book is missing.
    """
    yb_raw = m.get("yes_bid", 0) or 0
    ya_raw = m.get("yes_ask", 100) or 100
    try:
        yb = int(float(yb_raw))
        ya = int(float(ya_raw))
    except (TypeError, ValueError):
        yb, ya = 0, 100
    if yb == 0 and ya == 100:
        # Degenerate quote → derive from last_price
        try:
            mid_dollars = float(m.get("last_price_dollars", 0.5) or 0.5)
        except (TypeError, ValueError):
            mid_dollars = 0.5
        mid_cents = max(1, min(99, int(mid_dollars * 100)))
        yb = max(0, mid_cents - 1)
        ya = min(100, mid_cents + 1)
    quoted = (yb + ya) // 2
    return MarketSnapshot(
        event=m.get("event_ticker") or m.get("ticker", "?").split("-")[0],
        market=m.get("ticker", "X"),
        outcome=m.get("title", "")[:200],
        quoted_price=quoted,
        implied_probability=quoted / 100.0,
        yes_bid=yb,
        yes_ask=ya,
        volume=int(float(m.get("volume_fp", 0) or 0)),
        open_interest=int(float(m.get("open_interest_fp", 0) or 0)),
        source="catalog_meta",
        timestamp=datetime.now(timezone.utc),
    )


def make_context_markets(hits: list[dict], focus_ticker: str) -> list[ContextMarket]:
    """Build ContextMarket list from FAISS hits, dropping the focus itself."""
    out: list[ContextMarket] = []
    for h in hits:
        if h.get("ticker") == focus_ticker:
            continue
        # Compute a mid from cached prices if available
        yb = cents_to_dollars(h, "yes_bid", default=0.0)
        ya = cents_to_dollars(h, "yes_ask", default=1.0)
        if yb == 0.0 and ya == 1.0:
            try:
                mid = float(h.get("last_price_dollars", 0.5))
            except (TypeError, ValueError):
                mid = 0.5
            yb, ya = max(0.0, mid - 0.01), min(1.0, mid + 0.01)
        try:
            from src.catalog.models import CatalogMarket
            cm_kwargs = {
                "ticker": h.get("ticker", "X"),
                "event_ticker": h.get("event_ticker", ""),
                "title": h.get("title", ""),
                "subtitle": h.get("subtitle", ""),
                "category": h.get("category", "Unknown"),
                "yes_bid": yb,
                "yes_ask": ya,
                "implied_probability": h.get("implied_probability", (yb + ya) / 2),
                "volume": int(float(h.get("volume_fp", 0) or 0)),
                "open_interest": int(float(h.get("open_interest_fp", 0) or 0)),
                "close_time": h.get("close_time", ""),
                "rules_primary": h.get("rules_primary", "")[:200],
            }
            cm = CatalogMarket(**cm_kwargs)
        except Exception:
            continue
        out.append(ContextMarket(
            **cm.model_dump(),
            similarity_score=float(h.get("score", 0.7)),
            relevance_score=8.0,  # synthetic — would be LLM-rated in production
            relationship="related (synthetic — placeholder)",
        ))
    return out


def make_inference_report(focus_snap: MarketSnapshot,
                           context: list[ContextMarket]) -> InferenceReport:
    """Construct a minimal InferenceReport. Without real LLM inference,
    derived_probabilities/mispricings/edges are empty — exercising the
    signals' fallback-on-empty-inference paths."""
    return InferenceReport(
        focus_market=focus_snap,
        context_markets=context,
        consistency_analysis="(synthetic placeholder — no LLM inference run)",
        derived_probabilities=[],
        detected_mispricings=[],
        suggested_edges=[],
    )


def run_one(snap: MarketSnapshot, ctx: list[ContextMarket],
            report: InferenceReport, hist: HistoricalContext,
            signals: dict) -> dict:
    """Run all signals on one constructed report."""
    out: dict = {"ticker": snap.market}
    for name, sig in signals.items():
        try:
            edges = sig.signals(focus=snap, context=ctx, llm_report=report, history=hist)
            out[name] = {
                "n_edges": len(edges),
                "sides": [e.side for e in edges],
                "kelly_max": max((e.kelly_fraction for e in edges), default=0.0),
            }
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20, help="Number of test markets")
    p.add_argument("--k", type=int, default=10, help="Context size per market")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("reports/signals_integration.json"))
    args = p.parse_args()

    load_dotenv()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                    base_url=os.environ["OPENAI_BASE_URL"])

    print("=== Signals Integration Smoke Test ===")
    rows = load_resolved()
    if not rows:
        raise SystemExit("No resolved markets found — expected experiments/decay/data/")

    import random
    random.seed(args.seed)
    sample = random.sample(rows, min(args.n, len(rows)))
    print(f"  Pool: {len(rows)} resolved, sampled {len(sample)}")

    print("  Loading FAISS index...")
    vs = VectorStore()
    vs.load()
    print(f"    {vs.index.ntotal} vectors @ {vs.index.d}-dim")

    signals = {
        "raw_llm": RawLLMSignal(),
        "calibrated_llm": CalibratedLLMSignal("data/calibration_map.pkl"),
        "consistency_arb": ConsistencyArbSignal(),
        "bayesian_base_rate": BayesianBaseRateSignal(),
        "coherence_regression": CoherenceRegressionSignal(),
    }
    hist = HistoricalContext()  # default empty resolutions/candles

    results = []
    for i, m in enumerate(sample, 1):
        title = m.get("title", "")
        if not title:
            continue
        # Embed focus title and search FAISS
        e = client.embeddings.create(
            model="gemini-embedding-001", input=title, dimensions=512,
        )
        hits = vs.search(e.data[0].embedding, k=args.k + 1)
        snap = make_market_snapshot(m)
        ctx = make_context_markets(hits, focus_ticker=snap.market)
        report = make_inference_report(snap, ctx)
        row = run_one(snap, ctx, report, hist, signals)
        row["outcome"] = m.get("result")
        results.append(row)
        if i % 5 == 0:
            print(f"    {i}/{len(sample)} done")

    # Aggregate
    summary: dict = {sig: {"runs": 0, "errors": 0, "edges_emitted": 0,
                           "yes_edges": 0, "no_edges": 0, "max_kelly": 0.0}
                     for sig in signals}
    for r in results:
        for sig in signals:
            d = r.get(sig, {})
            summary[sig]["runs"] += 1
            if "error" in d:
                summary[sig]["errors"] += 1
            else:
                summary[sig]["edges_emitted"] += d["n_edges"]
                summary[sig]["yes_edges"] += d["sides"].count("yes")
                summary[sig]["no_edges"] += d["sides"].count("no")
                if d["kelly_max"] > summary[sig]["max_kelly"]:
                    summary[sig]["max_kelly"] = d["kelly_max"]

    print("\n=== SUMMARY ===")
    print(f"  {'Signal':<22} {'Runs':>5} {'Err':>4} {'Edges':>6} {'YES':>4} {'NO':>4} {'KellyMax':>9}")
    for sig, s in summary.items():
        print(f"  {sig:<22} {s['runs']:>5} {s['errors']:>4} "
              f"{s['edges_emitted']:>6} {s['yes_edges']:>4} {s['no_edges']:>4} "
              f"{s['max_kelly']:>9.3f}")

    # Save full per-market detail
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "n": len(results), "k": args.k,
        "summary": summary,
        "rows": results,
    }, indent=2, default=str))
    print(f"\n  Report: {args.output}")

    # Quick conclusion
    print("\nObservations:")
    for sig, s in summary.items():
        if s["errors"] > 0:
            print(f"  ❌ {sig}: {s['errors']}/{s['runs']} runs errored — needs fix")
        elif s["edges_emitted"] == 0:
            print(f"  ⚠️  {sig}: 0 edges across {s['runs']} markets — signal stays silent on empty-LLM input (expected for some)")
        else:
            print(f"  ✅ {sig}: emits {s['edges_emitted']} edges across {s['runs']} markets")


if __name__ == "__main__":
    main()
