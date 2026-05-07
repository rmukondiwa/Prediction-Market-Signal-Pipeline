"""
End-to-end LLM-signal edge measurement.

For each of N resolved markets:
  1. Load focus market metadata (settled outcome known)
  2. FAISS-retrieve k semantically-related context markets
  3. Call Gemini with the structured inference prompt (port of engine.py)
  4. Parse response into an InferenceReport
  5. Run all 5 signal models on it
  6. Record every emitted edge with its predicted side and the actual outcome

Aggregate outputs:
  - Per-signal hit rate (signal-suggested side wins / total emitted edges)
  - Per-signal naive PnL (suggested side at quoted_price → +/- $1)
  - Per-signal coverage (markets where signal emitted ≥1 edge)
  - Per-signal mean kelly (size proxy)

This is the "do these signals have measurable edge?" experiment. No full
inference cache hookup — uses Gemini directly via REST. Cost on n=30:
~$0.10 in Gemini tokens.

Usage:
    python -m scripts.measure_llm_signal_edge --n 30
    python -m scripts.measure_llm_signal_edge --n 50 --concurrency 2
    python -m scripts.measure_llm_signal_edge --resume   # reuses cached LLM responses
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from openai import OpenAI

from src.catalog.models import CatalogMarket
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


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
CACHE_DIR = Path("data/llm_edge_cache")


SYSTEM_PROMPT = """You are a quantitative prediction market analyst.
Given a focus market and a set of related context markets, identify any
mispricings and emit suggested edges with quantitative reasoning.

Be precise — every probability must be a specific number 0-1, not a range.
Every mispricing must cite specific prices and quantitative reasoning.

Respond with ONLY a JSON object:
{
  "consistency_analysis": "<one paragraph: do prices violate axioms?>",
  "derived_probabilities": [
    {"description": "...", "value": 0.X, "reasoning": "..."}
  ],
  "detected_mispricings": [
    {"ticker": "...", "title": "...", "direction": "overpriced|underpriced",
     "current_implied_prob": 0.X, "estimated_fair_prob": 0.X, "reasoning": "..."}
  ],
  "suggested_edges": [
    {"ticker": "...", "title": "...", "side": "yes|no",
     "confidence": "low|medium|high", "thesis": "...", "kelly_fraction": 0.X}
  ]
}"""


def cents_to_dollars(d: dict, k: str, default: float) -> float:
    v = d.get(k)
    if v is None or v == "":
        return default
    try:
        f = float(v)
        return f / 100.0 if f > 1.0 else f
    except (TypeError, ValueError):
        return default


def make_market_snapshot(m: dict) -> MarketSnapshot:
    yb_raw = int(float(m.get("yes_bid", 0) or 0))
    ya_raw = int(float(m.get("yes_ask", 100) or 100))
    if yb_raw == 0 and ya_raw == 100:
        try:
            mid_d = float(m.get("last_price_dollars", 0.5) or 0.5)
        except (TypeError, ValueError):
            mid_d = 0.5
        mc = max(1, min(99, int(mid_d * 100)))
        yb_raw, ya_raw = max(0, mc - 1), min(100, mc + 1)
    quoted = (yb_raw + ya_raw) // 2
    return MarketSnapshot(
        event=m.get("event_ticker") or m.get("ticker", "?").split("-")[0],
        market=m.get("ticker", "X"),
        outcome=m.get("title", "")[:200],
        quoted_price=quoted,
        implied_probability=quoted / 100.0,
        yes_bid=yb_raw,
        yes_ask=ya_raw,
        volume=int(float(m.get("volume_fp", 0) or 0)),
        open_interest=int(float(m.get("open_interest_fp", 0) or 0)),
        source="resolved_meta",
        timestamp=datetime.now(timezone.utc),
    )


def make_context_markets(hits: list[dict], focus_ticker: str) -> list[ContextMarket]:
    out: list[ContextMarket] = []
    for h in hits:
        if h.get("ticker") == focus_ticker:
            continue
        try:
            yb = int(float(h.get("yes_bid", 0) or 0))
            ya = int(float(h.get("yes_ask", 100) or 100))
        except (TypeError, ValueError):
            yb, ya = 0, 100
        cm_kwargs = {
            "ticker": h.get("ticker", "X"),
            "event_ticker": h.get("event_ticker", ""),
            "title": h.get("title", ""),
            "subtitle": h.get("subtitle", ""),
            "category": h.get("category", "Unknown"),
            "yes_bid": yb,
            "yes_ask": ya,
            "implied_probability": h.get("implied_probability", (yb + ya) / 200),
            "volume": int(float(h.get("volume_fp", 0) or 0)),
            "open_interest": int(float(h.get("open_interest_fp", 0) or 0)),
            "close_time": h.get("close_time", ""),
            "rules_primary": (h.get("rules_primary") or "")[:200],
        }
        try:
            cm = CatalogMarket(**cm_kwargs)
        except Exception:
            continue
        out.append(ContextMarket(
            **cm.model_dump(),
            similarity_score=float(h.get("score", 0.7)),
            relevance_score=8.0,
            relationship="related (FAISS-retrieved)",
        ))
    return out


def build_inference_prompt(snap: MarketSnapshot,
                            context: list[ContextMarket]) -> str:
    """Port of engine.py's _USER_TEMPLATE adapted for direct Gemini call."""
    if not context:
        ctx_block = "  (none — single-market analysis mode)"
    else:
        lines = []
        for c in context:
            lines.append(
                f"  {c.ticker} | {c.title}"
                + (f" -- {c.subtitle}" if c.subtitle.strip().lower() not in ("", "yes", "no") else "")
            )
            lines.append(
                f"    implied: {c.implied_probability:.1%} | similarity: {c.similarity_score:.2f}"
            )
        ctx_block = "\n".join(lines)
    return (
        f"FOCUS MARKET\n"
        f"  Ticker:        {snap.market}\n"
        f"  Title:         {snap.outcome}\n"
        f"  Implied prob:  {snap.implied_probability:.1%} ({snap.yes_bid}¢/{snap.yes_ask}¢)\n"
        f"  Volume:        {snap.volume:,}\n\n"
        f"CONTEXT MARKETS\n{ctx_block}\n\n"
        f"Analyze and respond with the JSON object specified in the system prompt."
    )


async def gemini_inference(session: aiohttp.ClientSession, api_key: str,
                            model: str, snap: MarketSnapshot,
                            context: list[ContextMarket],
                            sem: asyncio.Semaphore) -> dict:
    """Call Gemini with the inference prompt, parse JSON response."""
    user = build_inference_prompt(snap, context)
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 2000,
            "responseMimeType": "application/json",
        },
    }
    url = GEMINI_URL.format(model=model)
    backoff = 6.0
    async with sem:
        for attempt in range(5):
            try:
                async with session.post(
                    url, json=body,
                    headers={"X-goog-api-key": api_key, "Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as r:
                    if r.status == 429:
                        await asyncio.sleep(backoff)
                        backoff *= 1.7
                        continue
                    if r.status >= 400:
                        return {"error": f"HTTP {r.status}"}
                    data = await r.json()
                    break
            except Exception as e:
                return {"error": f"{type(e).__name__}: {e}"}
        else:
            return {"error": "rate-limit retries exhausted"}
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return {"error": "no candidate text"}
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except Exception as e:
        return {"error": f"json parse: {e}", "raw": cleaned[:300]}


def parse_inference_report(snap: MarketSnapshot,
                            context: list[ContextMarket],
                            raw: dict) -> InferenceReport | None:
    """Build InferenceReport from parsed JSON. None on parse failure."""
    if "error" in raw:
        return None
    try:
        derived = [DerivedProbability(**d) for d in raw.get("derived_probabilities", [])]
        mispricings = [Mispricing(**m) for m in raw.get("detected_mispricings", [])]
        edges = []
        for e in raw.get("suggested_edges", []):
            try:
                # Some LLM outputs miss/mis-spell kelly — default to 0.05
                e.setdefault("kelly_fraction", 0.05)
                edges.append(Edge(**e))
            except Exception:
                continue
        return InferenceReport(
            focus_market=snap,
            context_markets=context,
            consistency_analysis=raw.get("consistency_analysis", "")[:500],
            derived_probabilities=derived,
            detected_mispricings=mispricings,
            suggested_edges=edges,
        )
    except Exception as e:
        return None


def run_signals(report: InferenceReport,
                signals: dict,
                hist: HistoricalContext) -> dict:
    """Run all signals on the report. Returns {signal_name: edges}."""
    out: dict = {}
    for name, sig in signals.items():
        try:
            out[name] = sig.signals(
                focus=report.focus_market,
                context=report.context_markets,
                llm_report=report,
                history=hist,
            )
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    return out


async def main_async(args):
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY or OPENAI_API_KEY")

    print("=== End-to-end LLM signal edge measurement ===")

    # Sample resolved markets from archive
    rows_path = Path("experiments/decay/data/expanded_universe_meta.json")
    if not rows_path.exists():
        raise SystemExit(f"Missing {rows_path}")
    rows = [m for m in json.load(open(rows_path))
            if m.get("status") == "finalized" and m.get("result") in ("yes", "no")]
    random.seed(args.seed)
    sample = random.sample(rows, min(args.n, len(rows)))
    print(f"  Pool: {len(rows)} resolved, sampled {len(sample)}")

    print(f"  Loading FAISS index...")
    vs = VectorStore()
    vs.load()
    embed_client = OpenAI(api_key=api_key,
                           base_url=os.environ.get("OPENAI_BASE_URL"))

    signals = {
        "raw_llm": RawLLMSignal(),
        "calibrated_llm": CalibratedLLMSignal("data/calibration_map.pkl"),
        "consistency_arb": ConsistencyArbSignal(),
        "bayesian_base_rate": BayesianBaseRateSignal(),
        "coherence_regression": CoherenceRegressionSignal(),
    }
    hist = HistoricalContext()

    # Embed all focus titles in one batch (cheap)
    print(f"  Embedding {len(sample)} focus titles...")
    titles = [m.get("title", "") for m in sample]
    embed_resp = embed_client.embeddings.create(
        model="gemini-embedding-001", input=titles, dimensions=512,
    )
    focus_vecs = [d.embedding for d in embed_resp.data]

    # Build (snap, context) tuples
    pairs = []
    for m, v in zip(sample, focus_vecs):
        snap = make_market_snapshot(m)
        hits = vs.search(v, k=args.k + 1)
        ctx = make_context_markets(hits, focus_ticker=snap.market)[: args.k]
        pairs.append((m, snap, ctx))

    # Call inference engine for each — concurrent with rate-limit retry
    print(f"  Running Gemini inference on {len(pairs)} markets (concurrency={args.concurrency})...")
    sem = asyncio.Semaphore(args.concurrency)
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [
            gemini_inference(session, api_key, args.model, snap, ctx, sem)
            for _, snap, ctx in pairs
        ]
        for i, fut in enumerate(asyncio.as_completed(tasks), 1):
            r = await fut
            results.append(r)
            if i % 5 == 0:
                ok = sum(1 for x in results if "error" not in x)
                print(f"    {i}/{len(pairs)} done ({ok} valid)", flush=True)

    # Reorder results to match pairs order (asyncio.as_completed loses ordering)
    # → Re-run as gather to preserve order
    print(f"  (re-aligning results — using gather)")
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[
            gemini_inference(session, api_key, args.model, snap, ctx, sem)
            for _, snap, ctx in pairs
        ])

    # Score each
    per_signal: dict = defaultdict(lambda: {
        "edges_emitted": 0, "wins": 0, "losses": 0, "ties": 0,
        "naive_pnl": 0.0, "coverage_markets": 0, "mean_kelly": [],
        "errors": 0, "yes_emits": 0, "no_emits": 0,
    })
    n_inference_errors = 0
    detail = []
    for (m, snap, ctx), raw in zip(pairs, results):
        if "error" in raw:
            n_inference_errors += 1
            detail.append({"ticker": snap.market, "outcome": m["result"], "error": raw["error"]})
            continue
        report = parse_inference_report(snap, ctx, raw)
        if report is None:
            n_inference_errors += 1
            continue
        outcome = 1 if m["result"] == "yes" else 0
        market_price_yes = snap.implied_probability
        sigout = run_signals(report, signals, hist)
        row = {"ticker": snap.market, "outcome": m["result"],
               "n_mispricings": len(report.detected_mispricings),
               "n_edges_llm": len(report.suggested_edges)}
        for name, edges in sigout.items():
            ps = per_signal[name]
            if isinstance(edges, dict) and "error" in edges:
                ps["errors"] += 1
                continue
            if not edges:
                continue
            ps["coverage_markets"] += 1
            for e in edges:
                ps["edges_emitted"] += 1
                if e.side == "yes":
                    ps["yes_emits"] += 1
                    cost = max(0.01, market_price_yes)
                    payout = 1.0 if outcome == 1 else 0.0
                else:
                    ps["no_emits"] += 1
                    cost = max(0.01, 1.0 - market_price_yes)
                    payout = 1.0 if outcome == 0 else 0.0
                trade_pnl = payout - cost
                ps["naive_pnl"] += trade_pnl
                ps["mean_kelly"].append(e.kelly_fraction or 0.0)
                if (e.side == "yes" and outcome == 1) or (e.side == "no" and outcome == 0):
                    ps["wins"] += 1
                elif (e.side == "yes" and outcome == 0) or (e.side == "no" and outcome == 1):
                    ps["losses"] += 1
                else:
                    ps["ties"] += 1
            row[name] = {"n_edges": len(edges),
                         "sides": [e.side for e in edges]}
        detail.append(row)

    print(f"\n=== RESULTS ({len(detail)} markets, {n_inference_errors} inference errors) ===")
    print(f"\n  {'Signal':<22} {'Mkts':>5} {'Edges':>6} {'YES':>4} {'NO':>4} "
          f"{'Win':>4} {'Loss':>4} {'HitRate':>8} {'NaivePnL':>10} {'AvgKelly':>9}")
    for name in signals:
        s = per_signal[name]
        n_resolved = s["wins"] + s["losses"]
        hr = s["wins"] / n_resolved if n_resolved else 0.0
        avg_kelly = sum(s["mean_kelly"]) / len(s["mean_kelly"]) if s["mean_kelly"] else 0.0
        print(f"  {name:<22} {s['coverage_markets']:>5} {s['edges_emitted']:>6} "
              f"{s['yes_emits']:>4} {s['no_emits']:>4} "
              f"{s['wins']:>4} {s['losses']:>4} "
              f"{hr*100:>7.1f}% ${s['naive_pnl']:>+9.2f} {avg_kelly:>9.3f}")

    # Save full detail
    out_path = Path("reports/llm_signal_edge.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(),
        "n_markets": len(detail), "n_inference_errors": n_inference_errors,
        "model": args.model, "k_context": args.k,
        "per_signal": {k: {**v, "mean_kelly": sum(v["mean_kelly"]) / len(v["mean_kelly"]) if v["mean_kelly"] else 0.0}
                        for k, v in per_signal.items()},
        "detail": detail,
    }, indent=2, default=str))
    print(f"\n  → {out_path}")

    print("\n  Verdict per signal:")
    for name in signals:
        s = per_signal[name]
        n_resolved = s["wins"] + s["losses"]
        if n_resolved == 0:
            print(f"    {name}: no edges emitted across {len(detail)} markets — silent")
            continue
        hr = s["wins"] / n_resolved
        pnl_per_trade = s["naive_pnl"] / s["edges_emitted"]
        if hr >= 0.55 and pnl_per_trade > 0.03:
            print(f"    ✅ {name}: hit rate {hr*100:.0f}%, +${pnl_per_trade:.3f}/trade — has edge")
        elif hr >= 0.50:
            print(f"    ⚠️  {name}: hit rate {hr*100:.0f}%, +${pnl_per_trade:.3f}/trade — marginal")
        else:
            print(f"    ❌ {name}: hit rate {hr*100:.0f}%, ${pnl_per_trade:+.3f}/trade — no edge")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--model", default="gemini-2.0-flash")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
