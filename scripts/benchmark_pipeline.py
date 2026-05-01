"""
End-to-end pipeline latency benchmark.

Measures each stage:
    1. catalog + FAISS load
    2. FAISS vector search (focus vector reconstructed from index — no
       OpenAI embedding call needed, since the focus is already indexed)
    3. LLM rerank call
    4. LLM inference call
    5. report parsing

Uses Gemini directly (native API, JSON mode) so we don't need an OpenAI key.
The prompts are identical to those in src/context/reranker.py and
src/inference/engine.py — only the model and HTTP transport change.

Usage:
    python -m scripts.benchmark_pipeline --ticker KXBTCD-26DEC31-T200000
    python -m scripts.benchmark_pipeline --auto      # pick a busy ticker
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.request
from collections import Counter
from pathlib import Path

import faiss
import numpy as np

from src.catalog.models import CatalogMarket
from src.catalog.store import load_catalog
from src.catalog.vector_store import VectorStore
from src.context.models import CandidateMarket
from src.context.reranker import _SYSTEM_PROMPT as RERANK_SYS, _USER_TEMPLATE as RERANK_USER
from src.inference.engine import (
    _SYSTEM_PROMPT as INFER_SYS,
    _USER_TEMPLATE as INFER_USER,
    _build_context_block,
    _compute_kelly_fraction,
)
from src.inference.models import (
    DerivedProbability, Edge, InferenceReport, Mispricing,
)
from src.insight.models import MarketSnapshot
from datetime import datetime, timezone


GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


def gemini_call(system: str, user: str, api_key: str, max_tokens: int = 8000) -> tuple[dict, float]:
    """One Gemini call with timing. Returns (parsed_json, wall_seconds)."""
    body = {
        "contents": [{"parts": [{"text": user}]}],
        "systemInstruction": {"parts": [{"text": system}]},
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
            # 2.5-flash defaults to a thinking budget that adds wall-clock
            # latency before the first output token. Set to 0 for fastest
            # response (the rerank/inference tasks are well-structured enough
            # not to need explicit chain-of-thought).
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    req = urllib.request.Request(
        GEMINI_URL, method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-goog-api-key": api_key},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    elapsed = time.perf_counter() - t0
    payload = json.loads(raw)
    candidates = payload.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {payload}")
    parts = candidates[0].get("content", {}).get("parts") or []
    if not parts:
        raise RuntimeError(f"Gemini returned candidate with no parts: {candidates[0]}")
    text = parts[0].get("text", "")
    try:
        return json.loads(text), elapsed
    except json.JSONDecodeError as e:
        snippet = text[:600] + ("..." if len(text) > 600 else "")
        raise RuntimeError(f"Gemini JSON parse error at {e}: response begins:\n{snippet}") from e


def pick_busy_ticker(vs: VectorStore, catalog: list[CatalogMarket]) -> str:
    """Find a focus market with a rich event family — gives the grouping
    something interesting to do."""
    by_event: dict[str, list] = {}
    for c in catalog:
        by_event.setdefault(c.event_ticker, []).append(c)
    # Prefer events with lots of markets and a non-trivial title
    busy = sorted(by_event.items(), key=lambda kv: len(kv[1]), reverse=True)
    for event, ms in busy:
        if 4 <= len(ms) <= 20:
            for m in ms:
                if 2 <= m.yes_bid <= 98 and (m.yes_ask - m.yes_bid) <= 10:
                    return m.ticker
    return catalog[0].ticker


def fmt_ms(s: float) -> str:
    return f"{s*1000:7.1f} ms"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", type=str)
    p.add_argument("--auto", action="store_true", help="Pick a busy ticker automatically")
    p.add_argument("--k", type=int, default=30, help="Vector search top-k")
    p.add_argument("--api-key", type=str,
                   default=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Set GEMINI_API_KEY or pass --api-key")

    print(f"=== Pipeline benchmark (model: {GEMINI_MODEL}) ===\n")

    # Stage 1: load
    t0 = time.perf_counter()
    catalog = load_catalog()
    vs = VectorStore()
    vs.load("data/vectors")
    load_s = time.perf_counter() - t0
    print(f"[1] load catalog ({len(catalog)} mkts) + FAISS index   {fmt_ms(load_s)}")

    # Pick focus
    if args.ticker:
        focus_ticker = args.ticker
    elif args.auto:
        focus_ticker = pick_busy_ticker(vs, catalog)
    else:
        focus_ticker = pick_busy_ticker(vs, catalog)
    focus = next((m for m in catalog if m.ticker == focus_ticker), None)
    if focus is None:
        raise SystemExit(f"Ticker {focus_ticker} not in catalog")
    focus_idx = next((i for i, d in enumerate(vs.documents) if d.get("ticker") == focus_ticker), None)
    if focus_idx is None:
        raise SystemExit(f"Ticker {focus_ticker} not in vector index")
    print(f"    focus: {focus.ticker} | {focus.title[:80]}")
    print(f"           bid={focus.yes_bid}¢ ask={focus.yes_ask}¢ implied={focus.implied_probability:.1%}\n")

    # Stage 2: FAISS search using the precomputed focus vector
    t0 = time.perf_counter()
    vec = vs.index.reconstruct(focus_idx).reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(vec)
    scores, indices = vs.index.search(vec, args.k + 1)
    search_s = time.perf_counter() - t0
    print(f"[2] FAISS search (k={args.k}, vector reconstructed)    {fmt_ms(search_s)}")

    # Build candidates (skip the focus market itself)
    candidates: list[CandidateMarket] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == focus_idx or idx < 0:
            continue
        d = vs.documents[idx]
        candidates.append(CandidateMarket(
            ticker=d["ticker"], event_ticker=d.get("event_ticker", ""),
            title=d.get("title", ""), subtitle=d.get("subtitle", ""),
            category=d.get("category", ""), status=d.get("status", "open"),
            yes_bid=d.get("yes_bid", 0), yes_ask=d.get("yes_ask", 0),
            implied_probability=d.get("implied_probability", 0.0),
            similarity_score=float(score),
        ))
        if len(candidates) >= args.k:
            break

    # Stage 3: rerank via Gemini (same prompt as src/context/reranker.py)
    shuffled = candidates.copy()
    random.Random(focus.ticker).shuffle(shuffled)
    candidates_block = "\n".join(
        f"  [{i+1}] {c.ticker} | {c.title}"
        + (f" -- {c.subtitle}" if c.subtitle.strip().lower() not in {"", "yes", "no"} else "")
        + f" | implied: {c.implied_probability:.1%}"
        for i, c in enumerate(shuffled)
    )
    rerank_user = RERANK_USER.format(
        ticker=focus.ticker, title=focus.title,
        subtitle=focus.subtitle or "N/A",
        implied_prob=focus.implied_probability,
        candidates_block=candidates_block,
    )

    rerank_payload, rerank_s = gemini_call(RERANK_SYS, rerank_user, args.api_key, max_tokens=8000)
    print(f"[3] LLM rerank ({GEMINI_MODEL})                        {fmt_ms(rerank_s)}")

    rankings = rerank_payload.get("rankings", [])
    by_score = sorted(rankings, key=lambda r: float(r.get("relevance_score", 0)), reverse=True)
    context_markets = []
    cand_map = {c.ticker: c for c in shuffled}
    for r in by_score:
        if float(r.get("relevance_score", 0)) < 6.0:
            continue
        c = cand_map.get(r.get("ticker"))
        if c is None:
            continue
        from src.context.models import ContextMarket
        context_markets.append(ContextMarket(
            ticker=c.ticker, event_ticker=c.event_ticker,
            title=c.title, subtitle=c.subtitle,
            category=c.category, status=c.status,
            yes_bid=c.yes_bid, yes_ask=c.yes_ask,
            implied_probability=c.implied_probability,
            similarity_score=c.similarity_score,
            relevance_score=float(r.get("relevance_score", 0)),
            relationship=r.get("relationship", ""),
        ))

    # Stage 4: inference (same prompt as src/inference/engine.py)
    snapshot = MarketSnapshot(
        event=focus.title, market=focus.ticker, outcome="YES",
        quoted_price=int((focus.yes_bid + focus.yes_ask) / 2),
        implied_probability=focus.implied_probability,
        yes_bid=focus.yes_bid, yes_ask=focus.yes_ask,
        volume=0, open_interest=0, source="catalog",
        timestamp=datetime.now(timezone.utc),
    )
    infer_user = INFER_USER.format(
        ticker=snapshot.market, title=snapshot.event,
        implied_prob=snapshot.implied_probability,
        yes_bid=snapshot.yes_bid, yes_ask=snapshot.yes_ask,
        volume=snapshot.volume, open_interest=snapshot.open_interest,
        timestamp=snapshot.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
        context_block=_build_context_block(context_markets),
    )
    infer_payload, infer_s = gemini_call(INFER_SYS, infer_user, args.api_key, max_tokens=8000)
    print(f"[4] LLM inference ({GEMINI_MODEL})                     {fmt_ms(infer_s)}")

    # Stage 5: parse + assemble report (deterministic, no network)
    t0 = time.perf_counter()
    derived = [DerivedProbability(**x) for x in infer_payload.get("derived_probabilities", [])]
    misps = [Mispricing(**x) for x in infer_payload.get("detected_mispricings", [])]
    misp_map = {m.ticker: m for m in misps}
    edges = []
    for x in infer_payload.get("suggested_edges", []):
        kf = _compute_kelly_fraction(misp_map[x["ticker"]]) if x.get("ticker") in misp_map else 0.0
        edges.append(Edge(
            ticker=x.get("ticker", ""), title=x.get("title", ""),
            side=x.get("side", "yes"), confidence=x.get("confidence", "low"),
            thesis=x.get("thesis", ""), kelly_fraction=kf,
        ))
    report = InferenceReport(
        focus_market=snapshot, context_markets=context_markets,
        consistency_analysis=infer_payload.get("consistency_analysis", ""),
        derived_probabilities=derived,
        detected_mispricings=misps,
        suggested_edges=edges,
    )
    parse_s = time.perf_counter() - t0
    print(f"[5] parse + assemble report                           {fmt_ms(parse_s)}")

    total = load_s + search_s + rerank_s + infer_s + parse_s
    hot = search_s + rerank_s + infer_s + parse_s
    print(f"\n  TOTAL                                                {fmt_ms(total)}")
    print(f"  HOT PATH (excludes one-time load)                   {fmt_ms(hot)}")
    print(f"  Network-bound:  rerank + infer = {fmt_ms(rerank_s + infer_s)} ({(rerank_s+infer_s)/hot*100:.0f}% of hot path)")
    print(f"  Local-bound:    search + parse = {fmt_ms(search_s + parse_s)}")

    # Show grouping quality
    print(f"\n=== Grouping quality (top 8 of {len(context_markets)} context markets) ===\n")
    if not context_markets:
        print("  (no context markets passed the relevance threshold)")
    else:
        for i, c in enumerate(context_markets[:8], 1):
            print(f"  [{i}] score={c.relevance_score:.1f}/10  sim={c.similarity_score:.2f}")
            print(f"      {c.ticker:<55}  implied={c.implied_probability:.1%}")
            print(f"      {c.title[:90]}")
            print(f"      → {c.relationship[:130]}")
            print()

    # Same-event vs cross-event split
    same_event = sum(1 for c in context_markets if c.event_ticker == focus.event_ticker)
    cross = len(context_markets) - same_event
    print(f"  Same-event hits:  {same_event}  (deadline-stacked or bin neighbours)")
    print(f"  Cross-event hits: {cross}    (causal/conditional links beyond keyword overlap)")

    # Show inference output briefly
    print(f"\n=== Inference output ===\n")
    print(f"  consistency: {report.consistency_analysis[:200]}")
    print(f"  derived probs:    {len(report.derived_probabilities)}")
    print(f"  mispricings:      {len(report.detected_mispricings)}")
    print(f"  suggested edges:  {len(report.suggested_edges)}")
    if report.suggested_edges:
        e = report.suggested_edges[0]
        print(f"\n  Top edge: {e.ticker} {e.side} ({e.confidence}, kelly={e.kelly_fraction:.3f})")
        print(f"            {e.thesis[:200]}")


if __name__ == "__main__":
    main()
