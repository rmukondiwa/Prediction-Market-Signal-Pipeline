"""
Validate LLM signal edge against resolved-outcome data.

Picks N resolved binary markets, asks Gemini to estimate P(YES) given only
the question text (no market-implied price), and compares to actual outcomes.

Outputs:
  - Brier score (lower is better; 0.25 = chance-equivalent for 50/50)
  - Calibration table (predicted_bucket → actual_yes_rate)
  - Naive PnL: bet on the side LLM favors at $0.50 unit stake

Usage:
    python -m scripts.validate_llm_signal --n 50 --model gemini-2.0-flash
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def load_resolved_markets() -> list[dict]:
    expanded = json.load(open("data/expanded_universe_meta.json"))
    decay = json.load(open("data/decay_universe_meta.json"))
    return [
        m for m in expanded + decay
        if m.get("status") == "finalized" and m.get("result") in ("yes", "no")
    ]


def stratified_sample(markets: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Sample diversely across market series."""
    random.seed(seed)
    by_series: dict[str, list[dict]] = defaultdict(list)
    for m in markets:
        by_series[m["ticker"].split("-")[0]].append(m)
    sampled = []
    series_list = list(by_series.keys())
    random.shuffle(series_list)
    while len(sampled) < n and any(by_series.values()):
        for s in series_list:
            if not by_series[s]:
                continue
            sampled.append(by_series[s].pop(random.randrange(len(by_series[s]))))
            if len(sampled) >= n:
                break
    return sampled


SYSTEM_PROMPT = """You are estimating probabilities for binary prediction markets.
Given a question and its resolution rules, output ONLY a JSON object:
{"p_yes": <float 0-1>, "reasoning": "<one sentence>"}
Do not include any other text."""


def estimate_prob(client: OpenAI, model: str, market: dict) -> tuple[float, str]:
    """Ask LLM for P(YES). Returns (prob, reasoning)."""
    title = market.get("title", "")
    subtitle = market.get("subtitle") or market.get("yes_sub_title", "")
    rules = market.get("rules_primary", "")
    user = f"Title: {title}\nSubtitle: {subtitle}\nRules: {rules[:600]}"
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        max_tokens=200,
        temperature=0.0,
    )
    text = r.choices[0].message.content or ""
    # Try JSON parse; fallback to regex
    try:
        # Strip markdown fences if present
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
        data = json.loads(cleaned)
        return float(data["p_yes"]), data.get("reasoning", "")
    except Exception:
        m = re.search(r'"?p_yes"?\s*:\s*([0-9.]+)', text)
        if m:
            return float(m.group(1)), text[:100]
        # Last resort: any 0-1 number
        m = re.search(r'\b(0?\.\d+|1\.0|0)\b', text)
        return (float(m.group(1)) if m else 0.5), text[:100]


def brier_score(preds: list[float], outcomes: list[int]) -> float:
    return sum((p - o) ** 2 for p, o in zip(preds, outcomes)) / len(preds)


def calibration_table(preds: list[float], outcomes: list[int]) -> dict[str, dict]:
    buckets = [(0, 0.1), (0.1, 0.25), (0.25, 0.4), (0.4, 0.6), (0.6, 0.75), (0.75, 0.9), (0.9, 1.01)]
    table = {}
    for lo, hi in buckets:
        ix = [i for i, p in enumerate(preds) if lo <= p < hi]
        if not ix:
            continue
        actual = sum(outcomes[i] for i in ix) / len(ix)
        avg_pred = sum(preds[i] for i in ix) / len(ix)
        table[f"{lo:.2f}-{hi:.2f}"] = {
            "n": len(ix),
            "avg_predicted": round(avg_pred, 3),
            "actual_yes_rate": round(actual, 3),
            "calibration_gap": round(actual - avg_pred, 3),
        }
    return table


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--model", default="gemini-2.0-flash")
    p.add_argument("--output", type=Path, default=Path("reports/llm_signal_validation.json"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    load_dotenv()
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    print(f"=== LLM Signal Validation ===")
    print(f"Model: {args.model}")
    print(f"N: {args.n}\n")

    all_markets = load_resolved_markets()
    print(f"Pool: {len(all_markets)} resolved binary markets")
    sample = stratified_sample(all_markets, args.n, seed=args.seed)
    print(f"Sampled: {len(sample)} markets across series\n")

    results = []
    for i, mkt in enumerate(sample, 1):
        try:
            t0 = time.time()
            prob, reason = estimate_prob(client, args.model, mkt)
            elapsed = time.time() - t0
        except Exception as e:
            print(f"  [{i}/{len(sample)}] ERROR on {mkt['ticker']}: {e}")
            continue
        outcome = 1 if mkt["result"] == "yes" else 0
        # Final market price right before resolution (where available)
        try:
            mkt_price = float(mkt.get("previous_yes_bid_dollars", 0)) or float(mkt.get("last_price_dollars", 0.5))
        except (TypeError, ValueError):
            mkt_price = 0.5
        results.append({
            "ticker": mkt["ticker"],
            "title": mkt["title"][:80],
            "predicted": prob,
            "market_implied": mkt_price,
            "outcome": outcome,
            "result_label": mkt["result"],
            "reasoning": reason[:120],
        })
        flag = "✓" if (prob >= 0.5) == bool(outcome) else "✗"
        print(f"  [{i}/{len(sample)}] {flag} {mkt['ticker'][:45]:<45} pred={prob:.2f} actual={mkt['result']} ({elapsed:.1f}s)")

    if not results:
        raise SystemExit("No results — check API key and model.")

    preds = [r["predicted"] for r in results]
    outcomes = [r["outcome"] for r in results]
    mkt_implied = [r["market_implied"] for r in results]

    brier_llm = brier_score(preds, outcomes)
    brier_mkt = brier_score(mkt_implied, outcomes)
    brier_baseline = brier_score([0.28] * len(outcomes), outcomes)  # fixed base rate

    # Naive PnL: bet $1 on side LLM favors. Win = $1/p - 1; Loss = -$1
    pnl = 0.0
    n_trades = 0
    for r in results:
        side = "yes" if r["predicted"] >= 0.5 else "no"
        # Simulate: pay market price, get $1 if right
        if side == "yes":
            cost = max(0.01, r["market_implied"])
            payout = 1.0 if r["outcome"] == 1 else 0.0
        else:
            cost = max(0.01, 1.0 - r["market_implied"])
            payout = 1.0 if r["outcome"] == 0 else 0.0
        pnl += payout - cost
        n_trades += 1

    print(f"\n=== METRICS ===")
    print(f"Brier (LLM):       {brier_llm:.4f}  (lower = better)")
    print(f"Brier (market):    {brier_mkt:.4f}  ← market-implied baseline")
    print(f"Brier (base 28%):  {brier_baseline:.4f}  ← always-predict-base-rate")
    print(f"LLM beats market:  {'YES' if brier_llm < brier_mkt else 'NO'}")
    print(f"LLM beats baseline: {'YES' if brier_llm < brier_baseline else 'NO'}")

    print(f"\nNaive PnL (bet $1/market on LLM-favored side at market price):")
    print(f"  N trades: {n_trades}, total PnL: ${pnl:+.2f}, per trade: ${pnl/n_trades:+.3f}")

    cal = calibration_table(preds, outcomes)
    print(f"\n=== CALIBRATION ===")
    print(f"{'Bucket':<12} {'N':>4} {'avg_pred':>10} {'actual_yes':>11} {'gap':>8}")
    for b, d in cal.items():
        print(f"  {b:<10} {d['n']:>4} {d['avg_predicted']:>10.3f} {d['actual_yes_rate']:>11.3f} {d['calibration_gap']:>+8.3f}")

    # Save full results for further analysis
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model": args.model,
        "n": len(results),
        "brier_llm": brier_llm,
        "brier_market": brier_mkt,
        "brier_baseline": brier_baseline,
        "naive_pnl_per_trade": pnl / n_trades,
        "calibration": cal,
        "trades": results,
    }, indent=2))
    print(f"\n  → {args.output}")


if __name__ == "__main__":
    main()
