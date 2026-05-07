"""
Train CalibratedLLMSignal end-to-end on resolved-market data.

Workflow:
  1. Load resolved binary markets from expanded + decay universe metadata
  2. Run Gemini on each: title + rules → P(YES) estimate
  3. Save (gemini_prob, outcome) pairs as JSONL for reproducibility
  4. Fit isotonic regression mapping raw → calibrated probability
  5. Compute Brier on raw vs calibrated vs base-rate baseline
  6. Save calibration map to data/calibration_map.pkl

Concurrent Gemini calls via asyncio.Semaphore. Cost on n=400 with
gemini-2.0-flash: ~$0.03 in API spend.

Usage:
    python -m scripts.train_calibration_gemini --n 400
    python -m scripts.train_calibration_gemini --n 100 --crypto-only
    python -m scripts.train_calibration_gemini --resume   # uses cached pairs
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pickle
import random
import re
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)
SYSTEM_PROMPT = """You estimate probabilities for binary prediction markets.
Given a market's title and resolution rules, output ONLY a JSON object:
{"p_yes": <float 0-1>, "reasoning": "<one sentence>"}
Use base rates and any deductive logic from the rules. Don't include any other text."""

CACHE_PATH = Path("data/calibration_pairs.jsonl")
DEFAULT_OUT = Path("data/calibration_map.pkl")


def load_resolved_markets(crypto_only: bool = False) -> list[dict]:
    expanded = json.load(open("data/expanded_universe_meta.json"))
    decay = json.load(open("data/decay_universe_meta.json"))
    rows = [m for m in expanded + decay
            if m.get("status") == "finalized" and m.get("result") in ("yes", "no")]
    if crypto_only:
        crypto_prefixes = ("KXBTC", "KXETH", "KXSOL", "KXDOGE", "KXXRP",
                           "KXHYPE", "KXBNB")
        rows = [m for m in rows if m["ticker"].startswith(crypto_prefixes)]
    return rows


def stratified_sample(rows: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Diverse across series."""
    from collections import defaultdict
    random.seed(seed)
    by_series: dict[str, list[dict]] = defaultdict(list)
    for m in rows:
        by_series[m["ticker"].split("-")[0]].append(m)
    sampled: list[dict] = []
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


async def gemini_estimate(session: aiohttp.ClientSession, model: str,
                           api_key: str, market: dict,
                           sem: asyncio.Semaphore) -> dict | None:
    title = market.get("title", "")
    subtitle = market.get("subtitle") or market.get("yes_sub_title", "")
    rules = market.get("rules_primary", "")[:600]
    user = f"Title: {title}\nSubtitle: {subtitle}\nRules: {rules}"
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 200},
    }
    url = GEMINI_URL.format(model=model)
    # Retry on 429 with exponential backoff (free tier ~15 RPM, so worth waiting)
    backoff = 6.0
    async with sem:
        for attempt in range(5):
            try:
                async with session.post(
                    url, json=body,
                    headers={"X-goog-api-key": api_key, "Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as r:
                    if r.status == 429:
                        await asyncio.sleep(backoff)
                        backoff *= 1.7
                        continue
                    if r.status >= 400:
                        return {"ticker": market["ticker"], "error": f"HTTP {r.status}"}
                    data = await r.json()
                    break
            except Exception as e:
                return {"ticker": market["ticker"], "error": str(e)}
        else:
            return {"ticker": market["ticker"], "error": "HTTP 429 (exhausted retries)"}

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return {"ticker": market["ticker"], "error": "no candidate text"}

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    p_yes = None
    try:
        parsed = json.loads(cleaned)
        p_yes = float(parsed.get("p_yes"))
    except Exception:
        m = re.search(r'"?p_yes"?\s*:\s*([0-9.]+)', text)
        if m:
            p_yes = float(m.group(1))
    if p_yes is None or not (0 <= p_yes <= 1):
        return {"ticker": market["ticker"], "error": "no parseable p_yes",
                "raw": text[:200]}

    return {
        "ticker": market["ticker"],
        "title": title[:80],
        "predicted": p_yes,
        "outcome": 1 if market["result"] == "yes" else 0,
    }


async def gather_predictions(markets: list[dict], model: str, api_key: str,
                              concurrency: int = 8) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [gemini_estimate(session, model, api_key, m, sem) for m in markets]
        results: list[dict] = []
        for i, fut in enumerate(asyncio.as_completed(tasks), 1):
            r = await fut
            if r:
                results.append(r)
            if i % 25 == 0:
                ok = sum(1 for x in results if "predicted" in x)
                print(f"  {i}/{len(markets)} done ({ok} valid)", flush=True)
        return results


def fit_calibration(pairs: list[dict], out_path: Path) -> dict:
    """Fit isotonic regression and report metrics."""
    from sklearn.isotonic import IsotonicRegression

    valid = [p for p in pairs if "predicted" in p]
    raws = [p["predicted"] for p in valid]
    outcomes = [p["outcome"] for p in valid]

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raws, outcomes)
    calibrated = iso.predict(raws)

    base_rate = sum(outcomes) / len(outcomes)

    def brier(preds: list[float], obs: list[int]) -> float:
        return sum((p - o) ** 2 for p, o in zip(preds, obs)) / len(preds)

    metrics = {
        "n": len(valid),
        "base_rate": base_rate,
        "brier_raw": brier(raws, outcomes),
        "brier_calibrated": brier(list(calibrated), outcomes),
        "brier_baseline": brier([base_rate] * len(outcomes), outcomes),
    }
    metrics["calibration_lift"] = metrics["brier_baseline"] - metrics["brier_calibrated"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(iso, f)
    return metrics


def calibration_table(pairs: list[dict], iso=None) -> dict[str, dict]:
    from collections import defaultdict
    valid = [p for p in pairs if "predicted" in p]
    if not valid:
        return {}
    buckets = [(0, 0.1), (0.1, 0.25), (0.25, 0.4), (0.4, 0.6),
               (0.6, 0.75), (0.75, 0.9), (0.9, 1.01)]
    out = {}
    for lo, hi in buckets:
        ix = [p for p in valid if lo <= p["predicted"] < hi]
        if not ix:
            continue
        n = len(ix)
        avg = sum(p["predicted"] for p in ix) / n
        actual = sum(p["outcome"] for p in ix) / n
        row = {"n": n, "avg_pred_raw": round(avg, 3),
               "actual_yes": round(actual, 3),
               "gap_raw": round(actual - avg, 3)}
        if iso is not None:
            cal_avg = float(iso.predict([avg])[0])
            row["avg_pred_calibrated"] = round(cal_avg, 3)
            row["gap_calibrated"] = round(actual - cal_avg, 3)
        out[f"{lo:.2f}-{hi:.2f}"] = row
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--model", default="gemini-2.0-flash")
    p.add_argument("--crypto-only", action="store_true")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--resume", action="store_true",
                   help="Skip Gemini calls; refit using cached calibration_pairs.jsonl")
    args = p.parse_args()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY in .env")

    if args.resume and CACHE_PATH.exists():
        print(f"Resuming from {CACHE_PATH}")
        pairs = [json.loads(l) for l in CACHE_PATH.read_text().splitlines() if l.strip()]
    else:
        print(f"=== Calibration training ===")
        print(f"Model:        {args.model}")
        print(f"Crypto only:  {args.crypto_only}")
        all_markets = load_resolved_markets(crypto_only=args.crypto_only)
        print(f"Pool:         {len(all_markets)} resolved markets")
        sample = stratified_sample(all_markets, args.n)
        print(f"Sampling:     {len(sample)} markets across series")
        print(f"Concurrency:  {args.concurrency}")
        print()

        pairs = asyncio.run(gather_predictions(
            sample, args.model, api_key, concurrency=args.concurrency
        ))
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CACHE_PATH.open("w") as f:
            for p_ in pairs:
                f.write(json.dumps(p_, default=str) + "\n")
        print(f"  → cached {len(pairs)} pairs to {CACHE_PATH}")

    valid = [p for p in pairs if "predicted" in p]
    errors = [p for p in pairs if "predicted" not in p]
    print(f"\n=== Results ===")
    print(f"Valid pairs:  {len(valid)}")
    print(f"Errors:       {len(errors)}")

    if not valid:
        raise SystemExit("No valid pairs — check API key and model")

    metrics = fit_calibration(pairs, args.output)
    print(f"\nBrier scores (lower = better):")
    print(f"  Raw LLM:       {metrics['brier_raw']:.4f}")
    print(f"  Calibrated:    {metrics['brier_calibrated']:.4f}")
    print(f"  Base rate ({metrics['base_rate']:.0%}): {metrics['brier_baseline']:.4f}")
    print(f"  Calibration lift over baseline: {metrics['calibration_lift']:+.4f}")

    if metrics["brier_calibrated"] < metrics["brier_baseline"] - 0.005:
        print(f"  ✅ Calibrated signal beats baseline meaningfully — has edge")
    elif metrics["brier_calibrated"] < metrics["brier_baseline"]:
        print(f"  ⚠️  Marginal improvement; signal weak")
    else:
        print(f"  ❌ No improvement over baseline; signal has no edge")

    # Calibration table
    import pickle as _pkl
    iso = _pkl.load(args.output.open("rb"))
    table = calibration_table(pairs, iso)
    print(f"\nCalibration table (raw → calibrated → actual):")
    print(f"  {'bucket':<12} {'n':>4} {'raw':>8} {'cal':>8} {'actual':>8} {'gap_cal':>8}")
    for b, d in table.items():
        print(f"  {b:<12} {d['n']:>4} {d['avg_pred_raw']:>8.3f} "
              f"{d.get('avg_pred_calibrated', float('nan')):>8.3f} "
              f"{d['actual_yes']:>8.3f} {d.get('gap_calibrated', float('nan')):>+8.3f}")

    print(f"\n  → calibration map: {args.output}")


if __name__ == "__main__":
    main()
