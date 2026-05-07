"""
Cross-platform arbitrage scanner v2 — LLM-based question matching.

Replaces the v1 keyword/regex matcher (which found 0 hits because
Kalshi's strike/expiry structure rarely matches Polymarket's verbatim).
The v2 pipeline:

  1. Fetch high-volume Polymarket markets (Gamma API, no auth)
  2. Embed every Polymarket question via Gemini
  3. Pull crypto/financial Kalshi markets (filter by ticker prefix)
  4. Embed Kalshi market questions (same Gemini model, same dim)
  5. Pairwise cosine — for each Polymarket market, top-K Kalshi candidates
  6. For each candidate pair above similarity threshold, ask Gemini:
        "Do these resolve identically?" → yes/no/maybe
  7. For verified pairs, compute arb:
        K_yes_ask + (1 - P_yes_bid) < 1.00  → buy YES on K + buy NO on P
        (1 - K_yes_bid) + P_yes_ask < 1.00  → buy NO on K + buy YES on P

Output candidate arbs to reports/cross_platform_arb_llm.json. Manual review
required before placing trades — this is a candidate generator, not an
auto-execution path.

Usage:
    python -m scripts.scan_cross_platform_arb_llm
    python -m scripts.scan_cross_platform_arb_llm --top-k 5 --threshold 0.80
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

POLY_GAMMA = "https://gamma-api.polymarket.com"
UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

VERIFY_SYSTEM = """You evaluate whether two prediction-market questions resolve to the SAME outcome
in EVERY scenario. This is a strict equivalence check, not a similarity check.

Two questions resolve identically only if ALL of these match exactly:
  1. UNDERLYING — same asset/event (BTC, Trump 2024, etc.)
  2. THRESHOLD — same numeric value if any (must be identical, not "close")
  3. CONDITION TYPE — "ever touched during period" vs "final value at time T"
     are DIFFERENT and never match each other
  4. RESOLUTION TIME — same date AND time, or within minutes
  5. DIRECTION — "above" vs "below" must match

Output JSON only (no markdown):
  {"match": true|false, "confidence": 0.0-1.0, "reason": "<one sentence>"}

Examples that DO match:
  - "Bitcoin above $100k on Dec 31, 2026 at 5pm" ↔ "BTC > $100,000 at 5pm EST Dec 31 2026" → match
Examples that DON'T match (return false):
  - "BTC reaches $100k by Dec 31" (touched anytime) ↔ "BTC > $100k on Dec 31" (final value) → DIFFERENT condition type
  - "BTC > $150k by 2027" ↔ "BTC > $100k by 2027" → DIFFERENT threshold
  - "BTC > $100k on Dec 31 noon" ↔ "BTC > $100k on Dec 31 5pm" → DIFFERENT resolution time

When in doubt, return match=false. False positives in this check cause
losses because the "arb" only holds if BOTH legs settle the same way.
"""


# Polymarket's high-volume universe is polluted with farm/joke markets
# (e.g. "Will LeBron James win the 2028 Democratic presidential nomination?"
# at $50M volume). Filter them out by question keyword. We surface real
# crypto/macro/election markets while skipping the obvious junk.
_POLY_REJECT_PATTERNS = re.compile(
    r"\b(LeBron James|Oprah Winfrey|Chelsea Clinton|George Clooney|"
    r"Kim Kardashian|Phil Murphy|Mike Pence presidential|Byron Donalds|"
    r"Andrew Yang|Tim Walz|Hillary Clinton presidential|Jesus Christ|"
    r"Bernie Sanders presidential)\b",
    re.IGNORECASE,
)


def fetch_polymarket(limit: int, min_volume: float, tag_id: int | None = None) -> list[dict]:
    """Pull active high-volume Polymarket markets.

    Optional tag_id filters to a specific Polymarket category (their tag
    taxonomy is nontrivial — empirically tag_id=21 returns real crypto/
    economic markets, tag_id=2 returns the joke universe).

    Skips markets whose question matches the joke/farm reject list.
    """
    out = []
    offset = 0
    tag_clause = f"&tag_id={tag_id}" if tag_id else ""
    while len(out) < limit:
        url = (f"{POLY_GAMMA}/markets?limit=100&offset={offset}"
               f"&active=true&closed=false&order=volumeNum&ascending=false{tag_clause}")
        req = urllib.request.Request(url, headers=UA)
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                page = json.loads(r.read())
        except Exception as e:
            print(f"poly fetch failed: {e}")
            break
        if not page:
            break
        for m in page:
            try:
                vol = float(m.get("volumeNum", 0) or 0)
            except (ValueError, TypeError):
                vol = 0
            if vol < min_volume:
                continue
            q = m.get("question") or m.get("title") or ""
            if _POLY_REJECT_PATTERNS.search(q):
                continue  # skip farm/joke markets
            out.append(m)
        if len(page) < 100:
            break
        offset += 100
    return out[:limit]


def load_kalshi_catalog(prefixes: tuple[str, ...] | None = None) -> list[dict]:
    """Load the cached Kalshi catalog, optionally filter by ticker prefix."""
    catalog = json.load(open("data/catalog.json"))
    if prefixes:
        catalog = [m for m in catalog if m["ticker"].startswith(prefixes)]
    return catalog


def cosine_topk(query_vec: np.ndarray, corpus: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (top_k_indices, top_k_scores) for one query against the corpus."""
    # Both already L2-normalized; dot product = cosine similarity
    scores = corpus @ query_vec
    idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
    sorted_idx = idx[np.argsort(-scores[idx])]
    return sorted_idx, scores[sorted_idx]


async def gemini_verify(session: aiohttp.ClientSession, api_key: str,
                         poly_q: str, kalshi_q: str,
                         sem: asyncio.Semaphore) -> dict:
    """Ask Gemini if two questions resolve identically."""
    user = f"Q1 (Polymarket): {poly_q}\nQ2 (Kalshi): {kalshi_q}"
    body = {
        "system_instruction": {"parts": [{"text": VERIFY_SYSTEM}]},
        "contents": [{"parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 200},
    }
    backoff = 6.0
    async with sem:
        for attempt in range(5):
            try:
                async with session.post(
                    GEMINI_URL, json=body,
                    headers={"X-goog-api-key": api_key, "Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as r:
                    if r.status == 429:
                        await asyncio.sleep(backoff)
                        backoff *= 1.7
                        continue
                    if r.status >= 400:
                        return {"match": False, "confidence": 0.0,
                                "reason": f"HTTP {r.status}"}
                    data = await r.json()
                    break
            except Exception as e:
                return {"match": False, "confidence": 0.0, "reason": f"error: {e}"}
        else:
            return {"match": False, "confidence": 0.0, "reason": "rate limit retries exhausted"}

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return {"match": False, "confidence": 0.0, "reason": "no candidate text"}
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except Exception:
        return {"match": False, "confidence": 0.0, "reason": text[:100]}


_NUMBER_PATTERN = re.compile(
    r"\$?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(k|K|m|M|thousand|million|billion)?"
)


def extract_numbers_dollars(text: str) -> set[float]:
    """Extract dollar/numeric thresholds from question text.

    Returns a set of normalized dollar amounts. "100k" → 100000.0, "$50" → 50.0,
    etc. Used to deterministically filter out matches where one question says
    $100k and the other says $150k — those are NEVER arbs.
    """
    out: set[float] = set()
    for match in _NUMBER_PATTERN.finditer(text):
        num_s = match.group(1).replace(",", "")
        try:
            n = float(num_s)
        except ValueError:
            continue
        suffix = (match.group(2) or "").lower()
        if suffix in ("k", "thousand"):
            n *= 1_000
        elif suffix in ("m", "million"):
            n *= 1_000_000
        elif suffix == "billion":
            n *= 1_000_000_000
        # Skip year-like numbers (probably dates, not thresholds)
        if 1900 <= n <= 2100:
            continue
        # Skip very small numbers (probably percentages/decimals)
        if n < 10:
            continue
        out.add(round(n, 2))
    return out


def thresholds_compatible(poly_q: str, kalshi_q: str, tol_pct: float = 0.05) -> tuple[bool, str]:
    """Check whether the two questions reference compatible numeric thresholds.

    Returns (compatible, reason). Compatible = the largest threshold in each
    question is within tol_pct of the largest threshold in the other (e.g.
    $99,999.99 vs $100,000 within 0.05%). If either has no extractable
    threshold, defaults to compatible (LLM must judge).
    """
    p_nums = extract_numbers_dollars(poly_q)
    k_nums = extract_numbers_dollars(kalshi_q)
    if not p_nums or not k_nums:
        return True, "no numeric threshold to compare"
    # Use the largest number in each (typically the price threshold)
    p_max = max(p_nums)
    k_max = max(k_nums)
    rel_diff = abs(p_max - k_max) / max(p_max, k_max)
    if rel_diff <= tol_pct:
        return True, f"thresholds match (~{p_max:.0f} vs {k_max:.0f})"
    return False, f"threshold mismatch: poly={p_max:.0f} vs kalshi={k_max:.0f} ({rel_diff*100:.1f}% diff)"


def safe_price(d: dict, k: str, default: float = 0.5) -> float:
    """Safe float extraction for prices possibly stored as strings/None."""
    v = d.get(k)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def compute_arb(kalshi: dict, poly: dict, use_clob: bool = True) -> dict:
    """Given matched markets, compute arb bounds in both directions.

    Kalshi catalog stores yes_ask/yes_bid as integer CENTS (0-100); convert
    to dollars. When degenerate (1¢ ask / 0¢ bid = "no quotes" placeholder),
    fall back to `implied_probability` as zero-spread quote.

    Polymarket prices: prefer live CLOB top-of-book if `use_clob=True`,
    else fall back to AMM mid from Gamma's `outcomePrices` (zero-spread).
    CLOB gives us real spreads and depth — this is a meaningful upgrade
    for cross-platform arb math.
    """
    k_yes_ask_cents = safe_price(kalshi, "yes_ask", 100.0)
    k_yes_bid_cents = safe_price(kalshi, "yes_bid", 0.0)
    k_yes_ask = k_yes_ask_cents / 100.0
    k_yes_bid = k_yes_bid_cents / 100.0
    implied = safe_price(kalshi, "implied_probability", default=0.5)
    if k_yes_ask <= 0.01 and k_yes_bid <= 0.0:
        k_yes_ask = k_yes_bid = implied
    if k_yes_bid >= 0.99 and k_yes_ask >= 1.0:
        k_yes_ask = k_yes_bid = implied

    # Polymarket prices — CLOB-first, AMM fallback
    p_yes_ask = p_yes_bid = p_no_ask = p_no_bid = None
    poly_source = "amm_mid"
    if use_clob:
        try:
            from src.ingestion.polymarket.clob_client import get_market_books
            books = get_market_books(poly)
            if books:
                yes_book, no_book = books
                if yes_book.top_ask and yes_book.top_bid:
                    p_yes_ask = yes_book.top_ask.price
                    p_yes_bid = yes_book.top_bid.price
                if no_book.top_ask and no_book.top_bid:
                    p_no_ask = no_book.top_ask.price
                    p_no_bid = no_book.top_bid.price
                if all(x is not None for x in (p_yes_ask, p_yes_bid, p_no_ask, p_no_bid)):
                    poly_source = "clob"
        except Exception:
            pass

    if poly_source == "amm_mid":
        # Fallback to Gamma's AMM mid (zero-spread approximation)
        poly_prices = poly.get("outcomePrices")
        if isinstance(poly_prices, str):
            try:
                poly_prices = json.loads(poly_prices)
            except Exception:
                poly_prices = None
        p_yes = safe_price({"_": poly_prices[0] if poly_prices else None}, "_", default=0.5)
        p_yes_bid = p_yes_ask = p_yes
        p_no_bid = p_no_ask = 1.0 - p_yes

    # Real arb math:
    #   arb1: buy YES on K (cost K_yes_ask) + buy NO on P (cost P_no_ask) — gets $1 either way
    #   arb2: buy NO on K (cost 1-K_yes_bid) + buy YES on P (cost P_yes_ask)
    arb1_cost = k_yes_ask + p_no_ask
    arb2_cost = (1.0 - k_yes_bid) + p_yes_ask
    return {
        "k_yes_bid_dollars": round(k_yes_bid, 4),
        "k_yes_ask_dollars": round(k_yes_ask, 4),
        "k_implied_prob": round(implied, 4),
        "p_yes_bid": round(p_yes_bid, 4) if p_yes_bid is not None else None,
        "p_yes_ask": round(p_yes_ask, 4) if p_yes_ask is not None else None,
        "p_no_bid": round(p_no_bid, 4) if p_no_bid is not None else None,
        "p_no_ask": round(p_no_ask, 4) if p_no_ask is not None else None,
        "poly_source": poly_source,
        "arb_buy_K_yes_P_no_cost": round(arb1_cost, 4),
        "arb_buy_K_no_P_yes_cost": round(arb2_cost, 4),
        "best_arb_cost": round(min(arb1_cost, arb2_cost), 4),
        "edge": round(1.0 - min(arb1_cost, arb2_cost), 4),
    }


async def main_async(args: argparse.Namespace) -> None:
    load_dotenv()
    api_key = os.environ["OPENAI_API_KEY"]
    base = os.environ["OPENAI_BASE_URL"]
    client = OpenAI(api_key=api_key, base_url=base)

    print(f"=== Cross-platform arb v2 (LLM-matched) ===")
    tag = args.poly_tag if args.poly_tag > 0 else None
    print(f"[1/5] Pulling Polymarket markets (vol≥${args.min_poly_volume}, tag={tag})...")
    poly = fetch_polymarket(args.poly_limit, args.min_poly_volume, tag_id=tag)
    print(f"      Got {len(poly)} markets (after filtering joke/farm questions)")

    print(f"[2/5] Loading Kalshi catalog (filter: {args.kalshi_prefixes})...")
    prefixes = tuple(args.kalshi_prefixes.split(",")) if args.kalshi_prefixes else None
    kalshi = load_kalshi_catalog(prefixes)
    print(f"      Got {len(kalshi)} markets")

    if not poly or not kalshi:
        print("Empty pool — exiting")
        return

    print(f"[3/5] Embedding {len(poly)} Polymarket questions via Gemini...")
    poly_texts = [(m.get("question") or m.get("title") or "")[:300] for m in poly]
    poly_resp = client.embeddings.create(
        model="gemini-embedding-001", input=poly_texts, dimensions=512,
    )
    poly_vecs = np.array([d.embedding for d in poly_resp.data], dtype=np.float32)
    poly_vecs = poly_vecs / np.linalg.norm(poly_vecs, axis=1, keepdims=True).clip(min=1e-9)
    print(f"      {poly_vecs.shape}")

    print(f"[4/5] Embedding {len(kalshi)} Kalshi questions...")
    kal_texts = [(f"{m.get('title','')} -- {m.get('subtitle','')}")[:300] for m in kalshi]
    # Batch in chunks of 100 with throttle for free-tier
    kal_vecs_parts = []
    import time as _time
    for i in range(0, len(kal_texts), 100):
        batch = kal_texts[i : i + 100]
        # Embed batch with retry on 429
        for attempt in range(5):
            try:
                resp = client.embeddings.create(
                    model="gemini-embedding-001", input=batch, dimensions=512,
                )
                break
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    _time.sleep(6 * (1.7 ** attempt))
                    continue
                raise
        else:
            raise RuntimeError("embed retries exhausted")
        kal_vecs_parts.append(np.array([d.embedding for d in resp.data], dtype=np.float32))
        if i % 500 == 0:
            print(f"      embedded {i}/{len(kal_texts)}", flush=True)
        if i + 100 < len(kal_texts):
            _time.sleep(4.0)  # stay under 15 RPM
    kal_vecs = np.vstack(kal_vecs_parts)
    kal_vecs = kal_vecs / np.linalg.norm(kal_vecs, axis=1, keepdims=True).clip(min=1e-9)
    print(f"      {kal_vecs.shape}")

    print(f"[5/5] Finding top-{args.top_k} Kalshi matches per Polymarket question...")
    candidates: list[dict] = []
    for i, p in enumerate(poly):
        idx, scores = cosine_topk(poly_vecs[i], kal_vecs, args.top_k)
        for j, s in zip(idx, scores):
            if s < args.sim_threshold:
                continue
            candidates.append({
                "poly_id": p.get("id"), "poly_question": poly_texts[i],
                "poly_volume": p.get("volumeNum"),
                "poly_market": p,
                "kalshi_ticker": kalshi[j]["ticker"],
                "kalshi_title": kal_texts[j],
                "kalshi_market": kalshi[j],
                "similarity": float(s),
            })
    print(f"      {len(candidates)} candidate pairs above similarity {args.sim_threshold}")

    if not candidates:
        Path("reports").mkdir(parents=True, exist_ok=True)
        Path("reports/cross_platform_arb_llm.json").write_text(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "n_poly": len(poly), "n_kalshi": len(kalshi),
            "n_candidates": 0, "matches": [], "arbs": [],
        }, indent=2))
        print("\nNo candidates above similarity threshold — increase --top-k or lower --sim-threshold")
        return

    # Verify with Gemini chat
    print(f"\n[6/6] Verifying top {min(args.max_verify, len(candidates))} candidates with Gemini chat...")
    candidates.sort(key=lambda c: -c["similarity"])
    to_verify = candidates[: args.max_verify]
    sem = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [
            gemini_verify(session, api_key, c["poly_question"], c["kalshi_title"], sem)
            for c in to_verify
        ]
        verifications = []
        for j, fut in enumerate(asyncio.as_completed(tasks), 1):
            verifications.append(await fut)
            if j % 10 == 0:
                print(f"      {j}/{len(to_verify)} verified", flush=True)

    matches = []
    arbs = []
    n_threshold_filtered = 0
    for c, v in zip(to_verify, verifications):
        if not (v.get("match") and v.get("confidence", 0) >= 0.7):
            continue
        # Deterministic post-filter: even if the LLM said match, reject when
        # the numeric thresholds clearly differ (largest number in each question
        # must agree within 5%). This catches the LLM's tendency to call
        # "BTC > $100k by 2026" and "BTC > $150k by 2026" a match.
        compat, reason = thresholds_compatible(c["poly_question"], c["kalshi_title"])
        if not compat:
            n_threshold_filtered += 1
            continue
        arb = compute_arb(c["kalshi_market"], c["poly_market"])
        # Pull through the minimal Polymarket fields the orchestrator needs
        # to actually trade (clobTokenIds, conditionId). The full poly_market
        # object can be huge, so we keep just the trade-relevant subset.
        pm_full = c.get("poly_market") or {}
        poly_market_slim = {
            "id": pm_full.get("id"),
            "conditionId": pm_full.get("conditionId"),
            "clobTokenIds": pm_full.get("clobTokenIds"),
            "outcomes": pm_full.get("outcomes"),
            "outcomePrices": pm_full.get("outcomePrices"),
            "question": pm_full.get("question"),
            "volumeNum": pm_full.get("volumeNum"),
        }
        # Top-of-book sizes from CLOB so the orchestrator doesn't need a
        # second live fetch.
        try:
            from src.ingestion.polymarket.clob_client import get_market_books
            books = get_market_books(pm_full)
            if books:
                yes_book, no_book = books
                poly_market_slim["yes_top_ask_size"] = yes_book.top_ask.size if yes_book.top_ask else None
                poly_market_slim["no_top_ask_size"] = no_book.top_ask.size if no_book.top_ask else None
        except Exception:
            pass

        row = {
            "poly_id": c["poly_id"],
            "poly_question": c["poly_question"],
            "poly_market": poly_market_slim,
            "kalshi_ticker": c["kalshi_ticker"],
            "kalshi_title": c["kalshi_title"],
            "kalshi_market": c.get("kalshi_market"),
            "similarity": c["similarity"],
            "verification": v,
            "threshold_check": reason,
            "p_yes_ask_size": poly_market_slim.get("yes_top_ask_size"),
            "p_no_ask_size": poly_market_slim.get("no_top_ask_size"),
            **arb,
        }
        matches.append(row)
        if arb["best_arb_cost"] < 0.99:  # near or real arb
            arbs.append(row)
    print(f"  Threshold-mismatch filter dropped {n_threshold_filtered} LLM-claimed matches")

    # Output
    arbs.sort(key=lambda x: x["best_arb_cost"])
    out_path = Path("reports/cross_platform_arb_llm.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(),
        "n_poly": len(poly), "n_kalshi": len(kalshi),
        "n_candidates": len(candidates), "n_verified_match": len(matches),
        "matches": matches, "arbs": arbs,
    }, indent=2, default=str))

    print(f"\n=== RESULTS ===")
    print(f"  Verified matches:        {len(matches)}")
    print(f"  Arb-eligible (cost<$1):  {sum(1 for r in arbs if r['best_arb_cost'] < 1.0)}")
    print(f"  Near-arb (cost<$0.99):   {len(arbs)}")
    if arbs:
        print(f"\nTop 5 by smallest cost:")
        for a in arbs[:5]:
            print(f"  cost=${a['best_arb_cost']:.3f}  edge=${a['edge']:.3f}  K={a['kalshi_ticker'][:30]}")
            print(f"      {a['kalshi_title'][:100]}")
            print(f"      {a['poly_question'][:100]}")
    print(f"\n  Report: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--poly-limit", type=int, default=300)
    p.add_argument("--min-poly-volume", type=float, default=5_000.0)
    p.add_argument("--poly-tag", type=int, default=21,
                   help="Polymarket tag_id to filter (default 21=crypto/macro). "
                        "Set to 0 to fetch all categories.")
    p.add_argument("--kalshi-prefixes", default="KXBTCD,KXETHD,KXSOLD,KXBTC,KXETH,KX2024",
                   help="Comma-separated Kalshi ticker prefixes to filter (default: crypto + 2024 elections)")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--sim-threshold", type=float, default=0.75)
    p.add_argument("--max-verify", type=int, default=200,
                   help="Cap on Gemini chat verifications (cost control)")
    p.add_argument("--concurrency", type=int, default=2)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
