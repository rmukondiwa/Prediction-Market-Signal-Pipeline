"""
Scanner orchestrator — runs structural-arb + cross-platform-arb scanners
on a schedule, logs every hit (or absence) to logs/scanner_hits.jsonl.

The two scanners (`scan_alpha.py` and `scan_cross_platform_arb.py`) are
deterministic snapshots of the live order book and the Polymarket /
Kalshi cross-listing universe. They've returned zero hits in production
runs to date, but cheap to keep watching: if a market-maker outage or a
news-driven dislocation creates a brief mispricing, this catches it.

Scheduled cadence: 30 min for alpha (cheap, no API), 60 min for
cross-platform (Polymarket API calls).

Output: one JSON line per scan to `logs/scanner_hits.jsonl`. Format:
    {ts, scanner, n_riskless, n_kinks, n_pairs, top_hit, ...}

Usage:
    python -m scripts.run_scanners_loop &
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import subprocess
from pathlib import Path

LOG = Path("logs/scanner_hits.jsonl")

ALPHA_INTERVAL_S = 30 * 60      # 30 min
XPLATFORM_INTERVAL_S = 60 * 60  # 60 min


def append_log(record: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def run_alpha_scanner() -> dict:
    """Run scan_alpha.py and parse its JSON output."""
    rpt = Path("reports/alpha.json")
    proc = subprocess.run(
        [".venv/bin/python", "-m", "scripts.scan_alpha", "--output", str(rpt)],
        capture_output=True, timeout=120,
    )
    record = {
        "ts": dt.datetime.utcnow().isoformat(),
        "scanner": "scan_alpha",
        "exit_code": proc.returncode,
    }
    if rpt.exists():
        try:
            data = json.loads(rpt.read_text())
            record.update({
                "n_riskless_monotone": len(data.get("monotonicity_violations", [])),
                "n_riskless_partition": len(data.get("partition_violations", [])),
                "n_soft_kinks": len(data.get("soft_kinks", [])),
                "top_kink": (data.get("soft_kinks") or [{}])[0].get("event_ticker"),
            })
        except Exception as e:
            record["parse_error"] = str(e)
    if proc.returncode != 0:
        record["stderr_tail"] = proc.stderr.decode()[-500:]
    return record


def run_xplatform_scanner() -> dict:
    """Run scan_cross_platform_arb.py and parse its JSON output."""
    rpt = Path("reports/cross_platform_arb.json")
    proc = subprocess.run(
        [".venv/bin/python", "-m", "scripts.scan_cross_platform_arb",
         "--output", str(rpt)],
        capture_output=True, timeout=180,
    )
    record = {
        "ts": dt.datetime.utcnow().isoformat(),
        "scanner": "scan_cross_platform_arb",
        "exit_code": proc.returncode,
    }
    if rpt.exists():
        try:
            data = json.loads(rpt.read_text())
            record.update({
                "n_arbs": len(data.get("arbs", [])),
                "n_near_arbs": len(data.get("near_arbs", [])),
                "n_pairs_compared": data.get("pairs_compared", 0),
            })
        except Exception as e:
            record["parse_error"] = str(e)
    if proc.returncode != 0:
        record["stderr_tail"] = proc.stderr.decode()[-500:]
    return record


async def alpha_loop() -> None:
    while True:
        try:
            rec = await asyncio.to_thread(run_alpha_scanner)
            append_log(rec)
            n_hits = (rec.get("n_riskless_monotone", 0) +
                      rec.get("n_riskless_partition", 0))
            if n_hits > 0:
                # Genuine arb found — write a loud separate alert line
                append_log({**rec, "ALERT": "RISKLESS_ARB_FOUND"})
        except Exception as e:
            append_log({"ts": dt.datetime.utcnow().isoformat(),
                        "scanner": "scan_alpha", "error": str(e)})
        await asyncio.sleep(ALPHA_INTERVAL_S)


async def xplatform_loop() -> None:
    while True:
        try:
            rec = await asyncio.to_thread(run_xplatform_scanner)
            append_log(rec)
            if rec.get("n_arbs", 0) > 0:
                append_log({**rec, "ALERT": "CROSS_PLATFORM_ARB_FOUND"})
        except Exception as e:
            append_log({"ts": dt.datetime.utcnow().isoformat(),
                        "scanner": "scan_cross_platform_arb", "error": str(e)})
        await asyncio.sleep(XPLATFORM_INTERVAL_S)


async def main() -> None:
    append_log({
        "ts": dt.datetime.utcnow().isoformat(),
        "event": "scanner_loop_started",
        "alpha_interval_s": ALPHA_INTERVAL_S,
        "xplatform_interval_s": XPLATFORM_INTERVAL_S,
    })
    await asyncio.gather(alpha_loop(), xplatform_loop())


if __name__ == "__main__":
    asyncio.run(main())
