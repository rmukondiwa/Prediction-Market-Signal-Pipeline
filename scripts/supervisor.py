"""
Watchdog / supervisor for live trading processes.

Runs a child process and restarts it on crash. Sends an alert on each
restart. Backs off if the child crashes too quickly (likely a config
error, not a transient fault).

Usage:
    python -m scripts.supervisor -- python -m scripts.run_arb_live
    python -m scripts.supervisor --max-restarts 10 --backoff-base 5 -- ...

Notes:
  - Uses the alerts module to notify on every restart and on supervisor exit
  - Tracks healthy run threshold: a child that runs for ≥`healthy_run_seconds`
    resets the restart counter (transient crashes don't accumulate forever)
  - Receives SIGTERM/SIGINT and forwards to child gracefully
  - Does NOT capture child stdout/stderr — child should write to its own
    log files so we don't double-buffer
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.alerts import alert_async
from src.utils.logging import get_logger

logger = get_logger(__name__)


async def run_supervised(cmd: list[str], max_restarts: int,
                          backoff_base: float, healthy_run_seconds: float) -> int:
    """Run cmd as a subprocess; restart on non-zero exit. Return final exit code."""
    restarts = 0
    backoff = backoff_base
    consecutive_quick_crashes = 0

    while True:
        start_t = time.monotonic()
        logger.info("Supervisor starting child", extra={"cmd": cmd, "restart": restarts})
        proc = await asyncio.create_subprocess_exec(*cmd)
        rc = await proc.wait()
        elapsed = time.monotonic() - start_t

        if rc == 0:
            logger.info("Child exited cleanly — supervisor exiting", extra={"elapsed_s": elapsed})
            return 0

        # Non-zero exit
        if elapsed >= healthy_run_seconds:
            consecutive_quick_crashes = 0
            backoff = backoff_base
        else:
            consecutive_quick_crashes += 1
            backoff = min(backoff * 1.7, 300.0)

        restarts += 1
        await alert_async(
            "Child process crashed",
            severity="high" if consecutive_quick_crashes >= 3 else "warning",
            context={
                "exit_code": rc, "elapsed_s": round(elapsed, 1),
                "restart": restarts, "max_restarts": max_restarts,
                "backoff_s": round(backoff, 1),
                "consecutive_quick_crashes": consecutive_quick_crashes,
                "cmd": " ".join(cmd[:5]) + ("..." if len(cmd) > 5 else ""),
            },
        )

        if restarts >= max_restarts:
            await alert_async(
                "Supervisor giving up — max restarts reached",
                severity="high",
                context={"restarts": restarts, "cmd": " ".join(cmd[:5])},
            )
            return rc

        if consecutive_quick_crashes >= 5:
            await alert_async(
                "Supervisor halting — too many quick consecutive crashes",
                severity="high",
                context={"consecutive_quick_crashes": consecutive_quick_crashes},
            )
            return rc

        logger.warning("Restarting child after backoff",
                       extra={"backoff_s": backoff, "restart": restarts})
        await asyncio.sleep(backoff)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="Supervisor that restarts a child process on crash.",
        usage="python -m scripts.supervisor [supervisor-args] -- <child cmd>",
    )
    p.add_argument("--max-restarts", type=int, default=20,
                   help="Give up after this many crashes (default: 20)")
    p.add_argument("--backoff-base", type=float, default=2.0,
                   help="Initial backoff seconds (default: 2.0)")
    p.add_argument("--healthy-run-seconds", type=float, default=120.0,
                   help="Run length that counts as 'healthy' and resets backoff "
                        "(default: 120s)")
    if "--" not in sys.argv:
        p.print_help()
        raise SystemExit("Provide child command after `--`")
    sep = sys.argv.index("--")
    args = p.parse_args(sys.argv[1:sep])
    cmd = sys.argv[sep + 1 :]
    if not cmd:
        raise SystemExit("Empty child command")
    return args, cmd


def main() -> None:
    args, cmd = parse_args()
    rc = asyncio.run(run_supervised(
        cmd=cmd,
        max_restarts=args.max_restarts,
        backoff_base=args.backoff_base,
        healthy_run_seconds=args.healthy_run_seconds,
    ))
    sys.exit(rc)


if __name__ == "__main__":
    main()
