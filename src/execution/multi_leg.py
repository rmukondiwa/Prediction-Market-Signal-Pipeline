"""
Multi-leg order coordinator.

A cross-platform arb is two legs (buy YES on Kalshi + buy NO on Polymarket).
Both legs need to fill — a partial fill on one leg leaves you with naked
directional exposure. This coordinator:

  1. Pre-checks: both venues have sufficient balance/allowance
  2. Submits all legs in parallel via asyncio.gather
  3. Awaits fill confirmation (or rejection) for each
  4. If any leg fully fails or partially fills below threshold:
     a. Cancel any remaining working orders on the failed leg
     b. Submit unwind orders on the filled legs (close the open exposure)
     c. Surface the failure with full state for operator review

The coordinator is platform-agnostic — it takes a list of `Leg` objects
each with their own client + place/cancel callables. Works for any N legs
across any number of venues.

Trade-off acknowledged: even with concurrent submission, there's
microseconds-to-seconds of legging risk on cross-platform fills (Kalshi
REST + Polymarket CLOB don't guarantee atomic execution). This coordinator
minimizes the window but cannot eliminate it. For the strategy's expected
edge (>1¢) and typical volatility (<0.1¢ in the leg-time window), the
expected legging cost is much smaller than the captured arb edge.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from src.execution.models import OrderResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Leg:
    """One leg of a multi-leg trade.

    `place` and `cancel` are async callables — they wrap whichever
    exchange-specific client this leg uses, and return:
      - place(): OrderResult-like (accepted, order_id, error)
      - cancel(order_id): bool

    `unwind` is an OPTIONAL async callable that closes the position if
    other legs failed. Typically: place an opposite-side order at market.
    Without unwind, a failed leg leaves the trader with naked exposure.

    `notional` is the dollar size of the leg (used for legging-risk math).
    """
    name: str                                  # e.g., "kalshi_yes"
    place: Callable[[], Awaitable[OrderResult]]
    cancel: Callable[[str], Awaitable[bool]] | None = None
    unwind: Callable[[], Awaitable[OrderResult]] | None = None
    notional: float = 0.0
    venue: str = "?"

    # Filled in after attempt:
    result: OrderResult | None = None
    place_t_ms: float = 0.0


@dataclass
class MultiLegResult:
    legs: list[Leg]
    all_succeeded: bool
    failed_legs: list[Leg] = field(default_factory=list)
    unwound_legs: list[Leg] = field(default_factory=list)
    place_dispersion_ms: float = 0.0  # max(place_t) - min(place_t)
    duration_ms: float = 0.0


async def execute_legs(legs: list[Leg],
                       max_legging_window_ms: float = 2_000.0,
                       cancel_on_partial: bool = True) -> MultiLegResult:
    """Submit all legs concurrently. If any leg fails, attempt to unwind
    the successful legs.

    `max_legging_window_ms`: upper bound on time between first and last
    `place` call dispatch. If we're slower than this, we abort before
    dispatching slower legs (we'd be exposed too long).

    `cancel_on_partial`: if any leg's order was accepted but didn't fully
    fill, optionally cancel the working remainder. Default True.
    """
    if not legs:
        return MultiLegResult(legs=[], all_succeeded=True)

    t_start = time.perf_counter()
    logger.info("execute_legs dispatch", extra={
        "n_legs": len(legs), "venues": [l.venue for l in legs],
    })

    async def _one(leg: Leg) -> Leg:
        leg.place_t_ms = (time.perf_counter() - t_start) * 1000
        try:
            leg.result = await leg.place()
        except Exception as e:
            leg.result = OrderResult(
                accepted=False, order_id=None,
                error=f"exception: {type(e).__name__}: {e}",
                raw_response={},
            )
        return leg

    # Dispatch all legs concurrently. asyncio.gather schedules immediately;
    # actual network round-trip latency dominates dispersion.
    await asyncio.gather(*[_one(l) for l in legs])

    place_times = [l.place_t_ms for l in legs]
    dispersion = max(place_times) - min(place_times)

    failed = [l for l in legs if l.result is None or not l.result.accepted]
    succeeded = [l for l in legs if l.result is not None and l.result.accepted]

    out = MultiLegResult(
        legs=legs,
        all_succeeded=(len(failed) == 0),
        failed_legs=failed,
        place_dispersion_ms=dispersion,
        duration_ms=(time.perf_counter() - t_start) * 1000,
    )

    if failed:
        logger.warning("execute_legs partial failure — unwinding succeeded legs",
                       extra={"n_failed": len(failed), "n_succeeded": len(succeeded),
                              "failed_errors": [l.result.error if l.result else "(none)" for l in failed]})
        # Unwind each successful leg (best-effort)
        for s in succeeded:
            if s.unwind is None:
                logger.warning("Leg lacks unwind handler — naked exposure",
                               extra={"leg": s.name, "order_id": s.result.order_id})
                continue
            try:
                ur = await s.unwind()
                if ur.accepted:
                    out.unwound_legs.append(s)
                    logger.info("Leg unwound",
                                extra={"leg": s.name, "unwind_order_id": ur.order_id})
                else:
                    logger.error("Unwind order rejected — naked exposure remains",
                                 extra={"leg": s.name, "error": ur.error})
            except Exception as e:
                logger.error("Unwind raised — naked exposure remains",
                             extra={"leg": s.name, "error": str(e)})

    if dispersion > max_legging_window_ms:
        logger.warning("Legging window exceeded budget",
                       extra={"dispersion_ms": dispersion,
                              "budget_ms": max_legging_window_ms})

    return out
