"""Tests for src/execution/multi_leg.py — coordinator that submits N legs
concurrently and unwinds successful legs when any leg fails."""
from __future__ import annotations

import asyncio
import pytest

from src.execution.models import OrderResult
from src.execution.multi_leg import Leg, execute_legs


def _ok(order_id: str = "ok-1") -> OrderResult:
    return OrderResult(accepted=True, order_id=order_id, error=None, raw_response={})


def _fail(error: str = "rejected") -> OrderResult:
    return OrderResult(accepted=False, order_id=None, error=error, raw_response={})


async def _accept():
    return _ok()


async def _reject():
    return _fail()


async def test_all_legs_succeed():
    legs = [
        Leg(name=f"leg{i}", place=_accept, venue="kalshi") for i in range(3)
    ]
    result = await execute_legs(legs)
    assert result.all_succeeded
    assert not result.failed_legs
    assert not result.unwound_legs


async def test_one_leg_fails_and_others_unwound():
    unwound = []

    async def make_unwind(name):
        async def _u():
            unwound.append(name)
            return _ok(order_id=f"unwind-{name}")
        return _u

    leg_a = Leg(name="a", place=_accept, unwind=await make_unwind("a"), venue="kalshi")
    leg_b = Leg(name="b", place=_reject, venue="poly")
    leg_c = Leg(name="c", place=_accept, unwind=await make_unwind("c"), venue="poly")

    result = await execute_legs([leg_a, leg_b, leg_c])
    assert not result.all_succeeded
    assert len(result.failed_legs) == 1
    assert result.failed_legs[0].name == "b"
    assert sorted(unwound) == ["a", "c"]
    assert {l.name for l in result.unwound_legs} == {"a", "c"}


async def test_failed_leg_with_no_unwind_handler_logs_naked():
    # No unwind callable on the successful leg → naked exposure
    leg_a = Leg(name="a", place=_accept, unwind=None, venue="kalshi")
    leg_b = Leg(name="b", place=_reject, venue="poly")
    result = await execute_legs([leg_a, leg_b])
    assert not result.all_succeeded
    assert not result.unwound_legs  # nothing unwound, naked


async def test_dispersion_within_budget():
    # All legs respond instantly → dispersion should be tiny
    legs = [Leg(name=f"l{i}", place=_accept, venue="x") for i in range(3)]
    result = await execute_legs(legs, max_legging_window_ms=500.0)
    assert result.place_dispersion_ms < 100  # well under the 500ms budget
    assert result.all_succeeded


async def test_exception_in_place_treated_as_failure():
    async def _raise():
        raise RuntimeError("simulated network failure")
    leg = Leg(name="bad", place=_raise, venue="x")
    result = await execute_legs([leg])
    assert not result.all_succeeded
    assert "simulated network failure" in result.failed_legs[0].result.error
