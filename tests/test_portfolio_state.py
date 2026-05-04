from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.portfolio.models import Fill, WorkingOrder
from src.portfolio.state import InMemoryBackend, PortfolioState


@pytest.fixture
def state():
    backend = InMemoryBackend()
    return PortfolioState(backend, env="test")


async def test_initialize_sets_cash(state):
    await state.initialize(10_000.0)
    assert await state.get_cash() == 10_000.0
    assert await state.get_realized_pnl() == 0.0


async def test_apply_fill_debits_cash_and_opens_position(state):
    await state.initialize(1_000.0)
    fill = Fill(
        fill_id="f1", order_id=None, ticker="X-1", side="yes",
        contracts=10, price=0.30, fee=0.50,
        timestamp=datetime.now(timezone.utc),
        signal_model="raw_llm",
    )
    await state.apply_fill(fill)
    # cash debited by 10*0.30 + 0.50 = 3.50
    assert abs(await state.get_cash() - 996.50) < 1e-6
    pos = await state.get_position("X-1")
    assert pos is not None
    assert pos.contracts == 10
    assert pos.avg_cost == 0.30


async def test_apply_two_fills_same_side_averages_cost(state):
    await state.initialize(1_000.0)
    base = datetime.now(timezone.utc)
    await state.apply_fill(Fill(fill_id="f1", order_id=None, ticker="X-1", side="yes",
                                contracts=10, price=0.30, fee=0.0, timestamp=base, signal_model="m1"))
    await state.apply_fill(Fill(fill_id="f2", order_id=None, ticker="X-1", side="yes",
                                contracts=10, price=0.40, fee=0.0, timestamp=base, signal_model="m1"))
    pos = await state.get_position("X-1")
    assert pos.contracts == 20
    assert abs(pos.avg_cost - 0.35) < 1e-9


async def test_apply_opposite_side_closes_position(state):
    await state.initialize(1_000.0)
    base = datetime.now(timezone.utc)
    await state.apply_fill(Fill(fill_id="f1", order_id=None, ticker="X-1", side="yes",
                                contracts=10, price=0.30, fee=0.0, timestamp=base, signal_model="m1"))
    # Now buy the no side, which is equivalent to selling yes at (1 - no_price)
    # Buy 10 NO at 0.70 → equivalent to selling 10 YES at 0.30 → realized P&L = 0
    await state.apply_fill(Fill(fill_id="f2", order_id=None, ticker="X-1", side="no",
                                contracts=10, price=0.70, fee=0.0, timestamp=base, signal_model="m2"))
    # Position should be closed, realized P&L should be ~0 (closed at same equiv price)
    assert await state.get_position("X-1") is None
    pnl = await state.get_realized_pnl()
    assert abs(pnl) < 1e-6


async def test_opposite_side_partial_close_keeps_remainder(state):
    """Existing 50 YES, fill 30 NO → close 30 YES, 20 YES remain. No leftover."""
    await state.initialize(1_000.0)
    base = datetime.now(timezone.utc)
    await state.apply_fill(Fill(fill_id="f1", order_id=None, ticker="X-1", side="yes",
                                contracts=50, price=0.30, fee=0.0, timestamp=base, signal_model="m1"))
    await state.apply_fill(Fill(fill_id="f2", order_id=None, ticker="X-1", side="no",
                                contracts=30, price=0.70, fee=0.0, timestamp=base, signal_model="m2"))
    pos = await state.get_position("X-1")
    assert pos is not None
    assert pos.side == "yes"
    assert pos.contracts == 20  # 50 - 30 remaining


async def test_opposite_side_fully_closes_with_leftover_opens_new_position(state):
    """REGRESSION TEST for state.py bug surfaced 2026-05-01:
    Existing 50 YES, fill 100 NO → close 50 YES, OPEN 50 NO with leftover.
    Pre-fix: leftover 50 NO contracts were silently dropped."""
    await state.initialize(10_000.0)
    base = datetime.now(timezone.utc)
    await state.apply_fill(Fill(fill_id="f1", order_id=None, ticker="X-1", side="yes",
                                contracts=50, price=0.30, fee=0.0, timestamp=base, signal_model="m1"))
    # 100 NO @ $0.70 → closes 50 YES (realized PnL: 50 × (0.30 - 0.30) = 0),
    # leftover 50 NO @ $0.70 should open a new NO position
    await state.apply_fill(Fill(fill_id="f2", order_id=None, ticker="X-1", side="no",
                                contracts=100, price=0.70, fee=0.0, timestamp=base, signal_model="m2"))
    pos = await state.get_position("X-1")
    assert pos is not None, "leftover should have opened a new NO position (bug regression)"
    assert pos.side == "no"
    assert pos.contracts == 50
    assert pos.avg_cost == 0.70


async def test_opposite_side_fully_closes_with_leftover_realized_pnl_correct(state):
    """Verify realized PnL when leftover opens a new position matches the math."""
    await state.initialize(10_000.0)
    base = datetime.now(timezone.utc)
    # Open YES at 0.30
    await state.apply_fill(Fill(fill_id="f1", order_id=None, ticker="X-1", side="yes",
                                contracts=50, price=0.30, fee=0.0, timestamp=base, signal_model="m1"))
    # Buy 100 NO at 0.99 — closes 50 YES at equivalent sell of 0.01 (= 1 - 0.99)
    # Realized loss on closed portion: 50 × (0.01 - 0.30) = -14.50
    await state.apply_fill(Fill(fill_id="f2", order_id=None, ticker="X-1", side="no",
                                contracts=100, price=0.99, fee=0.0, timestamp=base, signal_model="m2"))
    pnl = await state.get_realized_pnl()
    assert abs(pnl - (-14.50)) < 1e-6, f"expected -$14.50, got ${pnl:.2f}"
    # And the new 50 NO position should be there
    pos = await state.get_position("X-1")
    assert pos.side == "no" and pos.contracts == 50 and abs(pos.avg_cost - 0.99) < 1e-9


async def test_doge_06_15_scenario_replays_correctly(state):
    """Reproduces the exact yesterday's DOGE 06:15 sequence to lock in the fix.
    7 fills: 1 YES at $0.775, then 4+4+100+100+100+100 NO at $0.975-$0.99."""
    await state.initialize(10_000.0)
    base = datetime.now(timezone.utc)
    fills = [
        ("yes", 50, 0.775, 0.61),
        ("no", 4, 0.975, 0.0068),
        ("no", 4, 0.975, 0.0068),
        ("no", 100, 0.99, 0.0693),
        ("no", 100, 0.99, 0.0693),
        ("no", 100, 0.99, 0.0693),
        ("no", 100, 0.99, 0.0693),
    ]
    for i, (side, ct, price, fee) in enumerate(fills):
        await state.apply_fill(Fill(
            fill_id=f"f{i}", order_id=None, ticker="DOGE", side=side,
            contracts=ct, price=price, fee=fee, timestamp=base, signal_model="test",
        ))
    # After all fills, position should be 350 NO (50 YES closed against the
    # NO ladder, leftover 8 + 100 + 100 + 100 + 100 - 4 - 4 closing = 350-ish).
    # Pre-fix this was 300 (lost 58 from fill 4's leftover after YES fully closed).
    pos = await state.get_position("DOGE")
    assert pos is not None
    assert pos.side == "no"
    # Pre-fix: 300 ct (BUG). Post-fix: should be MORE than 300.
    assert pos.contracts > 300, (
        f"Post-fix should retain more than 300 NO contracts; got {pos.contracts}. "
        f"If 300, the state.py leftover-drop bug regressed."
    )


async def test_settle_yes_position_winning(state):
    await state.initialize(1_000.0)
    await state.apply_fill(Fill(
        fill_id="f1", order_id=None, ticker="X-1", side="yes",
        contracts=10, price=0.30, fee=0.0,
        timestamp=datetime.now(timezone.utc), signal_model="raw_llm",
    ))
    pnl = await state.settle("X-1", settlement_value=1.0, t=datetime.now(timezone.utc))
    # Bought at 0.30, settled at 1.0 → P&L = 10*(1.0 - 0.30) = 7.0
    assert abs(pnl - 7.0) < 1e-9
    # Position closed, cash credited by 10*1.0 = 10
    assert await state.get_position("X-1") is None
    assert abs(await state.get_cash() - (1_000.0 - 3.0 + 10.0)) < 1e-6


async def test_settle_no_position_winning(state):
    await state.initialize(1_000.0)
    await state.apply_fill(Fill(
        fill_id="f1", order_id=None, ticker="X-1", side="no",
        contracts=10, price=0.40, fee=0.0,
        timestamp=datetime.now(timezone.utc), signal_model="raw_llm",
    ))
    pnl = await state.settle("X-1", settlement_value=0.0, t=datetime.now(timezone.utc))
    # Bought NO at 0.40, settled NO (= yes resolved no) → P&L = 10*(1.0 - 0.40) = 6.0
    assert abs(pnl - 6.0) < 1e-9


async def test_working_order_lifecycle(state):
    await state.initialize(1_000.0)
    order = WorkingOrder(
        order_id="o1", ticker="X-1", side="yes",
        contracts=10, limit_price=0.25,
        placed_at=datetime.now(timezone.utc),
    )
    await state.add_working_order(order)
    orders = await state.list_working_orders()
    assert len(orders) == 1
    assert orders[0].order_id == "o1"
    await state.cancel_order("o1")
    assert await state.list_working_orders() == []


async def test_snapshot_aggregates_state(state):
    await state.initialize(1_000.0)
    base = datetime.now(timezone.utc)
    await state.apply_fill(Fill(fill_id="f1", order_id=None, ticker="A", side="yes",
                                contracts=10, price=0.30, fee=0.0, timestamp=base, signal_model="m1"))
    snap = await state.snapshot(current_prices={"A": 0.50})
    assert snap.cash == 997.0
    assert len(snap.positions) == 1
    # unrealized = 10 * (0.50 - 0.30) = 2.0
    assert abs(snap.unrealized_pnl - 2.0) < 1e-9


async def test_namespace_isolation_between_envs():
    backend = InMemoryBackend()
    paper = PortfolioState(backend, env="paper")
    bt = PortfolioState(backend, env="bt:run1")
    await paper.initialize(1_000.0)
    await bt.initialize(2_000.0)
    assert await paper.get_cash() == 1_000.0
    assert await bt.get_cash() == 2_000.0
