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
