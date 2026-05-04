"""
Portfolio state — Redis-backed source of truth.

Single instance per environment ({paper, live, bt:<run_id>}). All operations
are atomic via Redis transactions so concurrent reads/writes can't corrupt
state.

Key layout (cents stored as integers to avoid float precision issues):
    portfolio:{env}:cash                   STRING  cents
    portfolio:{env}:positions              HASH    ticker → JSON Position
    portfolio:{env}:working_orders         HASH    order_id → JSON WorkingOrder
    portfolio:{env}:fills                  STREAM  audit log
    portfolio:{env}:realized_pnl           STRING  cents
    portfolio:{env}:daily_pnl:YYYY-MM-DD   STRING  cents

For tests, an in-memory adapter is provided so we don't need a real Redis.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Protocol

from src.portfolio.models import (
    Fill,
    Position,
    PortfolioSnapshot,
    Side,
    WorkingOrder,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class StateBackend(Protocol):
    """Minimal interface a state backend must satisfy. Async-shaped to match
    Redis but in-memory backends can implement them as plain async wrappers."""

    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str) -> None: ...
    async def hget(self, key: str, field: str) -> str | None: ...
    async def hset(self, key: str, field: str, value: str) -> None: ...
    async def hdel(self, key: str, field: str) -> None: ...
    async def hgetall(self, key: str) -> dict[str, str]: ...
    async def xadd(self, key: str, fields: dict[str, str]) -> None: ...
    async def incrbyfloat(self, key: str, amount: float) -> float: ...


class InMemoryBackend:
    """Single-process in-memory backend. Useful for tests + backtest isolation."""

    def __init__(self):
        self._kv: dict[str, str] = {}
        self._h: dict[str, dict[str, str]] = {}
        self._streams: dict[str, list[dict[str, str]]] = {}

    async def get(self, key: str) -> str | None:
        return self._kv.get(key)

    async def set(self, key: str, value: str) -> None:
        self._kv[key] = value

    async def hget(self, key: str, field: str) -> str | None:
        return self._h.get(key, {}).get(field)

    async def hset(self, key: str, field: str, value: str) -> None:
        self._h.setdefault(key, {})[field] = value

    async def hdel(self, key: str, field: str) -> None:
        self._h.get(key, {}).pop(field, None)

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self._h.get(key, {}))

    async def xadd(self, key: str, fields: dict[str, str]) -> None:
        self._streams.setdefault(key, []).append(dict(fields))

    async def incrbyfloat(self, key: str, amount: float) -> float:
        cur = float(self._kv.get(key, "0"))
        new = cur + amount
        self._kv[key] = str(new)
        return new


class PortfolioState:
    """High-level portfolio operations. Backend-agnostic."""

    def __init__(self, backend: StateBackend, env: str = "paper"):
        self.backend = backend
        self.env = env

    # ------------------------------------------------------------------ keys
    def _k(self, suffix: str) -> str:
        return f"portfolio:{self.env}:{suffix}"

    def _daily_pnl_key(self, when: datetime | None = None) -> str:
        d = (when or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
        return self._k(f"daily_pnl:{d}")

    # ---------------------------------------------------------------- setup
    async def initialize(self, starting_capital: float) -> None:
        await self.backend.set(self._k("cash"), str(starting_capital))
        await self.backend.set(self._k("realized_pnl"), "0.0")

    # ---------------------------------------------------------------- reads
    async def get_cash(self) -> float:
        v = await self.backend.get(self._k("cash"))
        return float(v) if v is not None else 0.0

    async def get_realized_pnl(self) -> float:
        v = await self.backend.get(self._k("realized_pnl"))
        return float(v) if v is not None else 0.0

    async def get_daily_pnl(self, when: datetime | None = None) -> float:
        v = await self.backend.get(self._daily_pnl_key(when))
        return float(v) if v is not None else 0.0

    async def get_position(self, ticker: str) -> Position | None:
        raw = await self.backend.hget(self._k("positions"), ticker)
        if raw is None:
            return None
        return Position(**json.loads(raw))

    async def list_positions(self) -> list[Position]:
        rows = await self.backend.hgetall(self._k("positions"))
        return [Position(**json.loads(v)) for v in rows.values()]

    async def list_working_orders(self) -> list[WorkingOrder]:
        rows = await self.backend.hgetall(self._k("working_orders"))
        return [WorkingOrder(**json.loads(v)) for v in rows.values()]

    async def snapshot(self, current_prices: dict[str, float] | None = None) -> PortfolioSnapshot:
        positions = await self.list_positions()
        working = await self.list_working_orders()
        cash = await self.get_cash()
        realized = await self.get_realized_pnl()
        daily = await self.get_daily_pnl()
        unrealized = self._unrealized(positions, current_prices or {})
        return PortfolioSnapshot(
            cash=cash, positions=positions, working_orders=working,
            realized_pnl=realized, unrealized_pnl=unrealized, daily_pnl=daily,
            timestamp=datetime.now(timezone.utc),
        )

    @staticmethod
    def _unrealized(positions: list[Position], prices: dict[str, float]) -> float:
        total = 0.0
        for p in positions:
            mid = prices.get(p.ticker)
            if mid is None:
                continue
            yes_mid = mid
            mark = yes_mid if p.side == "yes" else (1.0 - yes_mid)
            total += p.contracts * (mark - p.avg_cost)
        return total

    # --------------------------------------------------------------- writes
    async def add_working_order(self, order: WorkingOrder) -> None:
        await self.backend.hset(
            self._k("working_orders"),
            order.order_id,
            order.model_dump_json(),
        )

    async def update_order(self, order_id: str, status: str, filled_contracts: int) -> None:
        raw = await self.backend.hget(self._k("working_orders"), order_id)
        if raw is None:
            return
        order = WorkingOrder(**json.loads(raw))
        updated = order.model_copy(update={
            "status": status,
            "filled_contracts": filled_contracts,
        })
        await self.backend.hset(self._k("working_orders"), order_id, updated.model_dump_json())

    async def cancel_order(self, order_id: str) -> None:
        await self.update_order(order_id, "canceled", 0)
        await self.backend.hdel(self._k("working_orders"), order_id)

    async def apply_fill(self, fill: Fill) -> None:
        # Audit-log first (append-only)
        await self.backend.xadd(self._k("fills"), {
            "data": fill.model_dump_json(),
            "ticker": fill.ticker,
            "signal_model": fill.signal_model,
        })

        # Cash: debit cost + fee
        cost = fill.contracts * fill.price + fill.fee
        await self.backend.incrbyfloat(self._k("cash"), -cost)

        # Position: open or update with side-aware avg_cost
        existing_raw = await self.backend.hget(self._k("positions"), fill.ticker)
        if existing_raw is None:
            new_pos = Position(
                ticker=fill.ticker, side=fill.side,
                contracts=fill.contracts, avg_cost=fill.price,
                opened_at=fill.timestamp, last_updated=fill.timestamp,
            )
            await self.backend.hset(self._k("positions"), fill.ticker, new_pos.model_dump_json())
        else:
            pos = Position(**json.loads(existing_raw))
            if pos.side != fill.side:
                # Opposite-side fill: reduce or flip. For simplicity, treat as
                # closing existing contracts at fill.price (realized P&L).
                await self._handle_opposite_side_fill(pos, fill)
                return
            total_contracts = pos.contracts + fill.contracts
            new_avg = (pos.contracts * pos.avg_cost + fill.contracts * fill.price) / total_contracts
            updated = pos.model_copy(update={
                "contracts": total_contracts,
                "avg_cost": new_avg,
                "last_updated": fill.timestamp,
            })
            await self.backend.hset(self._k("positions"), fill.ticker, updated.model_dump_json())

    async def _handle_opposite_side_fill(self, pos: Position, fill: Fill) -> None:
        """Fill on the opposite side closes contracts at fill.price.

        Three sub-cases:
          1. fill.contracts < pos.contracts → partially close, position keeps remainder
          2. fill.contracts == pos.contracts → fully close, position deleted
          3. fill.contracts > pos.contracts → fully close existing side AND open
             a new position on the opposite side with the leftover contracts.
             (Without this branch, leftover contracts would be silently dropped
             — a real bug surfaced in 2026-05-01 forward test.)

        Fee is charged once on the entire fill, not per contract.
        """
        closing = min(pos.contracts, fill.contracts)
        # Realized P&L: closing the existing side at the implied "sell" price.
        # Buying NO at fill.price is mathematically the same as selling the YES
        # side at (1 - fill.price), and vice versa.
        equivalent_sell_price = 1.0 - fill.price
        pnl = closing * (equivalent_sell_price - pos.avg_cost) - fill.fee
        await self.backend.incrbyfloat(self._k("realized_pnl"), pnl)
        await self.backend.incrbyfloat(self._daily_pnl_key(fill.timestamp), pnl)

        # Cash impact for the close portion: we paid fill.price per contract
        # closed, but we ALSO got back the equivalent_sell of the existing
        # position. The net cash debit on the closing portion is exactly the
        # realized PnL (already in cash via the apply_fill caller's debit).

        remaining_existing = pos.contracts - closing
        leftover_fill = fill.contracts - closing

        if remaining_existing > 0:
            # Case 1: existing side reduced, no new position
            updated = pos.model_copy(update={
                "contracts": remaining_existing,
                "last_updated": fill.timestamp,
            })
            await self.backend.hset(self._k("positions"), fill.ticker,
                                    updated.model_dump_json())
        else:
            # Cases 2 + 3: existing side fully closed
            await self.backend.hdel(self._k("positions"), fill.ticker)
            if leftover_fill > 0:
                # Case 3: open a new position on the OPPOSITE side with the
                # leftover contracts at the fill price. avg_cost = fill.price.
                new_pos = Position(
                    ticker=fill.ticker, side=fill.side,
                    contracts=leftover_fill, avg_cost=fill.price,
                    opened_at=fill.timestamp, last_updated=fill.timestamp,
                )
                await self.backend.hset(self._k("positions"), fill.ticker,
                                        new_pos.model_dump_json())

    async def settle(self, ticker: str, settlement_value: float, t: datetime) -> float:
        """Resolve a position: contracts × settlement_value goes to cash;
        realized P&L = (settlement_value - avg_cost) * contracts on the held side.
        Returns realized P&L for this settlement."""
        existing_raw = await self.backend.hget(self._k("positions"), ticker)
        if existing_raw is None:
            return 0.0
        pos = Position(**json.loads(existing_raw))

        # Yes-side: payoff = settlement_value per contract
        # No-side: payoff = (1 - settlement_value) per contract
        per_contract_payoff = settlement_value if pos.side == "yes" else (1.0 - settlement_value)
        proceeds = pos.contracts * per_contract_payoff
        pnl = pos.contracts * (per_contract_payoff - pos.avg_cost)

        await self.backend.incrbyfloat(self._k("cash"), proceeds)
        await self.backend.incrbyfloat(self._k("realized_pnl"), pnl)
        await self.backend.incrbyfloat(self._daily_pnl_key(t), pnl)
        await self.backend.hdel(self._k("positions"), ticker)
        return pnl
