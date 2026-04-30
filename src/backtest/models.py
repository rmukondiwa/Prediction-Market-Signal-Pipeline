"""
Backtest data types.

Configurable inputs (BacktestConfig), simulated outputs (SimulatedFill,
SignalModelMetrics, BacktestReport).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RiskLimitsConfig(BaseModel):
    """Same shape as src.portfolio.models.RiskLimits, restated here so backtest
    configs can be loaded from YAML without importing the portfolio module."""
    max_per_market_usd: float = 500.0
    max_total_exposure_usd: float = 5_000.0
    max_correlated_exposure_usd: float = 1_500.0
    daily_loss_limit_usd: float = 500.0
    max_kelly_fraction: float = 0.25
    kill_switch: bool = False


class BacktestConfig(BaseModel):
    start_date: datetime
    end_date: datetime
    granularity: str = "1h"
    universe: list[str] | None = None  # None = all resolved markets in window
    starting_capital: float = 10_000.0
    fill_model: str = "midpoint"
    slippage_per_contract: float = 0.01
    fee_per_contract: float = 0.07
    risk_limits: RiskLimitsConfig = Field(default_factory=RiskLimitsConfig)
    signal_models: list[str] = Field(default_factory=lambda: ["raw_llm"])


class SimulatedFill(BaseModel):
    ticker: str
    side: str
    contracts: int
    price: float
    fee: float
    timestamp: datetime
    signal_model: str
    edge_thesis: str
    estimated_fair_prob: float
    current_implied_prob: float


class SignalModelMetrics(BaseModel):
    name: str
    n_decisions: int
    n_fills: int
    pnl_realized: float
    pnl_unrealized: float
    by_confidence: dict[str, dict[str, float]] = Field(default_factory=dict)
    brier_score: float
    sharpe: float | None
    max_drawdown: float
    edge_decay_curve: list[tuple[int, float]] = Field(default_factory=list)
    fills: list[SimulatedFill] = Field(default_factory=list)


class BacktestReport(BaseModel):
    config: BacktestConfig
    market_baseline_brier: float
    per_signal: dict[str, SignalModelMetrics] = Field(default_factory=dict)
    cache_stats: dict[str, Any] = Field(default_factory=dict)


class Candle(BaseModel):
    """OHLC bar from Kalshi candlestick API or reconstructed from archive."""
    ticker: str
    timestamp: datetime
    open: float  # in cents 0..100
    high: float
    low: float
    close: float
    volume: int = 0
