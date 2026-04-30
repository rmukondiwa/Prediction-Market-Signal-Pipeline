"""
Signal model output types.

`CalibratedEdge` is the trading signal — distinct from `Edge` (which is
LLM-asserted). The probabilities here are intended to be empirically grounded
by the SignalModel that produced them.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class CalibratedEdge(BaseModel):
    """A trading signal: which ticker, which side, how much to size."""
    ticker: str
    title: str
    side: str  # "yes" or "no"
    estimated_fair_prob: float  # calibrated by the signal model, NOT the LLM
    current_implied_prob: float
    edge_pp: float  # estimated_fair - implied, in probability points
    confidence: float  # 0..1, model-specific scoring
    kelly_fraction: float  # quarter-Kelly, clamped 0..0.25
    thesis: str
    source_signal_model: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class HistoricalContext(BaseModel):
    """
    Read-only handle to historical data signal models may need.

    The runner constructs one of these and passes it to every SignalModel
    invocation. Models that need history fetch from here; stateless models
    ignore it.
    """
    archive_root: Path = Field(default_factory=lambda: Path("data/archive"))
    resolutions: dict[str, Any] = Field(default_factory=dict)
    candles: dict[str, list[dict]] = Field(default_factory=dict)
    as_of: datetime | None = None  # current backtest timestep, if running

    model_config = {"arbitrary_types_allowed": True}
