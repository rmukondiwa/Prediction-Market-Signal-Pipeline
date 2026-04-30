"""
CalibratedLLMSignal — empirical correction of the LLM's probability bias.

Fits an isotonic regression on (LLM_estimated_fair_prob, settled_outcome) pairs
from resolved markets. Applies the calibration map to live LLM outputs to
produce a corrected `estimated_fair_prob`. The LLM's *ranking* often survives
even when its absolute numbers are off — calibration learning extracts that.

The fit happens offline via `scripts/fit_calibration.py`; this module just
loads the pickled mapping and applies it.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from src.context.models import ContextMarket
from src.inference.models import InferenceReport
from src.insight.models import MarketSnapshot
from src.signals.models import CalibratedEdge, HistoricalContext
from src.signals.protocol import quarter_kelly
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CalibratedLLMSignal:
    name = "calibrated_llm"

    def __init__(self, calibration_map_path: Path | str):
        self.path = Path(calibration_map_path)
        self.map: Any = None  # IsotonicRegression or similar
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Calibration map not found at {self.path}. "
                "Fit one first via: python -m scripts.fit_calibration"
            )
        with self.path.open("rb") as fh:
            self.map = pickle.load(fh)
        logger.info("Calibration map loaded", extra={"path": str(self.path)})

    def _calibrate(self, raw_prob: float) -> float:
        # IsotonicRegression has a .predict(X) interface; fall back to
        # callable-style if a bare function was pickled instead.
        if hasattr(self.map, "predict"):
            value = float(self.map.predict([raw_prob])[0])
        elif callable(self.map):
            value = float(self.map(raw_prob))
        else:
            raise TypeError(f"Unknown calibration map type: {type(self.map)}")
        return max(0.001, min(0.999, value))

    def signals(
        self,
        focus: MarketSnapshot,
        context: list[ContextMarket],
        llm_report: InferenceReport,
        history: HistoricalContext,
    ) -> list[CalibratedEdge]:
        out: list[CalibratedEdge] = []
        for edge in llm_report.suggested_edges:
            mispricing = next(
                (m for m in llm_report.detected_mispricings if m.ticker == edge.ticker),
                None,
            )
            if mispricing is None:
                continue

            implied = mispricing.current_implied_prob
            raw_fair = mispricing.estimated_fair_prob
            calibrated_fair = self._calibrate(raw_fair)

            # Recompute kelly with calibrated probability — this is the whole point
            kf = quarter_kelly(calibrated_fair, implied, edge.side)
            if kf <= 0:
                continue

            edge_pp = (calibrated_fair - implied) if edge.side == "yes" else ((1 - calibrated_fair) - (1 - implied))
            out.append(
                CalibratedEdge(
                    ticker=edge.ticker, title=edge.title, side=edge.side,
                    estimated_fair_prob=calibrated_fair,
                    current_implied_prob=implied,
                    edge_pp=edge_pp,
                    confidence=min(1.0, abs(edge_pp) * 4),  # heuristic; tune later
                    kelly_fraction=kf,
                    thesis=(
                        f"LLM said fair={raw_fair:.2%}, isotonic-calibrated to {calibrated_fair:.2%}. "
                        f"Implied={implied:.2%}."
                    ),
                    source_signal_model=self.name,
                    metadata={"raw_llm_fair": raw_fair},
                )
            )
        return out


def fit_isotonic(
    raw_probs: list[float],
    outcomes: list[float],
):
    """Train an IsotonicRegression. Pure function so tests can call it directly."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_probs, outcomes)
    return iso


def save_calibration(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(model, fh)
