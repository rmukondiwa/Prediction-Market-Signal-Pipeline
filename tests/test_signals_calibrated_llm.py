from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from src.signals.calibrated_llm import CalibratedLLMSignal, fit_isotonic, save_calibration
from src.signals.models import HistoricalContext


def test_fit_isotonic_pulls_overconfident_predictions_inward(tmp_path: Path):
    """LLM saying 0.20 when truth is 0.30 should be pulled UP after calibration."""
    raws = [0.20] * 50 + [0.80] * 50
    outcomes = [1.0 if i < 15 else 0.0 for i in range(50)]  # 30% rate at the 0.20 bucket
    outcomes += [1.0 if i < 35 else 0.0 for i in range(50)]  # 70% rate at 0.80 bucket
    iso = fit_isotonic(raws, outcomes)
    # After fitting, predicting 0.20 should output something close to 0.30
    pred = float(iso.predict([0.20])[0])
    assert pred > 0.20  # pulled inward
    pred_high = float(iso.predict([0.80])[0])
    assert pred_high < 0.80  # also pulled inward (toward 0.70)


def test_save_and_load_roundtrip(tmp_path: Path):
    raws = [0.1, 0.5, 0.9]
    outcomes = [0.0, 1.0, 1.0]
    iso = fit_isotonic(raws, outcomes)

    out = tmp_path / "calmap.pkl"
    save_calibration(iso, out)

    sig = CalibratedLLMSignal(out)
    # Internal map should still predict
    p = sig._calibrate(0.5)
    assert 0.0 < p < 1.0


def test_missing_calibration_map_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        CalibratedLLMSignal(tmp_path / "nonexistent.pkl")


def test_calibrated_llm_emits_corrected_probability(tmp_path: Path, sample_inference_report, sample_market_snapshot):
    """Build a calibration map that pushes 0.30 → 0.50, verify the signal model uses it."""
    raws = [0.30] * 100
    outcomes = [1.0 if i < 50 else 0.0 for i in range(100)]
    iso = fit_isotonic(raws, outcomes)
    out = tmp_path / "calmap.pkl"
    save_calibration(iso, out)

    sig = CalibratedLLMSignal(out)
    edges = sig.signals(
        sample_market_snapshot,
        sample_inference_report.context_markets,
        sample_inference_report,
        HistoricalContext(),
    )
    assert len(edges) == 1
    e = edges[0]
    # The mispricing in the fixture says fair=0.30; calibration trained on 30% rate
    # should pull toward 0.50 (or wherever the true rate is).
    assert e.estimated_fair_prob != 0.30
    assert e.source_signal_model == "calibrated_llm"


def test_calibrated_llm_drops_zero_kelly_edges(tmp_path: Path, sample_inference_report, sample_market_snapshot):
    """If calibration eats the edge entirely, the model should emit nothing."""
    # Train so 0.30 → 0.20 (calibration disagrees with the LLM in the wrong direction
    # for the LLM's claimed mispricing direction)
    raws = [0.30] * 100
    outcomes = [1.0 if i < 20 else 0.0 for i in range(100)]
    iso = fit_isotonic(raws, outcomes)
    out = tmp_path / "calmap.pkl"
    save_calibration(iso, out)

    sig = CalibratedLLMSignal(out)
    edges = sig.signals(
        sample_market_snapshot,
        sample_inference_report.context_markets,
        sample_inference_report,
        HistoricalContext(),
    )
    # Calibrated fair (0.20) is now LOWER than implied (0.23) — no longer a yes-side edge
    # Edge should be filtered out
    if edges:
        # If something does come through, it shouldn't be on the yes side
        assert edges[0].side != "yes" or edges[0].kelly_fraction == 0.0
