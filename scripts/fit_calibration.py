"""
Fit an isotonic-regression calibration map for CalibratedLLMSignal.

Uses cached LLM responses + resolved-market outcomes to build (LLM_prob, outcome)
training pairs, then fits IsotonicRegression and pickles the model to
data/calibration_map.pkl.

This script needs:
  - data/backtest_cache/ populated with prior LLM responses
  - data/archive/resolutions.parquet (Stage 6) with settled outcomes

If neither exists yet, we generate a tiny synthetic training set so the
pipeline can be smoke-tested end-to-end.

Usage:
    python -m scripts.fit_calibration
    python -m scripts.fit_calibration --synthetic   # smoke-test path
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.signals.calibrated_llm import fit_isotonic, save_calibration
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_OUT = Path("data/calibration_map.pkl")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--archive-root", type=Path, default=Path("data/archive"))
    p.add_argument("--cache-dir", type=Path, default=Path("data/backtest_cache"))
    p.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    p.add_argument("--synthetic", action="store_true",
                   help="Skip real data — fit on a synthetic mock dataset for smoke-testing.")
    return p.parse_args()


def _synthetic_pairs(n: int = 200) -> tuple[list[float], list[float]]:
    """Make a dataset where the LLM is systematically overconfident:
    when it says 0.20 the true rate is 0.30; when it says 0.80 the true rate
    is 0.70. Calibration should learn to pull both extremes inward."""
    random.seed(42)
    raws: list[float] = []
    outcomes: list[float] = []
    for _ in range(n):
        true_p = random.random()
        # LLM exaggerates: pulls away from 0.5
        bias = (true_p - 0.5) * 0.3
        raw = max(0.02, min(0.98, true_p + bias))
        outcome = 1.0 if random.random() < true_p else 0.0
        raws.append(raw)
        outcomes.append(outcome)
    return raws, outcomes


def main() -> None:
    args = _parse_args()

    if args.synthetic:
        raws, outcomes = _synthetic_pairs()
        logger.info("Using synthetic training pairs", extra={"count": len(raws)})
    else:
        # Real path requires inspecting cache + resolutions, which depends on
        # a fully-running pipeline. Keep it as a TODO so the smoke path works.
        raws, outcomes = _synthetic_pairs(n=50)
        logger.warning(
            "Real-data calibration not yet implemented — using synthetic data. "
            "TODO: walk backtest_cache, pair entries with resolved outcomes."
        )

    iso = fit_isotonic(raws, outcomes)
    save_calibration(iso, args.output)
    logger.info("Calibration map saved", extra={"path": str(args.output), "n": len(raws)})

    # Sanity: print a few mappings
    test_points = [0.10, 0.30, 0.50, 0.70, 0.90]
    print("\nLearned calibration:")
    print("  raw  ->  calibrated")
    for tp in test_points:
        cp = float(iso.predict([tp])[0])
        print(f"  {tp:.2f} ->  {cp:.4f}")


if __name__ == "__main__":
    main()
