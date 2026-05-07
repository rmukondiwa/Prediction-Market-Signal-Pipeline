# Prediction Market Signal Pipeline

Cross-platform arbitrage and signal infrastructure for prediction markets.
Built around Kalshi (primary) and Polymarket (secondary), with LLM-assisted
question matching, Gemini-embedded vector search across 15,801 markets, and
a full live-trading stack.

```
┌──────────────────┐    ┌────────────────┐    ┌───────────────────┐
│ Kalshi WS + REST │ ─▶ │  Catalog +     │ ─▶ │  Arb scanners     │
│ Polymarket CLOB  │    │  FAISS index   │    │  (within / cross) │
└──────────────────┘    └────────────────┘    └────────┬──────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌────────────────┐    ┌─────────────────────┐
│  Live alerts    │ ◀─ │  Multi-leg     │ ◀─ │  Risk gates         │
│  (4 sinks)      │    │  coordinator   │    │  + portfolio state  │
└─────────────────┘    └────────────────┘    └─────────────────────┘
```

**Status:** infrastructure complete. 179 unit tests passing. Live trading
gated behind typed confirmation.

## Quickstart

```bash
git clone <repo> && cd Prediction-Market-Signal-Pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Add API keys to .env (see USER_MANUAL.md § 2)
cp .env.example .env && $EDITOR .env

# First-time: build the catalog + vector index
python -m scripts.build_index

# Run scanners, see what they find (read-only, no money at risk)
python -m scripts.run_scanners_loop &
python -m scripts.scan_cross_platform_arb_llm --poly-limit 50 --max-verify 30

# Paper-mode arb orchestrator
python -m scripts.run_arb_live
```

## Documentation

| Doc | What it covers |
|---|---|
| **[docs/USER_MANUAL.md](docs/USER_MANUAL.md)** | **Start here.** End-to-end usage manual with diagrams, examples, troubleshooting, cost reference. |
| [docs/OPERATOR_CHEATSHEET.md](docs/OPERATOR_CHEATSHEET.md) | One-page reference. Pin while running live. |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture with diagrams, layer-by-layer component inventory. |
| [docs/ALPHA_RESEARCH.md](docs/ALPHA_RESEARCH.md) | 10 unexplored alpha sources with concrete test instructions + a kill-fast research protocol. |
| [docs/INFRASTRUCTURE.md](docs/INFRASTRUCTURE.md) | Existing infra reference — Kalshi WS, Redis, etc. |
| [docs/STRATEGIES.md](docs/STRATEGIES.md) | Strategy catalog (built / researched / identified). |
| [experiments/decay/README.md](experiments/decay/README.md) | Archived directional decay strategy work. |

## Test suite

```bash
pytest tests/ -q
# → 179 passed
```

## License

Private — not for distribution.
