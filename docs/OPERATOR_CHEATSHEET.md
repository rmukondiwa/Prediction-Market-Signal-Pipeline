# Operator Cheat Sheet

One-page reference for running the system. Keep this open while operating.

---

## Quick commands

| What | Command |
|---|---|
| **Run all tests** | `pytest tests/ -q` |
| **Refresh catalog + index** | `python -m scripts.build_index --skip-fetch --no-filter` |
| **Train calibration** | `python -m scripts.train_calibration_gemini --n 200 --crypto-only` |
| **Run scanner orchestrator** | `python -m scripts.run_scanners_loop &` |
| **One-shot arb scan (LLM)** | `python -m scripts.scan_cross_platform_arb_llm` |
| **Paper arb orchestrator** | `python -m scripts.run_arb_live` |
| **Measure LLM signal edge** | `python -m scripts.measure_llm_signal_edge --n 30` |
| **Live arb (under supervisor)** | see § Live below |

---

## Live arb command (memorize)

```bash
python -m scripts.supervisor --max-restarts 5 --backoff-base 5 -- \
  python -m scripts.run_arb_live --live \
    --bankroll 500 \
    --daily-loss-limit 30 \
    --drawdown-limit 20 \
    --min-edge 0.02 \
    --max-trade-usd 50 \
    --interval-seconds 300 \
    --use-redis
```

Type `I CONFIRM` exactly when prompted.

---

## Severity → action

| Alert severity | Action |
|---|---|
| `info` | None — log only |
| `warning` | Glance at logs, ignore unless pattern |
| `high` | **Stop, look, act.** Likely positions to manage. |

---

## Stop everything

| Reason | Command |
|---|---|
| Graceful shutdown | `kill -TERM <supervisor_pid>` |
| Hard kill | `kill -KILL <supervisor_pid> <child_pid>` |
| Locate processes | `ps aux \| grep -E "supervisor\|run_arb_live"` |

After hard kill, manually verify:
1. Kalshi positions: `https://kalshi.com/account/positions`
2. Polymarket positions: connect wallet to polymarket.com
3. Open orders on either: cancel manually if needed

---

## Risk gates at a glance

```
            DAILY KILL SWITCH       ← halts new orders entirely
            DRAWDOWN TRAIL           ← halts when peak−current ≥ trail_limit
                ▼
            DRAWDOWN RAMP            ← multiplier 1.0 / 0.5 / 0.25
            (33%/66%/99% of daily)
                ▼
            ┌──────────────────────────────────────┐
            │ size = min(                          │
            │   bankroll × max_fraction,           │
            │   depth × depth_take,                │
            │   per_market_tier_cap_remaining,     │
            │   per_asset_cap_remaining,           │
            │   max_per_order                      │
            │ ) × ramp_multiplier                  │
            └──────────────────────────────────────┘
```

**Default config (RiskConfig):**
- `max_fraction_per_fill` = 5%
- `depth_take_fraction` = 25%
- `max_per_market_usd` = $500
- `max_per_asset_usd` = $1,000
- `daily_loss_limit_usd` = $300
- `drawdown_limit_usd` = $200
- Tier caps cumulative: 30% / 60% / 100% × per_market

---

## Logs to watch during a live run

```
tail -f logs/arb_live.jsonl       # one line per arb attempt
tail -f logs/alerts.jsonl          # all alerts
tail -f logs/scanner_hits.jsonl    # scanner cycle output
```

Filter for trouble:
```bash
grep -E '"severity":"high"' logs/alerts.jsonl
grep '"all_legs_succeeded": false' logs/arb_live.jsonl
```

---

## Drawdown / kill scenarios — what happens

| Realized PnL | Ramp multiplier | Status |
|---|---|---|
| $+50 | 1.00 | Normal, full size |
| $0 | 1.00 | Flat, full size |
| −$50 (limit $300) | 1.00 | Below 33%, full size |
| −$100 | 0.50 | At 33%, fills halved |
| −$150 | 0.50 | Still 33-66%, half |
| −$200 | 0.25 | Past 66%, quarter size |
| −$280 | 0.25 | Approaching limit |
| **−$300** | **0.00** | **KILLED — kill switch tripped** |
| −$1000 | 0.00 | Stays killed (process restart needed) |

Drawdown trail:
| Peak realized | Current realized | Drawdown | Status |
|---|---|---|---|
| $+150 | $+50 | $100 | OK (trail_limit=$200) |
| $+150 | $0 | $150 | OK |
| $+150 | −$50 | $200 | **Trail tripped** |

---

## Scanner cadence

| Scanner | Interval | Cost/cycle | Output file |
|---|---|---|---|
| `scan_alpha` (within-Kalshi) | 30 min | $0 | `reports/alpha.json` |
| `scan_cross_platform_arb` (v1) | 60 min | $0 | `reports/cross_platform_arb.json` |
| `scan_cross_platform_arb_llm` (v2) | manual | ~$0.005 | `reports/cross_platform_arb_llm.json` |
| `run_arb_live`-internal | 5 min | ~$0.005 | `logs/arb_live.jsonl` |

---

## Calibration map sanity check

After running `train_calibration_gemini`, verify the map produces sensible
output:

```python
import pickle, numpy as np
m = pickle.load(open("data/calibration_map.pkl", "rb"))
xs = np.linspace(0.05, 0.95, 19)
print("LLM_prob → calibrated:")
for x, y in zip(xs, m.predict(xs)):
    print(f"  {x:.2f} → {y:.3f}  shift={y-x:+.3f}")
```

Looks healthy: monotone non-decreasing, doesn't pin at 0/1, has variation
across the input range.

Looks broken: large flat steps (suggests under-trained), or non-monotone
(suggests the underlying signal has no real structure).

---

## Webhooks: minimal Discord setup

1. Discord server → Settings → Integrations → Webhooks → New Webhook
2. Copy webhook URL
3. Add to `.env`:
   ```
   DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/<id>/<token>
   ```
4. Test:
   ```python
   from src.utils.alerts import alert_async
   import asyncio
   asyncio.run(alert_async("Hello", severity="warning",
                            context={"test": True}))
   ```

---

## Common diagnostics

```bash
# Is the supervisor alive?
ps aux | grep supervisor

# Is Redis up?
redis-cli ping  # → PONG

# Latest realized PnL?
redis-cli get portfolio:live_arb:realized_pnl

# Latest cash?
redis-cli get portfolio:live_arb:cash

# All open positions?
redis-cli hgetall portfolio:live_arb:positions

# Reset state (DANGER — wipes Redis ledger)
redis-cli FLUSHDB
```

---

*Pin this. Reread before each live session.*
