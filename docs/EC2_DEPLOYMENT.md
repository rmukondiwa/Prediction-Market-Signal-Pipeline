# EC2 Deployment

24/7 cloud host for the prediction market pipeline. All four Docker services run here continuously, collecting live market data into Redis and the archive.

Companion to [DOCKER.md](DOCKER.md) and [INFRASTRUCTURE.md](INFRASTRUCTURE.md).

---

## Instance details

| Field | Value |
|---|---|
| Provider | AWS EC2 |
| Instance type | t3.small (2 vCPU, 2 GB RAM) |
| OS | Ubuntu 24.04 LTS |
| Region | us-east-2 (Ohio) |
| Public IP | `<EC2_PUBLIC_IP>` (check AWS console) |
| Key file | `<path-to-your-key.pem>` |
| Code path | `~/Prediction-Market-Signal-Pipeline/` |

---

## SSH access

```bash
ssh -i <path-to-your-key.pem> ubuntu@<EC2_PUBLIC_IP>
```

All commands below assume you are SSH'd into the instance unless stated otherwise.

---

## Running services

```bash
cd ~/Prediction-Market-Signal-Pipeline
docker compose ps
```

Expected output (all four should show `Up`):

```
NAME                                          STATUS
...-ingestion-1          Up X hours   # Kalshi WebSocket → Redis
...-live-book-scanner-1  Up X hours   # order book monotonicity scanner
...-redis-1              Up X hours   # data store
...-scanners-1           Up X hours   # signal scanners on Redis stream
```

### What each service does

| Service | Command | Purpose |
|---|---|---|
| `redis` | `redis:7-alpine` | Stores live event streams; data persists across restarts |
| `ingestion` | `python3 main.py` | Subscribes to Kalshi WebSocket, writes market/trade/orderbook events to Redis Streams |
| `scanners` | `python3 -m scripts.run_scanners_loop` | Reads Redis streams, runs arb/signal scanners, logs hits to `logs/` |
| `live-book-scanner` | `python3 -m scripts.scan_live_book` | Checks L2 order book for monotonicity violations every 100ms; logs to `logs/live_book_violations.jsonl` |

---

## Checking logs

```bash
# All services, live-follow
docker compose logs -f

# Single service
docker compose logs -f ingestion
docker compose logs -f live-book-scanner

# Last 50 lines without following
docker compose logs --tail 50 ingestion

# Scanner hits (structural arb alerts)
tail -f logs/scanner_hits.jsonl

# Order book violations
tail -f logs/live_book_violations.jsonl
```

---

## Starting / restarting services

```bash
cd ~/Prediction-Market-Signal-Pipeline

# Start everything (core + trading profiles)
docker compose --profile core --profile trading up -d

# Restart a single crashed service
docker compose restart ingestion

# Full stop then start (use when you've changed .env or rebuilt the image)
docker compose --profile core --profile trading down
docker compose --profile core --profile trading up -d
```

---

## Updating the code

Pull the latest changes from GitHub and rebuild the Docker image:

```bash
cd ~/Prediction-Market-Signal-Pipeline
git pull
docker compose build
docker compose --profile core --profile trading up -d
```

The build step (~60-90s) only reinstalls Python packages if `requirements.txt` changed. Source-only changes rebuild much faster.

---

## Uploading data files from local machine

The FAISS index and catalog are too large to build on the t3.small (OOM at 170k markets). Build locally and upload:

```bash
# Run from your local machine (not SSH session)
scp -i <path-to-your-key.pem> \
  data/catalog.json \
  data/vectors.faiss \
  data/vectors_meta.json \
  ubuntu@<EC2_PUBLIC_IP>:~/Prediction-Market-Signal-Pipeline/data/
```

These files land in `./data/` which is bind-mounted into all containers — no restart needed.

---

## Running one-off scripts remotely

```bash
# Fetch settled markets and update the resolution archive
docker compose run --rm app python3 -m scripts.fetch_resolutions --settled --max-settled 5000

# Fetch resolutions for specific expired tickers
docker compose run --rm app python3 -m scripts.fetch_resolutions \
  --tickers KXBTCD-26MAY14-T83199,KXBTCD-26MAY14-T80000 --concurrency 4

# Check how many resolutions are in the archive
docker compose run --rm app python3 -c "
import pyarrow.parquet as pq
t = pq.read_table('data/archive/resolutions.parquet')
rows = t.to_pylist()
settled = [r for r in rows if r.get('settlement_value') is not None]
print(f'Total: {len(rows)}, Settled: {len(settled)}')
"
```

---

## Key files on the instance

```
~/Prediction-Market-Signal-Pipeline/
├── .env                    # API keys and config (never commit this)
├── keys/
│   └── <your-key-file>.txt # Kalshi RSA private key (bind-mounted read-only)
├── data/
│   ├── catalog.json        # 22k Kalshi markets with titles + prices
│   ├── vectors.faiss       # FAISS index (57 MB, uploaded from local)
│   ├── vectors_meta.json   # Per-vector metadata
│   └── archive/
│       └── resolutions.parquet  # Settlement history for backtests
├── logs/
│   ├── scanner_hits.jsonl
│   └── live_book_violations.jsonl
└── reports/
```

---

## .env gotchas

Two settings differ between local development and EC2:

| Setting | Local | EC2 (Docker) |
|---|---|---|
| `REDIS_HOST` | `localhost` | `redis` |
| `KALSHI_PRIVATE_KEY_PATH` | `keys/<your-key-file>.txt` | `keys/<your-key-file>.txt` |

`REDIS_HOST=redis` is required inside Docker because containers communicate via the Docker network using service names, not `localhost`.

---

## If a service keeps restarting

```bash
# Check the error
docker compose logs --tail 30 <service-name>
```

Common causes:

| Error | Fix |
|---|---|
| `FileNotFoundError: data/catalog.json` | Upload `catalog.json` from local (see above) |
| `ConnectionError: connecting to localhost:6379` | Set `REDIS_HOST=redis` in `.env`, restart |
| `FileNotFoundError: keys/<your-key-file>.txt` | `sudo chown ubuntu:ubuntu ~/Prediction-Market-Signal-Pipeline/keys` then re-upload the key |
| OOM killed (exit code 137) | Don't run `build_index` on EC2 — build locally and upload |

---

## Why EC2 matters for backtesting

The EC2 instance collects price snapshots continuously into `data/archive/`. After 30+ days of collection:

- `run_backtest.py` can replay price history with realistic entry prices (T-7 days before expiration)
- `bayesian_base_rate` and `coherence_regression` signals will have settled analogues to draw from
- `consistency_arb` will have simultaneous snapshots of deadline-stacked market pairs

Until then, `measure_llm_signal_edge` is the available signal validation tool (tests direction accuracy on resolved markets using near-expiration catalog prices).
