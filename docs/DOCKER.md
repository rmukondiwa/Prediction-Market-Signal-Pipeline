# Docker Deployment

Containerised runtime for the pipeline. Single image (`pmsp:latest`), services split by Docker Compose profile.
Companion to [INFRASTRUCTURE.md](INFRASTRUCTURE.md) and [OPERATOR_CHEATSHEET.md](OPERATOR_CHEATSHEET.md).

---

## Container layout

```
┌─────────────────────────────────────────────────────────┐
│ pmsp:latest  (python:3.12-slim + requirements.txt)       │
│                                                          │
│  ingestion  ──► main.py          Kalshi WS → Redis       │
│  scanners   ──► run_scanners_loop.py   arb watcher       │
│  scheduler  ──► cron → build_index    daily 06:00 UTC    │
│  app        ──► (no entrypoint)        one-off commands  │
│  infer      ──► infer.py              on-demand          │
└─────────────────────────────────────────────────────────┘
              │                  │
              ▼                  ▼
    redis:7-alpine        ./data  ./logs  ./keys
    (named volume)        (bind-mounted from host)
```

---

## Profiles

| Profile | Services started |
|---|---|
| `core` | `redis` + `ingestion` |
| `trading` | `redis` + `scanners` |
| `infra` | `scheduler` |
| `all` | `redis` + `ingestion` + `scanners` + `scheduler` |
| `tools` | enables `infer` for `docker compose run` |
| _(none)_ | nothing auto-starts — only `docker compose build` |

---

## Prerequisites

`.env` must contain:

```
# Kalshi auth
KALSHI_API_KEY_ID=...
KALSHI_PRIVATE_KEY_PATH=keys/<your-key-file>.txt

# LLM
OPENAI_API_KEY=...
GROQ_API_KEY=...

# Redis — must be the service name inside containers
REDIS_HOST=redis
REDIS_PORT=6379
```

`keys/` must exist on the host. It is never copied into the image — it is bind-mounted read-only at runtime.

---

## Build

```bash
docker compose build
```

Force clean rebuild:

```bash
docker compose build --no-cache
```

Rebuild whenever `requirements.txt` or source changes. The image is cached in layers: dependency install (~90s) only reruns if `requirements.txt` changes.

---

## Starting services

```bash
# Core pipeline only (ingestion + Redis)
docker compose --profile core up -d

# Add arb scanners
docker compose --profile trading up -d

# Add daily index scheduler
docker compose --profile infra up -d

# Everything
docker compose --profile all up -d
```

---

## One-off commands

The `app` service has no entrypoint — use it for any ad-hoc command:

```bash
# Build the full market index (catalog + FAISS)
docker compose run --rm app python3 -m scripts.build_index

# Cap at 500 markets for a quick dev run
docker compose run --rm app python3 -m scripts.build_index --max-markets 500

# Re-embed without re-fetching (reuses existing data/catalog.json)
docker compose run --rm app python3 -m scripts.build_index --skip-fetch

# List valid tickers from the catalog
docker compose run --rm app python3 -c "
import json
ms = json.load(open('data/catalog.json'))
for m in ms[:30]: print(m['ticker'], '-', m['title'][:70])
"
```

---

## Inference

```bash
# Full inference (uses catalog prices, no Redis required)
docker compose --profile tools run --rm infer INXI-2026-Y --no-redis

# Dry run — retrieval + reranking only, skips LLM inference
docker compose --profile tools run --rm infer INXI-2026-Y --dry-run

# Rebuild index then infer in one shot
docker compose --profile tools run --rm infer INXI-2026-Y --refresh-catalog --no-redis
```

The `infer` service has `python3 infer.py` as its entrypoint — arguments appended directly are the ticker + flags.

---

## Volumes

| Host path | Container path | Contains |
|---|---|---|
| `./data/` | `/app/data/` | `catalog.json`, `vectors.faiss`, `vectors_meta.json` |
| `./logs/` | `/app/logs/` | Scanner hits, cron output |
| `./reports/` | `/app/reports/` | Per-scan JSON reports |
| `./keys/` | `/app/keys/` (`:ro`) | Kalshi RSA private key |
| _(named)_ `redis_data` | `/data` | Redis persistence |

Artifacts written inside a container persist on the host immediately. The FAISS index built by `build_index` is available to `infer` and `scanners` without restarting anything.

---

## Scheduler

The `scheduler` service runs `cron -f` with a single job (`docker/crontab`):

```
0 6 * * * cd /app && python3 -m scripts.build_index >> /app/logs/build_index.log 2>&1
```

Rebuilds the full market catalog and FAISS index every day at 06:00 UTC. Output in `logs/build_index.log`.

To trigger an immediate rebuild without waiting:

```bash
docker compose run --rm app python3 -m scripts.build_index
```

---

## Logs

```bash
# All containers
docker compose logs -f

# Single service
docker compose logs -f ingestion
docker compose logs -f scanners

# Cron output
tail -f logs/build_index.log

# Scanner hits (JSONL)
tail -f logs/scanner_hits.jsonl
grep '"ALERT"' logs/scanner_hits.jsonl
```

---

## Stop / update

```bash
# Stop all running services
docker compose --profile all down

# Stop and wipe Redis state (positions, streams)
docker compose --profile all down -v

# Pull + rebuild + restart
git pull
docker compose build
docker compose --profile all up -d
```

---

## Gotchas

1. **`REDIS_HOST` must be `redis` inside Docker**, not `localhost`. Change it back to `localhost` when running locally outside containers.
2. **`keys/` is not in the image.** It is bind-mounted at runtime. If the host path doesn't exist or the env var `KALSHI_PRIVATE_KEY_PATH` points to the wrong filename, ingestion and `build_index` both crash on startup.
3. **`docker compose build` with no profile** works because the `app` service has no profile assigned. All other app services are profile-gated and are skipped by a plain `build`.
4. **Index freshness.** The `infer` and `scanners` containers read `./data/` from the host. If the scheduler hasn't run yet (or `build_index` was never run manually), those containers will fail with a missing-file error on startup.
5. **`app` service starts on every `up` call.** It has no profile, so it starts alongside whatever profile you activate. It exits immediately (no command set) — this is harmless.
