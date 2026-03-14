# Prediction Market Signal Pipeline

A modular pipeline that ingests real-time prediction market data, normalizes it into exchange-agnostic schemas, and generates structured market insights using an LLM.

---

## Architecture

The system is composed of five discrete layers, each with a single responsibility:

```
Kalshi WebSocket API
        |
        v
Ingestion Layer        — connect, authenticate, receive raw messages
        |
        v
Normalization Layer    — convert exchange-specific data into internal schemas
        |
        v
Event Bus (Redis)      — publish normalized events to named streams
        |
        v
Extraction Layer       — read latest snapshot from Redis, derive quoted fields
        |
        v
Insight Generation     — prompt LLM with structured snapshot, parse output
        |
        v
Output (JSON)          — InsightReport printed to stdout
```

---

## Workflow Structure

### 1. Ingestion (`src/ingestion/`)
`main.py` starts a persistent WebSocket connection to the Kalshi API. Authentication uses RSA-PSS signing — the client signs `timestamp + "GET" + "/trade-api/ws/v2"` and attaches the signature as a request header. Once connected, it subscribes to three channels per ticker: `ticker` (price updates), `orderbook_delta` (orderbook changes), and `trade` (executed trades).

### 2. Normalization (`src/ingestion/kalshi/normalizer.py`)
Raw Kalshi messages are converted into internal exchange-agnostic schemas:
- `ticker` → `MarketEvent`
- `trade` → `TradeEvent`
- `orderbook_snapshot` / `orderbook_delta` → `OrderBookEvent`

No downstream component ever sees Kalshi-specific fields. This is the key architectural decision that allows new exchanges to be added without touching anything beyond the ingestion layer.

### 3. Event Bus (`src/publisher/event_publisher.py`)
Each normalized event is serialized and published to a Redis Stream:
- `MarketEvent` → `market_events`
- `TradeEvent` → `trade_events`
- `OrderBookEvent` → `orderbook_events`

### 4. Extraction (`src/insight/extractor.py`)
`insight.py` reads the most recent entry for the configured ticker from Redis. It checks `market_events` first and falls back to `orderbook_events` if no ticker data is available. Derived fields (`quoted_price`, `implied_probability`) are computed deterministically from the raw bid/ask levels and returned as a `MarketSnapshot`.

### 5. Insight Generation (`src/insight/generator.py`)
The `MarketSnapshot` is formatted into a structured prompt and sent to OpenAI (`gpt-5-mini`) via the Responses API. The model returns a validated `LLMInsight` object containing an `insight_summary` and three `follow_up_actions`. These are combined with the snapshot into a final `InsightReport` and printed as JSON.

---

## Deterministic vs LLM-Driven

| Component | Type | Reason |
|---|---|---|
| WebSocket client | Deterministic | Protocol-level connection and auth |
| Message parser | Deterministic | Rule-based type classification |
| Normalizer | Deterministic | Field mapping from Kalshi schema to internal schema |
| Event publisher | Deterministic | Serialization and Redis write |
| Extractor | Deterministic | Arithmetic derivation of quoted price and implied probability |
| Insight generator | **LLM-driven** | Natural language summary and follow-up actions require reasoning |

The LLM is used only at the final step. All data collection, parsing, normalization, and extraction are fully deterministic and produce the same output for the same input.

---

## External Source: Kalshi

Kalshi is a regulated prediction market exchange where users trade contracts on real-world events. It was selected because:
- It provides a public WebSocket API with real-time market data
- Contracts are priced in cents (0–100¢), mapping directly to implied probabilities
- The API supports multiple subscription channels (price, orderbook, trades) in a single connection
- It is one of the few regulated U.S. prediction market exchanges with accessible developer tooling

Integration required RSA key-pair authentication. The private key is used to sign each WebSocket handshake, and the key ID is passed as a request header. The API endpoint was identified via Kalshi's developer documentation and confirmed via the `api.elections.kalshi.com` production host.

---

## Scalability Design

The prototype is small but the architecture is designed to extend naturally:

**Multiple input sources**
Each exchange lives in its own subdirectory under `src/ingestion/` (e.g. `kalshi/`, `polymarket/`, `predictit/`). Each implements its own WebSocket client, parser, and normalizer, but outputs the same `MarketEvent`, `TradeEvent`, and `OrderBookEvent` schemas. Everything downstream is already exchange-agnostic.

**Repeated monitoring over time**
Redis Streams act as a persistent, replayable event log. Running `main.py` continuously appends new events. The insight layer can be triggered on a schedule (e.g. via cron or a task queue) to generate a fresh insight for any ticker at any interval.

**Structured storage**
A storage layer (consuming Redis Streams and batch-inserting into PostgreSQL) would sit between the event bus and the extraction layer. The consumer group pattern ensures no events are lost across restarts, and the exchange-agnostic schemas map directly to database tables without any transformation.

**Automated alerting**
The `follow_up_actions` field in `InsightReport` is designed to be machine-readable. A monitoring loop could parse these actions, compare snapshots over time, and route alerts to Slack, email, or a dashboard when thresholds are crossed (e.g. implied probability moves >5%, volume spikes).

---

## Running the Pipeline

### Prerequisites
- Python 3.11+
- Docker (for Redis)
- A Kalshi API key (RSA key pair)
- An OpenAI API key

### Setup
```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH, KALSHI_MARKET_TICKERS, OPENAI_API_KEY
```

### Start Redis
```bash
docker run -d -p 6379:6379 redis:latest
```

### Run ingestion (populate Redis)
```bash
python main.py
```

### Generate insight
```bash
python insight.py
```

---

## Example Output

**Input:** Live orderbook snapshot for `KXINXMAXY-01JAN2027` (S&P 500 max yearly return market)

**Output:**
```json
{
  "structured_data": {
    "event": "KXINXMAXY (01JAN2027)",
    "market": "KXINXMAXY-01JAN2027",
    "outcome": "YES",
    "quoted_price": 0,
    "implied_probability": 0.0,
    "yes_bid": 0,
    "yes_ask": 1,
    "volume": 0,
    "open_interest": 0,
    "source": "kalshi",
    "timestamp": "2026-03-14T16:19:41.154598Z"
  },
  "insight_summary": "The market is pricing the YES outcome at essentially 0% (quoted price 0¢, implied probability 0.0%) with a one-cent ask and no bid liquidity; there is zero volume and zero open interest at the snapshot time. The 1¢ ask suggests a minimal standing quote but overall the market is inactive and currently provides no validated probability signal. For research and risk monitoring this means the contract is uninformative now — any future trade, OI change, or spread movement would be a material signal worth investigating.",
  "follow_up_actions": [
    "Compare this snapshot to the prior 24–72 hour snapshots to detect any changes in quoted price, bid/ask, volume, or open interest; flag any nonzero volume or OI and record the time of first change.",
    "Query the market metadata and lifecycle status (open/closed/resolved, official resolution date, and full event text) to confirm whether inactivity is due to listing state or proximity to resolution.",
    "Subscribe to real-time updates and generate an immediate alert if: quoted price >0¢, implied probability increases by >0.5 percentage points, or volume or open interest becomes >0; include timestamp and before/after snapshot in the alert payload."
  ]
}
```
