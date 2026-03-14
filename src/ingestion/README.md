# Ingestion Layer

Responsible for connecting to the Kalshi WebSocket API, receiving real-time market data, parsing and normalizing it into internal schemas, and publishing events to Redis Streams.

---

## How It Works

### 1. Startup (`main.py`)
Loads your `.env`, creates the config objects, connects to Redis, then starts the WebSocket client.

### 2. Authentication (`kalshi/websocket_client.py`)
Before connecting, it builds auth headers by:
- Taking the current timestamp in milliseconds
- Signing `timestamp + "GET" + "/trade-api/ws/v2"` with your RSA private key
- Attaching `KALSHI-ACCESS-KEY`, `KALSHI-ACCESS-TIMESTAMP`, and `KALSHI-ACCESS-SIGNATURE` as headers on the WebSocket handshake

### 3. Subscription (`kalshi/websocket_client.py`)
Once connected, it sends a subscribe command to Kalshi requesting 3 channels for each configured market ticker:
- `ticker` — best bid/ask price updates
- `orderbook_delta` — incremental orderbook changes
- `trade` — executed trades

### 4. Receiving Messages (`kalshi/message_parser.py`)
Every raw JSON message from Kalshi is parsed and classified into a typed enum: `TICKER`, `TRADE`, `ORDERBOOK_SNAPSHOT`, `ORDERBOOK_DELTA`, or `SUBSCRIBED`.

### 5. Normalization (`kalshi/normalizer.py`)
The Kalshi-specific message is converted into an exchange-agnostic internal schema:
- `ticker` → `MarketEvent` (yes_bid, yes_ask, last_price, volume...)
- `trade` → `TradeEvent` (yes_price, count, taker_side...)
- `orderbook_snapshot/delta` → `OrderBookEvent`

This is the key architectural decision — everything downstream only ever sees these internal schemas, never raw Kalshi data.

### 6. Publishing (`../publisher/event_publisher.py`)
Each normalized event is serialized and written to a Redis Stream:
- `MarketEvent` → `market_events` stream
- `TradeEvent` → `trade_events` stream
- `OrderBookEvent` → `orderbook_events` stream

Each entry stores both individual fields (for querying) and a full `data` field with the complete JSON payload.

### 7. Reconnection (`../utils/retry.py`)
If the WebSocket drops at any point, the client automatically retries with exponential backoff — starting at 1 second, doubling each attempt, capped at 60 seconds.

---

## Data Flow

```
Kalshi WebSocket API
        |
        v
KalshiWebSocketClient  (auth, connect, subscribe, reconnect)
        |
        v
MessageParser          (raw JSON → typed ParsedMessage)
        |
        v
Normalizer             (Kalshi format → internal schemas)
        |
        v
EventPublisher         (serialize → Redis Streams)
        |
        v
market_events / trade_events / orderbook_events
```
