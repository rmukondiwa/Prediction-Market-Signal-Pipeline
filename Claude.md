# Prediction Market Signal Pipeline

## Project Overview

The Prediction Market Signal Pipeline is a modular system designed to ingest,
normalize, store, and analyze prediction market data in order to generate
signals based on market inefficiencies.

The architecture is inspired by real quantitative trading infrastructure and
prioritizes:

- real-time data ingestion
- low-latency event streaming
- modular system design
- exchange-agnostic downstream systems

The initial exchange supported is **Kalshi**.

Future integrations may include:

- Polymarket
- PredictIt
- Manifold Markets

---

# System Architecture

The pipeline is composed of several layers.

1. Ingestion Layer
2. Normalization Layer
3. Event Bus
4. Storage Layer
5. Signal Generation Layer
6. Strategy Evaluation

Each layer is developed in **separate feature branches**.

---

# Data Flow

Kalshi WebSocket API
        |
        v
KalshiWebSocketClient
        |
        v
Message Parser
        |
        v
Normalizer
        |
        v
Internal Event Schema
        |
        v
Event Publisher
        |
        v
Redis Streams
        |
        v
Storage + Signal Engines

---

# Technology Stack

Language:
Python 3.11+

Core Libraries:

asyncio  
websockets  
aiohttp  
pydantic  

Messaging / Event Bus:

Redis Streams

Future Storage:

PostgreSQL  
TimescaleDB

---

# Repository Structure

src/

ingestion/
kalshi/
websocket_client.py
message_parser.py
normalizer.py

publisher/
event_publisher.py

models/
market_event.py
trade_event.py
orderbook_event.py

storage/

signals/

config/
kalshi_config.py
redis_config.py

utils/
logging.py
retry.py

---

# Internal Event Schemas

All downstream components must rely on internal standardized event schemas.

This ensures the pipeline is **exchange-agnostic**.

Example MarketEvent:

Fields:

market_id  
ticker  
bid  
ask  
last_price  
volume  
timestamp

Example TradeEvent:

market_id  
price  
size  
side  
timestamp

---

# Branch Development Phases

The system will be implemented incrementally using feature branches.

---

# Phase 1 — feature/ingestion-layer

Current Phase.

Goal:

Build a real-time ingestion system that connects to the Kalshi API using
WebSockets and streams market data into Redis Streams.

This layer is responsible ONLY for:

- connecting to Kalshi
- receiving real-time market data
- parsing messages
- normalizing events
- publishing to Redis Streams

It should NOT contain:

- signal generation
- analytics
- storage logic
- trading logic

---

## Ingestion Components

KalshiWebSocketClient

Responsible for:

- connecting to Kalshi WebSocket endpoint
- maintaining persistent connection
- handling reconnect logic
- managing subscriptions
- receiving raw messages

Must use:

asyncio  
websockets library

---

MessageParser

Responsible for:

- interpreting raw JSON messages
- identifying message types
- extracting market fields

---

Normalizer

Responsible for:

- converting exchange-specific data into internal event schemas

Example outputs:

MarketEvent  
TradeEvent  
OrderBookEvent

---

EventPublisher

Responsible for:

- serializing normalized events
- publishing them to Redis Streams

Example streams:

market_events  
trade_events  
orderbook_events

---

# Phase 2 — feature/storage-layer

Goal:

Store normalized events from Redis Streams into a persistent database.

Responsibilities:

- Redis stream consumers
- batch inserts
- schema design
- historical market data storage

Target database:

PostgreSQL

Future optimization:

TimescaleDB

---

# Phase 3 — feature/signal-engine

Goal:

Generate signals based on prediction market pricing data.

Potential signal types:

- probability mispricing
- cross-market arbitrage
- volatility spikes
- event probability divergence

Inputs:

historical market data  
live market stream

Outputs:

signal events

---

# Phase 4 — feature/backtesting-engine

Goal:

Evaluate signal strategies against historical market data.

Components:

historical replay engine  
signal performance evaluation  
strategy comparison tools

Metrics:

Sharpe ratio  
max drawdown  
hit rate  
expected value

---

# Phase 5 — feature/strategy-layer

Goal:

Implement strategy logic on top of generated signals.

Examples:

mean reversion strategies  
probability momentum strategies  
market inefficiency detection

---

# Coding Rules

Claude must follow these coding rules when generating code for this project.

---

## Async First

All network operations must be asynchronous.

Use:

asyncio

Avoid blocking calls.

Never use synchronous requests in the ingestion layer.

---

## Streaming First

The system must prioritize streaming data over polling.

Polling REST endpoints should only be used for:

- initialization
- metadata
- fallback scenarios

---

## Modular Design

Each module must have a single responsibility.

Examples:

websocket_client → connection management  
parser → message parsing  
normalizer → schema transformation  
publisher → event streaming

Modules must remain loosely coupled.

---

## Exchange Agnostic Design

Downstream components must not depend on Kalshi-specific formats.

All exchange-specific fields must be converted into internal schemas.

---

## Structured Logging

All components must use structured logging.

Avoid print statements.

Use a centralized logging utility located in:

utils/logging.py

Log important events such as:

connection opened  
connection lost  
reconnect attempts  
message parsing errors

---

## Reconnect and Resilience

WebSocket clients must support:

automatic reconnect  
exponential backoff  
heartbeat / ping checks

This prevents ingestion failures.

---

## Error Handling

All network operations must include:

try/except handling  
retry logic  
clear error logs

Avoid silent failures.

---

## Type Safety

Use type hints across all modules.

Use Pydantic models for event schemas when appropriate.

---

## Configuration

All credentials and endpoints must be defined in config files.

Never hardcode secrets in the codebase.

---

# Future Exchanges

The architecture must allow easy addition of new exchanges.

Example structure:

ingestion/

kalshi/
polymarket/
predictit/

Each exchange should implement its own:

websocket client  
parser  
normalizer

But output the same internal event schemas.

---

# Development Philosophy

The system should resemble real quantitative trading data pipelines.

Key priorities:

low latency  
fault tolerance  
clean architecture  
modular expansion

The goal is to build a reusable infrastructure for prediction market
signal generation.