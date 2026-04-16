# CLAUDE.md — Context-Aware Inference Engine Extension

## What this project is

An extension to an existing prediction market data pipeline. The existing system (phases 1-2, merged to `main`) ingests live Kalshi data via WebSocket into Redis Streams and runs single-market LLM insight reports. This extension (phase 4) adds **cross-market context-aware inference**: pulling the full market universe, making it semantically searchable, finding causally related markets, and reasoning across the group to surface mispricings.

## Architecture: Event-driven pipeline

Each stage produces a serialized artifact that the next stage consumes. Stages are independently testable, cacheable, and replaceable. Two execution paths:

- **Cold path** (runs daily/on-demand via `scripts/build_index.py`): catalog fetch → embed → index → disk
- **Hot path** (runs per inference via `infer.py`): load index → vector search → rerank → inference → report

Artifacts flow between stages as files or in-memory objects — never direct function coupling between stages.

## What already exists (DO NOT modify unless necessary)

```
main.py                          # Entry point: Kalshi WS → Redis ingestion pipeline
insight.py                       # Entry point: single-market insight (Redis → LLM → report)
src/config/kalshi_config.py      # Dataclass, loads from .env. rest_base_url defaults to
                                 # https://api.elections.kalshi.com/trade-api/v2
src/config/redis_config.py       # Dataclass, .url property for connection strings
src/ingestion/kalshi/
  websocket_client.py            # RSA-PSS auth, reconnect logic
  message_parser.py              # Raw JSON → ParsedMessage with MessageType enum
  normalizer.py                  # ParsedMessage → MarketEvent | TradeEvent | OrderBookEvent
src/models/
  market_event.py                # Pydantic: yes_bid, yes_ask, volume, open_interest
  trade_event.py                 # Pydantic: yes_price, no_price, count, taker_side
  orderbook_event.py             # Pydantic: yes_levels, no_levels, snapshot/delta
src/publisher/event_publisher.py # Routes events to Redis Streams via XADD
src/insight/
  extractor.py                   # Redis XREVRANGE → MarketSnapshot (no LLM)
  generator.py                   # OpenAI structured output → InsightReport
                                 # IMPORTANT: study strict_schema() helper and response_format pattern — reuse both
  models.py                      # Pydantic: MarketSnapshot, LLMInsight, InsightReport
src/utils/
  logging.py                     # Structured logger. ALWAYS use: from src.utils.logging import get_logger
  retry.py                       # retry_with_backoff() — exponential backoff for async ops
src/catalog/
  models.py                      # ALREADY EXISTS. Defines CatalogMarket Pydantic model. Read this first.
  __init__.py                    # Empty
src/context/__init__.py          # Empty, directory scaffolded
src/inference/__init__.py        # Empty, directory scaffolded
```

## Files to create

Listed in dependency order. Each file depends only on the artifacts of the previous stage.

### Stage 1: Market catalog fetcher

**`src/catalog/fetcher.py`**

Three async functions:
- `fetch_all_markets(base_url) -> list[dict]` — Paginate `GET /markets?status=open&limit=1000`. Use `aiohttp`. Cursor-based pagination: keep fetching until cursor is empty string.
- `fetch_event(session, base_url, event_ticker, semaphore) -> dict | None` — `GET /events/{event_ticker}`. Wrap with `retry_with_backoff()` from `src/utils/retry.py`.
- `build_catalog(base_url) -> list[CatalogMarket]` — Orchestrator: fetch all markets, collect unique event_tickers, fetch each event with `asyncio.Semaphore(18)` for rate limiting (Kalshi allows 20 req/s), merge market + event metadata, return list of CatalogMarket.

Key details:
- Many markets share the same event_ticker. De-duplicate before fetching events. ~3000 markets → ~300-500 unique events.
- Market objects do NOT have a human-readable title. Title comes from the parent event via `GET /events/{event_ticker}`.
- If event fetch fails after retries, log warning and use ticker string as fallback title. Do not crash.
- Compute implied_probability as: `(yes_bid + yes_ask) / 2 / 100`
- Use a single `aiohttp.ClientSession` for all requests in one `build_catalog()` call.
- No authentication needed — both endpoints are public.

**`src/catalog/store.py`**

Two functions:
- `save_catalog(markets, path=Path("data/catalog.json"))` — Use `model_dump(mode="json")` for Pydantic serialization. Create `data/` directory if missing.
- `load_catalog(path) -> list[CatalogMarket]` — Read and deserialize.

Use `orjson` for fast serialization if available, fall back to stdlib `json`.

### Stage 2: Embedding pipeline and vector store

**`src/catalog/embedder.py`**

Two functions:
- `build_embedding_inputs(catalog) -> list[str]` — For each market: `"{title} -- {subtitle}"`. If subtitle is empty or just "Yes"/"No", use title only.
- `embed_texts(texts, model="text-embedding-3-small", dimensions=512, batch_size=100) -> list[list[float]]` — Call OpenAI embeddings API in batches of 100. Log progress. Total cost for 5k markets: ~$0.005.

**`src/catalog/vector_store.py`**

Use FAISS, not raw numpy. FAISS `IndexFlatIP` gives SIMD-accelerated brute-force inner product search — same algorithm as numpy matmul but ~20x faster at this scale.

```python
import faiss
import numpy as np
import json
from pathlib import Path

class VectorStore:
    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        self.documents: list[dict] = []

    def add(self, embeddings: list[list[float]], metadata: list[dict]) -> None:
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)  # normalize for cosine similarity via inner product
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.documents = metadata

    def search(self, query_embedding: list[float], k: int = 30) -> list[dict]:
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({**self.documents[idx], "score": float(score)})
        return results

    def save(self, path: str = "data/vectors") -> None:
        faiss.write_index(self.index, f"{path}.faiss")
        Path(f"{path}_meta.json").write_text(json.dumps(self.documents))

    def load(self, path: str = "data/vectors") -> None:
        self.index = faiss.read_index(f"{path}.faiss")
        self.documents = json.loads(Path(f"{path}_meta.json").read_text())
```

### Stage 3: Context retrieval and LLM reranking

**`src/context/models.py`**

```python
from src.catalog.models import CatalogMarket

class CandidateMarket(CatalogMarket):
    similarity_score: float

class ContextMarket(CatalogMarket):
    similarity_score: float
    relevance_score: float   # 0-10, from LLM
    relationship: str        # LLM explanation of causal link
```

**`src/context/retriever.py`**

`retrieve_candidates(ticker, catalog, vector_store, k=30) -> list[CandidateMarket]`
1. Find focus market in catalog by ticker
2. Embed its title using OpenAI (same model as index)
3. Search vector_store for top k
4. Exclude the focus market itself
5. If ticker not in catalog (new/stale): embed ticker string directly as fallback, log warning

**`src/context/reranker.py`**

`rerank(focus, candidates, model, score_threshold=6.0) -> list[ContextMarket]`

LLM provider: **Groq** (Llama 3.3 70B). Groq's API is OpenAI-compatible:
```python
from groq import Groq
client = Groq()  # reads GROQ_API_KEY from env
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[...],
    response_format={"type": "json_object"},
    max_tokens=2000,
)
```

The reranking prompt must instruct the LLM to look for:
- Direct causal links (event A causes event B)
- Shared underlying drivers (both affected by same force)
- Conditional relationships (if A then B becomes more likely)
- Temporal dependencies (A must happen before B can happen)
- NOT just keyword matches

CRITICAL: Randomize candidate order before building the prompt. LLMs have a slight bias toward items listed first.

The `relationship` field matters — it feeds into the inference engine so it can focus on quantitative analysis, not re-deriving why each market matters.

Handle empty results gracefully — if LLM returns zero markets above threshold, return empty list.

### Stage 4: Structured inference engine

**`src/inference/models.py`**

```python
from pydantic import BaseModel
from src.insight.models import MarketSnapshot
from src.context.models import ContextMarket

class DerivedProbability(BaseModel):
    description: str    # e.g., "P(oil spike | iran strike)"
    value: float        # 0.0 to 1.0
    reasoning: str

class Mispricing(BaseModel):
    ticker: str
    title: str
    direction: str              # "overpriced" or "underpriced"
    current_implied_prob: float
    estimated_fair_prob: float
    reasoning: str

class Edge(BaseModel):
    ticker: str
    title: str
    side: str           # "yes" or "no"
    confidence: str     # "low", "medium", "high"
    thesis: str
    kelly_fraction: float  # quarter-Kelly position size (0.0 to 1.0)

class InferenceReport(BaseModel):
    focus_market: MarketSnapshot
    context_markets: list[ContextMarket]
    consistency_analysis: str
    derived_probabilities: list[DerivedProbability]
    detected_mispricings: list[Mispricing]
    suggested_edges: list[Edge]
```

**`src/inference/engine.py`**

`run_inference(snapshot, context, model) -> InferenceReport`

LLM provider: **Groq** (Llama 3.3 70B for speed, or use llama-3.1-405b-reasoning if available for stronger reasoning).

The inference prompt must request four types of analysis:
1. **Consistency**: Do prices violate probability axioms? P(by June) > P(by December) is impossible.
2. **Conditional probability**: Derive P(B|A) from priced markets. E.g., P(oil spike | Iran strike) = P(both)/P(strike).
3. **Cross-market divergence**: Do correlated markets agree? Rate cut at 70% but stocks haven't moved.
4. **Stale pricing**: Has one market stopped trading while related markets moved?

Include the `relationship` field from the reranker in each context market's prompt block — gives the model pre-computed causal links.

If context list is empty, fall back to single-market analysis (similar to existing `src/insight/generator.py`).

**Quarter-Kelly position sizing**: For each suggested edge, compute:
```
edge = estimated_fair_prob - current_implied_prob
if side == "no": edge = (1 - estimated_fair_prob) - (1 - current_implied_prob)
payout = 1.0 / current_implied_prob - 1.0
kelly = (estimated_fair_prob * payout - (1 - estimated_fair_prob)) / payout
quarter_kelly = kelly * 0.25
```
Clamp to [0.0, 0.25]. Negative kelly means no edge — don't suggest the trade.

### Stage 5: Integration and CLI

**`scripts/__init__.py`** — Empty.

**`scripts/build_index.py`**

```
Usage: python -m scripts.build_index

Steps:
1. Fetch catalog from Kalshi REST API
2. Save catalog to data/catalog.json
3. Build embedding inputs from catalog
4. Embed via OpenAI
5. Build VectorStore, add embeddings
6. Save VectorStore to data/vectors.faiss + data/vectors_meta.json
7. Log summary: N markets indexed
```

**`infer.py`** (root level)

```
Usage:
  python infer.py TICKER
  python infer.py TICKER --dry-run
  python infer.py TICKER --refresh-catalog
  python infer.py TICKER --no-redis

CLI flags:
  ticker (positional)     Market ticker to analyze. Falls back to KALSHI_INSIGHT_TICKER env var.
  --dry-run               Run retrieval + reranking only. Print context markets. Skip inference.
  --refresh-catalog       Rebuild catalog and embedding index before inference.
  --no-redis              Skip Redis snapshot; use catalog prices. For dev without ingestion running.

Error handling:
  No data/catalog.json    → Print "Run: python -m scripts.build_index" and exit 1
  No data/vectors.faiss   → Same
  Ticker not in catalog   → Log warning, embed ticker directly, proceed
  Redis unreachable       → If --no-redis: use catalog prices. Else: print error, suggest --no-redis
  LLM API error           → Retry once, then print raw exception
  Reranker returns 0      → Log warning, run inference with empty context (single-market mode)
```

## Conventions (follow existing codebase exactly)

- Modern type hints: `str | None`, `list[str]` (NOT `Optional`, NOT `List`)
- `@dataclass` with `field(default_factory=lambda: ...)` for config classes
- `Pydantic BaseModel` for all data schemas
- Structured logging via `get_logger(__name__)` — NEVER use `print()`
- `asyncio` for all I/O
- Entry points pattern: `load_dotenv() → config → asyncio.run(main())`
- All `__init__.py` files are empty. Use explicit imports.
- Wrap HTTP calls with `retry_with_backoff()` from `src/utils/retry.py`

## Environment variables required

```
# Existing (already in .env)
KALSHI_API_KEY=...          # For WebSocket auth (not needed for REST read-only)
REDIS_HOST=...
REDIS_PORT=...

# New (add to .env)
OPENAI_API_KEY=...          # For embeddings only
GROQ_API_KEY=...            # For reranker + inference LLM calls
KALSHI_INSIGHT_TICKER=...   # Default ticker for infer.py (optional)
```

## Dependencies to add to requirements.txt

```
faiss-cpu>=1.7.4
groq>=0.4.0
orjson>=3.9.0
aiohttp>=3.9.0
numpy>=1.26.0
```

## Key design decisions and rationale

**Why FAISS over raw numpy?** Same brute-force algorithm but SIMD-accelerated. ~20x faster at 5k vectors. Single `pip install`, no server. Industry standard (Meta, used by Zilliz/Pinecone under the hood).

**Why Groq over OpenAI for LLM calls?** Groq's LPU delivers 120ms median TTFT and 330 tok/s on Llama 3.3 70B — 3-5x faster than GPT-4o. OpenAI-compatible API, so the code is provider-agnostic (just change the client). Reranker call: ~800ms. Inference call: ~1.2s. Total hot-path latency: ~2.5s vs ~12s+ with GPT-4o.

**Why event-driven pipeline over monolith?** Each stage produces a serialized artifact (JSON, FAISS index, Pydantic model list). Stages are independently testable, cacheable, and swappable. Same pattern as quant signal pipelines: data → features → alpha → risk → execution.

**Why two-stage retrieval (embed + LLM rerank)?** Embeddings are fast but shallow — they match words, not causal chains. "Iran" finds "Iran" but not "oil prices." The LLM is slow but can reason: "a military strike would disrupt oil supply through the Strait of Hormuz." Using both gives you speed (FAISS narrows 5k to 30 in microseconds) and depth (LLM reasons over 30 items in ~800ms).

**Why quarter-Kelly for position sizing?** Full Kelly maximizes long-run growth but assumes perfect probability estimates. Your estimates come from an LLM — they have uncertainty. Quarter-Kelly captures most of the compounding benefit while reducing variance by 75%. This is the professional standard in prediction market trading.

## Acceptance criteria (definition of done)

- [ ] `python -m scripts.build_index` produces `data/catalog.json` + `data/vectors.faiss` + `data/vectors_meta.json`
- [ ] Catalog contains all active Kalshi markets (expect 1000-5000) with non-empty titles
- [ ] `python infer.py TICKER --dry-run` shows context markets with scores and relationships
- [ ] `python infer.py TICKER --no-redis` produces a complete InferenceReport as JSON
- [ ] `python infer.py TICKER` (with Redis running) uses live snapshot data
- [ ] All error cases produce clear messages, not stack traces
- [ ] Reranker surfaces causal links (not just keyword matches): e.g., Iran strike → oil prices
- [ ] InferenceReport includes specific price estimates for mispricings, not generic "might be mispriced"
- [ ] Suggested edges include quarter-Kelly position sizes
- [ ] End-to-end hot-path latency < 3 seconds on Groq