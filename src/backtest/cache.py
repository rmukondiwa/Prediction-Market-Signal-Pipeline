"""
Disk-backed inference cache for reproducible backtests.

The backtester drives identical LLM call sequences across many signal-model
runs; the cache makes that cheap. Live trading does NOT pass a cache — the
asymmetric failure mode of stale cache hits in production is not worth the
marginal savings, and live snapshots change every WS tick anyway so the
cache rarely hits.

Key inputs (sha256-hashed):
    prompt + model + temperature + prompt_version

Layout on disk: cache_dir/<first 2 hex chars>/<full hash>.json (sharded so
no single directory accumulates 100k+ files).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


class InferenceCache:
    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def _key(self, prompt: str, model: str, temperature: float, prompt_version: str) -> str:
        payload = json.dumps(
            {"p": prompt, "m": model, "t": float(temperature), "v": prompt_version},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        shard = key[:2]
        return self.dir / shard / f"{key}.json"

    def get(self, prompt: str, model: str, temperature: float, prompt_version: str) -> dict | None:
        path = self._path(self._key(prompt, model, temperature, prompt_version))
        if not path.exists():
            self._misses += 1
            return None
        try:
            entry = json.loads(path.read_text())
            self._hits += 1
            return entry.get("response")
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cache entry unreadable, treating as miss",
                           extra={"path": str(path), "error": str(exc)})
            self._misses += 1
            return None

    def put(
        self,
        prompt: str,
        model: str,
        temperature: float,
        prompt_version: str,
        response: dict,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
    ) -> None:
        key = self._key(prompt, model, temperature, prompt_version)
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry: dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "temperature": float(temperature),
            "prompt_version": prompt_version,
            "response": response,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(entry))

    def stats(self) -> dict[str, int | float]:
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total > 0 else 0.0
        return {"hits": self._hits, "misses": self._misses, "total": total, "hit_rate": hit_rate}
