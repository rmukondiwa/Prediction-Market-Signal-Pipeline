from __future__ import annotations

from src.backtest.cache import InferenceCache


def test_cache_miss_returns_none(tmp_cache_dir):
    cache = InferenceCache(tmp_cache_dir)
    assert cache.get("p", "m", 0.0, "v1") is None
    assert cache.misses == 1
    assert cache.hits == 0


def test_cache_put_then_get(tmp_cache_dir):
    cache = InferenceCache(tmp_cache_dir)
    cache.put("p", "m", 0.0, "v1", {"answer": 42})
    response = cache.get("p", "m", 0.0, "v1")
    assert response == {"answer": 42}
    assert cache.hits == 1


def test_cache_isolation_by_prompt(tmp_cache_dir):
    cache = InferenceCache(tmp_cache_dir)
    cache.put("p1", "m", 0.0, "v1", {"a": 1})
    cache.put("p2", "m", 0.0, "v1", {"a": 2})
    assert cache.get("p1", "m", 0.0, "v1") == {"a": 1}
    assert cache.get("p2", "m", 0.0, "v1") == {"a": 2}


def test_cache_isolation_by_prompt_version(tmp_cache_dir):
    """Bumping PROMPT_VERSION must invalidate prior entries."""
    cache = InferenceCache(tmp_cache_dir)
    cache.put("p", "m", 0.0, "v1", {"old": True})
    assert cache.get("p", "m", 0.0, "v2") is None  # different version → miss
    assert cache.get("p", "m", 0.0, "v1") == {"old": True}


def test_cache_isolation_by_temperature(tmp_cache_dir):
    cache = InferenceCache(tmp_cache_dir)
    cache.put("p", "m", 0.0, "v1", {"deterministic": True})
    assert cache.get("p", "m", 0.7, "v1") is None


def test_cache_persists_across_instances(tmp_cache_dir):
    """Two InferenceCache instances pointing at the same dir should see each
    other's writes — the cache is the dir, not the instance."""
    c1 = InferenceCache(tmp_cache_dir)
    c1.put("p", "m", 0.0, "v1", {"x": 1})

    c2 = InferenceCache(tmp_cache_dir)
    assert c2.get("p", "m", 0.0, "v1") == {"x": 1}


def test_cache_corrupt_entry_treated_as_miss(tmp_cache_dir):
    """A garbage file should not crash — just count as a miss."""
    cache = InferenceCache(tmp_cache_dir)
    cache.put("p", "m", 0.0, "v1", {"x": 1})
    # Find the file we just wrote and corrupt it
    files = list(tmp_cache_dir.rglob("*.json"))
    assert len(files) == 1
    files[0].write_text("not valid json")
    assert cache.get("p", "m", 0.0, "v1") is None


def test_cache_stats(tmp_cache_dir):
    cache = InferenceCache(tmp_cache_dir)
    cache.get("p1", "m", 0.0, "v1")  # miss
    cache.put("p1", "m", 0.0, "v1", {})
    cache.get("p1", "m", 0.0, "v1")  # hit
    cache.get("p1", "m", 0.0, "v1")  # hit
    stats = cache.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["total"] == 3
    assert stats["hit_rate"] == 2 / 3
