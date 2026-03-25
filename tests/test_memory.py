"""Tests for ganesha.memory — MemoryStore with TTL-based expiry."""

import time
import pytest

from ganesha.memory import MemoryStore


class TestMemoryStore:
    def test_store_and_recall(self):
        store = MemoryStore()
        store.store("name", "Ganesha")
        assert store.recall("name") == "Ganesha"

    def test_recall_missing_key(self):
        store = MemoryStore()
        assert store.recall("nope") is None
        assert store.recall("nope", default="fallback") == "fallback"

    def test_forget(self):
        store = MemoryStore()
        store.store("x", 42)
        assert store.forget("x") is True
        assert store.recall("x") is None
        assert store.forget("x") is False

    def test_ttl_expiry(self):
        store = MemoryStore()
        store.store("temp", "value", ttl=0.05)  # 50ms
        assert store.recall("temp") == "value"
        time.sleep(0.1)
        assert store.recall("temp") is None

    def test_prune_expired(self):
        store = MemoryStore()
        store.store("a", 1, ttl=0.05)
        store.store("b", 2, ttl=0.05)
        store.store("c", 3)  # no expiry
        time.sleep(0.1)
        removed = store.prune_expired()
        assert removed == 2
        assert store.recall("c") == 3

    def test_list_keys(self):
        store = MemoryStore()
        store.store("alpha", 1)
        store.store("beta", 2)
        keys = store.list_keys()
        assert set(keys) == {"alpha", "beta"}

    def test_list_keys_excludes_expired(self):
        store = MemoryStore()
        store.store("alive", 1)
        store.store("dead", 2, ttl=0.05)
        time.sleep(0.1)
        keys = store.list_keys()
        assert keys == ["alive"]

    def test_contains(self):
        store = MemoryStore()
        store.store("x", 1)
        assert "x" in store
        assert "y" not in store

    def test_default_ttl(self):
        store = MemoryStore(default_ttl=0.05)
        store.store("item", "val")
        assert store.recall("item") == "val"
        time.sleep(0.1)
        assert store.recall("item") is None

    def test_clear(self):
        store = MemoryStore()
        store.store("a", 1)
        store.store("b", 2)
        store.clear()
        assert len(store) == 0

    def test_empty_key_raises(self):
        store = MemoryStore()
        with pytest.raises(ValueError):
            store.store("", "value")
