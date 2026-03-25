"""
Key-value memory store with TTL-based expiry for multi-agent systems.

:class:`MemoryStore` provides a simple but effective way for agents to
persist and recall short-lived facts, intermediate results, or shared
state across conversation turns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class _MemoryEntry:
    """Internal wrapper around a stored value."""
    value: Any
    created_at: float = field(default_factory=time.time)
    ttl: Optional[float] = None  # seconds; None means no expiry

    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    @property
    def expires_at(self) -> Optional[float]:
        if self.ttl is None:
            return None
        return self.created_at + self.ttl

    @property
    def remaining_ttl(self) -> Optional[float]:
        if self.ttl is None:
            return None
        remaining = (self.created_at + self.ttl) - time.time()
        return max(0.0, remaining)


class MemoryStore:
    """A key-value memory store with optional time-to-live (TTL) expiry.

    This is designed for lightweight agent memory -- storing facts,
    tool results, or intermediate computations that should be accessible
    across turns but may expire after a configurable duration.

    Parameters:
        default_ttl: Default TTL in seconds for new entries. ``None``
            means entries never expire unless an explicit TTL is provided.
    """

    def __init__(self, default_ttl: Optional[float] = None) -> None:
        self._store: Dict[str, _MemoryEntry] = {}
        self._default_ttl = default_ttl

    # -- Core operations ---------------------------------------------------

    def store(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store a value under *key*.

        Args:
            key: A string key.
            value: Any Python value.
            ttl: Time-to-live in seconds. If ``None``, uses the store
                default. Pass ``0`` to explicitly disable expiry for this
                entry even when a default is set.
        """
        if not key:
            raise ValueError("key must be a non-empty string")
        effective_ttl = ttl if ttl is not None else self._default_ttl
        # A ttl of 0 means "never expire"
        if effective_ttl is not None and effective_ttl <= 0:
            effective_ttl = None
        self._store[key] = _MemoryEntry(value=value, ttl=effective_ttl)

    def recall(self, key: str, default: Any = None) -> Any:
        """Retrieve the value for *key*, or *default* if missing or expired."""
        entry = self._store.get(key)
        if entry is None:
            return default
        if entry.is_expired:
            del self._store[key]
            return default
        return entry.value

    def forget(self, key: str) -> bool:
        """Remove *key* from the store. Returns ``True`` if the key existed."""
        if key in self._store:
            del self._store[key]
            return True
        return False

    # -- Inspection --------------------------------------------------------

    def list_keys(self, include_expired: bool = False) -> List[str]:
        """Return a list of stored keys.

        By default, expired keys are excluded (and cleaned up).
        """
        if include_expired:
            return list(self._store.keys())

        valid: List[str] = []
        expired_keys: List[str] = []
        for k, entry in self._store.items():
            if entry.is_expired:
                expired_keys.append(k)
            else:
                valid.append(k)
        for k in expired_keys:
            del self._store[k]
        return valid

    def __len__(self) -> int:
        """Return the number of stored entries (including expired)."""
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        entry = self._store.get(key)
        if entry is None:
            return False
        if entry.is_expired:
            del self._store[key]
            return False
        return True

    # -- Bulk operations ---------------------------------------------------

    def prune_expired(self) -> int:
        """Remove all expired entries. Returns the number removed."""
        expired = [k for k, e in self._store.items() if e.is_expired]
        for k in expired:
            del self._store[k]
        return len(expired)

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()

    # -- Representation ----------------------------------------------------

    def __repr__(self) -> str:
        return f"MemoryStore(entries={len(self._store)}, default_ttl={self._default_ttl})"
