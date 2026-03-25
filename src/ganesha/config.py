"""
Configuration management for Ganesha.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class GaneshaConfig:
    """Central configuration for a Ganesha instance.

    Attributes:
        max_tokens: Default token budget for context windows.
        tokens_per_word: Ratio used by the heuristic token counter.
        default_prune_strategy: Strategy name used when pruning without
            an explicit strategy argument.
        sliding_window_messages: Default sliding window message limit.
        sliding_window_tokens: Default sliding window token limit.
        preserve_system_messages: Whether system messages are protected
            from pruning and sliding window eviction.
        memory_default_ttl: Default TTL (seconds) for the memory store.
    """

    max_tokens: int = 4096
    tokens_per_word: float = 1.3
    default_prune_strategy: str = "truncate"
    sliding_window_messages: Optional[int] = None
    sliding_window_tokens: Optional[int] = None
    preserve_system_messages: bool = True
    memory_default_ttl: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "tokens_per_word": self.tokens_per_word,
            "default_prune_strategy": self.default_prune_strategy,
            "sliding_window_messages": self.sliding_window_messages,
            "sliding_window_tokens": self.sliding_window_tokens,
            "preserve_system_messages": self.preserve_system_messages,
            "memory_default_ttl": self.memory_default_ttl,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GaneshaConfig":
        known_fields = {
            "max_tokens", "tokens_per_word", "default_prune_strategy",
            "sliding_window_messages", "sliding_window_tokens",
            "preserve_system_messages", "memory_default_ttl", "extra",
        }
        kwargs = {k: v for k, v in data.items() if k in known_fields}
        return cls(**kwargs)
