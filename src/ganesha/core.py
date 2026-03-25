"""
Core context management engine for Ganesha.

Provides the ContextManager, TokenCounter, Message dataclass, and
SlidingWindow for managing conversation history with token budgets.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

class TokenCounter:
    """Estimates token counts from text using a word-based approximation.

    The default ratio of 1.3 tokens per word is a reasonable heuristic for
    English text processed by most modern tokenizers.  A custom ratio can be
    supplied at construction time.
    """

    def __init__(self, tokens_per_word: float = 1.3) -> None:
        if tokens_per_word <= 0:
            raise ValueError("tokens_per_word must be positive")
        self._ratio = tokens_per_word

    def count(self, text: str) -> int:
        """Return the estimated number of tokens in *text*."""
        if not text:
            return 0
        words = text.split()
        return max(1, math.ceil(len(words) * self._ratio))

    def count_messages(self, messages: Sequence["Message"]) -> int:
        """Return the total estimated token count across *messages*."""
        return sum(m.token_count for m in messages)

    @property
    def ratio(self) -> float:
        return self._ratio


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

class Role(str, Enum):
    """Standard chat roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A single message in a conversation.

    Attributes:
        role: The chat role (system, user, assistant, tool).
        content: The textual content of the message.
        timestamp: Unix timestamp of when the message was created.
        token_count: Estimated token count (auto-calculated if not provided).
        metadata: Arbitrary extra data attached to the message.
    """

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_count == 0 and self.content:
            self.token_count = TokenCounter().count(self.content)

    # Convenience helpers --------------------------------------------------

    @property
    def is_system(self) -> bool:
        return self.role == Role.SYSTEM.value or self.role == "system"

    @property
    def is_user(self) -> bool:
        return self.role == Role.USER.value or self.role == "user"

    @property
    def is_assistant(self) -> bool:
        return self.role == Role.ASSISTANT.value or self.role == "assistant"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the message to a plain dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Deserialize a message from a plain dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            token_count=data.get("token_count", 0),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------

class SlidingWindow:
    """Keeps only the most recent messages within a count *or* token limit.

    Parameters:
        max_messages: Maximum number of messages to retain (``None`` for
            unlimited).
        max_tokens: Maximum total token count to retain (``None`` for
            unlimited).
        preserve_system: When ``True`` (default), system messages are never
            evicted by the sliding window.
    """

    def __init__(
        self,
        max_messages: Optional[int] = None,
        max_tokens: Optional[int] = None,
        preserve_system: bool = True,
    ) -> None:
        if max_messages is not None and max_messages < 1:
            raise ValueError("max_messages must be >= 1")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self._max_messages = max_messages
        self._max_tokens = max_tokens
        self._preserve_system = preserve_system

    @property
    def max_messages(self) -> Optional[int]:
        return self._max_messages

    @property
    def max_tokens(self) -> Optional[int]:
        return self._max_tokens

    # ------------------------------------------------------------------

    def apply(self, messages: List[Message]) -> List[Message]:
        """Return a trimmed copy of *messages* that fits within the window."""
        if not messages:
            return []

        system_msgs: List[Message] = []
        non_system: List[Message] = []
        for m in messages:
            if self._preserve_system and m.is_system:
                system_msgs.append(m)
            else:
                non_system.append(m)

        # --- enforce max_messages ---
        if self._max_messages is not None:
            available = max(0, self._max_messages - len(system_msgs))
            non_system = non_system[-available:] if available > 0 else []

        # --- enforce max_tokens ---
        if self._max_tokens is not None:
            system_tokens = sum(m.token_count for m in system_msgs)
            budget = self._max_tokens - system_tokens
            if budget <= 0:
                return system_msgs

            kept: List[Message] = []
            running = 0
            for m in reversed(non_system):
                if running + m.token_count > budget:
                    break
                kept.append(m)
                running += m.token_count
            non_system = list(reversed(kept))

        return system_msgs + non_system


# ---------------------------------------------------------------------------
# Pruning strategies (string enum for use with ContextManager.prune)
# ---------------------------------------------------------------------------

class PruneStrategy(str, Enum):
    """Built-in pruning strategies."""
    TRUNCATE = "truncate"
    SUMMARIZE = "summarize"
    PRIORITY = "priority"


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

class ContextManager:
    """Manages conversation history with a token budget.

    The ``ContextManager`` is the primary entry-point for Ganesha.  It stores
    an ordered list of :class:`Message` objects and provides helpers for
    retrieving context within a token budget, pruning old messages, and
    compressing the conversation.

    Parameters:
        max_tokens: The token budget for the context window.
        counter: A :class:`TokenCounter` instance (default heuristic used
            if ``None``).
        sliding_window: An optional :class:`SlidingWindow` applied
            automatically on every :meth:`get_context` call.
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        counter: Optional[TokenCounter] = None,
        sliding_window: Optional[SlidingWindow] = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self._max_tokens = max_tokens
        self._counter = counter or TokenCounter()
        self._window = sliding_window
        self._messages: List[Message] = []

    # -- Properties --------------------------------------------------------

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def messages(self) -> List[Message]:
        """Return a shallow copy of all stored messages."""
        return list(self._messages)

    @property
    def total_tokens(self) -> int:
        """Return the total estimated tokens across all stored messages."""
        return self._counter.count_messages(self._messages)

    @property
    def message_count(self) -> int:
        return len(self._messages)

    # -- Mutation ----------------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Append a new message and return it."""
        msg = Message(
            role=role,
            content=content,
            token_count=self._counter.count(content),
            metadata=metadata or {},
        )
        self._messages.append(msg)
        return msg

    def add(self, message: Message) -> None:
        """Append an existing :class:`Message` object."""
        self._messages.append(message)

    def clear(self) -> None:
        """Remove all messages."""
        self._messages.clear()

    # -- Retrieval ---------------------------------------------------------

    def get_context(
        self,
        max_tokens: Optional[int] = None,
    ) -> List[Message]:
        """Return messages that fit within the token budget.

        If a :class:`SlidingWindow` was configured, it is applied first.
        Then the result is trimmed to *max_tokens* (falling back to the
        instance default) by dropping the oldest non-system messages.
        """
        budget = max_tokens if max_tokens is not None else self._max_tokens

        msgs = list(self._messages)
        if self._window is not None:
            msgs = self._window.apply(msgs)

        # Trim to budget (keep system, drop oldest non-system first)
        system = [m for m in msgs if m.is_system]
        others = [m for m in msgs if not m.is_system]

        system_cost = sum(m.token_count for m in system)
        remaining = budget - system_cost
        if remaining <= 0:
            return system

        kept: List[Message] = []
        running = 0
        for m in reversed(others):
            if running + m.token_count > remaining:
                break
            kept.append(m)
            running += m.token_count
        kept.reverse()

        return system + kept

    # -- Pruning -----------------------------------------------------------

    def prune(self, strategy: str = "truncate", target_tokens: Optional[int] = None) -> int:
        """Prune messages in-place using the named *strategy*.

        Returns the number of messages removed.
        """
        from ganesha.compression import (
            TruncateStrategy,
            SummarizeStrategy,
            PriorityStrategy,
        )

        target = target_tokens if target_tokens is not None else self._max_tokens

        strategy_map = {
            "truncate": TruncateStrategy,
            "summarize": SummarizeStrategy,
            "priority": PriorityStrategy,
        }
        cls = strategy_map.get(strategy)
        if cls is None:
            raise ValueError(
                f"Unknown strategy {strategy!r}. "
                f"Choose from {list(strategy_map)}"
            )

        compressor = cls()
        before = len(self._messages)
        self._messages = compressor.compress(list(self._messages), target)
        return before - len(self._messages)

    # -- Compression shortcut ----------------------------------------------

    def compress(self, target_tokens: Optional[int] = None) -> int:
        """Compress using the default summarize strategy.

        Returns the number of messages removed.
        """
        return self.prune(strategy="summarize", target_tokens=target_tokens)

    # -- Stats -------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return a summary of the current context state."""
        by_role: Dict[str, int] = {}
        for m in self._messages:
            by_role[m.role] = by_role.get(m.role, 0) + 1

        return {
            "message_count": len(self._messages),
            "total_tokens": self.total_tokens,
            "max_tokens": self._max_tokens,
            "utilization": round(self.total_tokens / self._max_tokens, 4)
            if self._max_tokens
            else 0.0,
            "messages_by_role": by_role,
        }

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_tokens": self._max_tokens,
            "messages": [m.to_dict() for m in self._messages],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextManager":
        mgr = cls(max_tokens=data["max_tokens"])
        for md in data["messages"]:
            mgr.add(Message.from_dict(md))
        return mgr

    def __repr__(self) -> str:
        return (
            f"ContextManager(messages={len(self._messages)}, "
            f"tokens={self.total_tokens}/{self._max_tokens})"
        )
