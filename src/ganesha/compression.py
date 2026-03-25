"""
Compression strategies for conversation context.

Each strategy implements the :class:`Compressor` protocol which requires a
``compress(messages, target_tokens)`` method returning a reduced message list.
"""

from __future__ import annotations

import math
from typing import List, Protocol, runtime_checkable

from ganesha.core import Message, TokenCounter


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Compressor(Protocol):
    """Protocol that all compression strategies must satisfy."""

    def compress(self, messages: List[Message], target_tokens: int) -> List[Message]:
        """Return a compressed list of messages fitting within *target_tokens*."""
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Truncate strategy
# ---------------------------------------------------------------------------

class TruncateStrategy:
    """Drop the oldest non-system messages until the budget is met.

    System messages are always preserved.  Non-system messages are removed
    front-to-back (oldest first) until the total token count is within
    *target_tokens*.
    """

    def compress(self, messages: List[Message], target_tokens: int) -> List[Message]:
        counter = TokenCounter()

        system: List[Message] = [m for m in messages if m.is_system]
        others: List[Message] = [m for m in messages if not m.is_system]

        total = counter.count_messages(messages)
        if total <= target_tokens:
            return list(messages)

        # Drop oldest non-system messages one at a time
        while others and counter.count_messages(system + others) > target_tokens:
            others.pop(0)

        return system + others


# ---------------------------------------------------------------------------
# Summarize strategy
# ---------------------------------------------------------------------------

class SummarizeStrategy:
    """Merge consecutive messages with the same role into a single message.

    After merging, if the result still exceeds *target_tokens*, the merged
    content is truncated word-by-word from the beginning until it fits.

    This is a *local* summarisation heuristic that does not call an LLM.
    """

    def compress(self, messages: List[Message], target_tokens: int) -> List[Message]:
        if not messages:
            return []

        counter = TokenCounter()

        # Phase 1: merge consecutive same-role messages
        merged: List[Message] = []
        for msg in messages:
            if merged and merged[-1].role == msg.role and not msg.is_system:
                combined = merged[-1].content + "\n" + msg.content
                merged[-1] = Message(
                    role=msg.role,
                    content=combined,
                    timestamp=msg.timestamp,
                    token_count=counter.count(combined),
                    metadata=msg.metadata,
                )
            else:
                merged.append(Message(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp,
                    token_count=msg.token_count,
                    metadata=msg.metadata,
                ))

        total = counter.count_messages(merged)
        if total <= target_tokens:
            return merged

        # Phase 2: truncate oldest non-system merged messages
        system = [m for m in merged if m.is_system]
        others = [m for m in merged if not m.is_system]

        while others and counter.count_messages(system + others) > target_tokens:
            oldest = others[0]
            words = oldest.content.split()
            if len(words) <= 1:
                others.pop(0)
                continue
            # Remove first half of words
            half = max(1, len(words) // 2)
            trimmed = " ".join(words[half:])
            others[0] = Message(
                role=oldest.role,
                content=trimmed,
                timestamp=oldest.timestamp,
                token_count=counter.count(trimmed),
                metadata=oldest.metadata,
            )
            # If still empty after trimming, drop entirely
            if not trimmed.strip():
                others.pop(0)

        return system + others


# ---------------------------------------------------------------------------
# Priority strategy
# ---------------------------------------------------------------------------

class PriorityStrategy:
    """Keep system messages and recent user messages; compress assistant responses.

    Priority ordering:
      1. System messages (always kept in full).
      2. The most recent *keep_recent* user messages.
      3. The most recent assistant message.
      4. Remaining messages are dropped.

    If the result still exceeds *target_tokens* after selection, assistant
    messages are truncated to fit.
    """

    def __init__(self, keep_recent: int = 3) -> None:
        if keep_recent < 1:
            raise ValueError("keep_recent must be >= 1")
        self._keep_recent = keep_recent

    def compress(self, messages: List[Message], target_tokens: int) -> List[Message]:
        counter = TokenCounter()

        system: List[Message] = [m for m in messages if m.is_system]
        user_msgs: List[Message] = [m for m in messages if m.is_user]
        assistant_msgs: List[Message] = [m for m in messages if m.is_assistant]
        other_msgs: List[Message] = [
            m for m in messages
            if not m.is_system and not m.is_user and not m.is_assistant
        ]

        # Keep recent user messages
        kept_user = user_msgs[-self._keep_recent:]

        # Keep the last assistant message
        kept_assistant = assistant_msgs[-1:] if assistant_msgs else []

        result = system + kept_user + kept_assistant

        total = counter.count_messages(result)
        if total <= target_tokens:
            return result

        # Truncate assistant messages to fit
        system_user_tokens = counter.count_messages(system + kept_user)
        assistant_budget = target_tokens - system_user_tokens

        if assistant_budget <= 0:
            return system + kept_user

        trimmed_asst: List[Message] = []
        for m in kept_assistant:
            if m.token_count <= assistant_budget:
                trimmed_asst.append(m)
                assistant_budget -= m.token_count
            else:
                words = m.content.split()
                # Estimate how many words fit
                target_words = max(1, int(assistant_budget / counter.ratio))
                truncated_content = " ".join(words[:target_words])
                trimmed_asst.append(Message(
                    role=m.role,
                    content=truncated_content,
                    timestamp=m.timestamp,
                    token_count=counter.count(truncated_content),
                    metadata=m.metadata,
                ))
                break

        return system + kept_user + trimmed_asst
