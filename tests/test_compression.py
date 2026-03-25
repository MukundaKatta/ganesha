"""Tests for ganesha.compression — TruncateStrategy, SummarizeStrategy, PriorityStrategy."""

import pytest

from ganesha.core import Message, TokenCounter
from ganesha.compression import (
    Compressor,
    TruncateStrategy,
    SummarizeStrategy,
    PriorityStrategy,
)


def _make_messages(count, role="user", words=10):
    """Helper to create a list of messages."""
    return [
        Message(role=role, content=" ".join(f"word{j}" for j in range(words)))
        for _ in range(count)
    ]


# ---------------------------------------------------------------------------
# TruncateStrategy
# ---------------------------------------------------------------------------

class TestTruncateStrategy:
    def test_no_truncation_needed(self):
        strategy = TruncateStrategy()
        msgs = _make_messages(2, words=3)
        result = strategy.compress(msgs, target_tokens=5000)
        assert len(result) == 2

    def test_truncates_oldest(self):
        strategy = TruncateStrategy()
        msgs = _make_messages(10, words=10)
        result = strategy.compress(msgs, target_tokens=20)
        counter = TokenCounter()
        total = counter.count_messages(result)
        assert total <= 20
        assert len(result) < 10

    def test_preserves_system(self):
        strategy = TruncateStrategy()
        msgs = [
            Message(role="system", content="You are helpful"),
        ] + _make_messages(10, words=10)
        result = strategy.compress(msgs, target_tokens=20)
        assert any(m.is_system for m in result)


# ---------------------------------------------------------------------------
# SummarizeStrategy
# ---------------------------------------------------------------------------

class TestSummarizeStrategy:
    def test_merges_consecutive_same_role(self):
        strategy = SummarizeStrategy()
        msgs = [
            Message(role="user", content="Hello"),
            Message(role="user", content="How are you?"),
            Message(role="assistant", content="Fine!"),
        ]
        result = strategy.compress(msgs, target_tokens=5000)
        # Two consecutive user messages should be merged
        assert len(result) == 2
        assert "Hello" in result[0].content
        assert "How are you?" in result[0].content

    def test_no_merge_different_roles(self):
        strategy = SummarizeStrategy()
        msgs = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
            Message(role="user", content="Bye"),
        ]
        result = strategy.compress(msgs, target_tokens=5000)
        assert len(result) == 3

    def test_truncation_after_merge(self):
        strategy = SummarizeStrategy()
        msgs = _make_messages(20, words=15)
        result = strategy.compress(msgs, target_tokens=30)
        counter = TokenCounter()
        total = counter.count_messages(result)
        assert total <= 30


# ---------------------------------------------------------------------------
# PriorityStrategy
# ---------------------------------------------------------------------------

class TestPriorityStrategy:
    def test_keeps_system_and_recent_user(self):
        strategy = PriorityStrategy(keep_recent=2)
        msgs = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="Old question"),
            Message(role="assistant", content="Old answer"),
            Message(role="user", content="Recent question 1"),
            Message(role="assistant", content="Recent answer 1"),
            Message(role="user", content="Recent question 2"),
        ]
        result = strategy.compress(msgs, target_tokens=5000)
        roles = [m.role for m in result]
        assert "system" in roles
        # Should keep the 2 most recent user messages
        user_contents = [m.content for m in result if m.is_user]
        assert "Recent question 1" in user_contents
        assert "Recent question 2" in user_contents

    def test_truncates_assistant_when_over_budget(self):
        strategy = PriorityStrategy(keep_recent=1)
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="question"),
            Message(role="assistant", content=" ".join(["word"] * 100)),
        ]
        result = strategy.compress(msgs, target_tokens=20)
        counter = TokenCounter()
        total = counter.count_messages(result)
        assert total <= 20

    def test_invalid_keep_recent(self):
        with pytest.raises(ValueError):
            PriorityStrategy(keep_recent=0)

    def test_protocol_compliance(self):
        """All strategies satisfy the Compressor protocol."""
        assert isinstance(TruncateStrategy(), Compressor)
        assert isinstance(SummarizeStrategy(), Compressor)
        assert isinstance(PriorityStrategy(), Compressor)
