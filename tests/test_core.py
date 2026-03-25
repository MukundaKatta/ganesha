"""Tests for ganesha.core — ContextManager, TokenCounter, Message, SlidingWindow."""

import time
import pytest

from ganesha.core import ContextManager, Message, SlidingWindow, TokenCounter


# ---------------------------------------------------------------------------
# TokenCounter
# ---------------------------------------------------------------------------

class TestTokenCounter:
    def test_empty_string(self):
        c = TokenCounter()
        assert c.count("") == 0

    def test_single_word(self):
        c = TokenCounter()
        assert c.count("hello") >= 1

    def test_multi_word(self):
        c = TokenCounter()
        count = c.count("hello world foo bar")
        # 4 words * 1.3 = 5.2 -> ceil = 6
        assert count == 6

    def test_custom_ratio(self):
        c = TokenCounter(tokens_per_word=2.0)
        assert c.count("one two three") == 6  # 3 * 2.0 = 6

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError):
            TokenCounter(tokens_per_word=0)

    def test_count_messages(self):
        c = TokenCounter()
        msgs = [
            Message(role="user", content="hello world"),
            Message(role="assistant", content="hi there friend"),
        ]
        total = c.count_messages(msgs)
        assert total == msgs[0].token_count + msgs[1].token_count


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

class TestMessage:
    def test_auto_token_count(self):
        m = Message(role="user", content="one two three four five")
        assert m.token_count > 0

    def test_role_helpers(self):
        assert Message(role="system", content="x").is_system
        assert Message(role="user", content="x").is_user
        assert Message(role="assistant", content="x").is_assistant

    def test_round_trip_dict(self):
        m = Message(role="user", content="hello", metadata={"key": "val"})
        d = m.to_dict()
        m2 = Message.from_dict(d)
        assert m2.role == m.role
        assert m2.content == m.content
        assert m2.metadata == m.metadata


# ---------------------------------------------------------------------------
# SlidingWindow
# ---------------------------------------------------------------------------

class TestSlidingWindow:
    def test_max_messages(self):
        window = SlidingWindow(max_messages=2)
        msgs = [
            Message(role="user", content="one"),
            Message(role="assistant", content="two"),
            Message(role="user", content="three"),
        ]
        result = window.apply(msgs)
        assert len(result) == 2
        assert result[0].content == "two"
        assert result[1].content == "three"

    def test_max_messages_preserves_system(self):
        window = SlidingWindow(max_messages=2)
        msgs = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="one"),
            Message(role="assistant", content="two"),
            Message(role="user", content="three"),
        ]
        result = window.apply(msgs)
        # system + 1 recent non-system (max_messages=2, system takes 1 slot)
        assert any(m.is_system for m in result)
        assert len(result) == 2

    def test_max_tokens(self):
        window = SlidingWindow(max_tokens=10)
        msgs = [
            Message(role="user", content="a " * 20),   # ~26 tokens
            Message(role="user", content="hello"),      # ~2 tokens
        ]
        result = window.apply(msgs)
        assert len(result) == 1
        assert result[0].content == "hello"

    def test_empty_input(self):
        window = SlidingWindow(max_messages=5)
        assert window.apply([]) == []

    def test_invalid_max_messages(self):
        with pytest.raises(ValueError):
            SlidingWindow(max_messages=0)


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_add_message(self):
        mgr = ContextManager(max_tokens=4096)
        msg = mgr.add_message("user", "Hello!")
        assert mgr.message_count == 1
        assert msg.role == "user"

    def test_get_context_within_budget(self):
        mgr = ContextManager(max_tokens=4096)
        mgr.add_message("user", "Hi")
        mgr.add_message("assistant", "Hello!")
        ctx = mgr.get_context()
        assert len(ctx) == 2

    def test_get_context_respects_budget(self):
        mgr = ContextManager(max_tokens=10)
        for i in range(20):
            mgr.add_message("user", f"Message number {i} with some extra words here")
        ctx = mgr.get_context()
        total = sum(m.token_count for m in ctx)
        assert total <= 10

    def test_get_context_preserves_system(self):
        mgr = ContextManager(max_tokens=20)
        mgr.add_message("system", "Be helpful")
        for i in range(10):
            mgr.add_message("user", f"Message {i} with more words to fill tokens")
        ctx = mgr.get_context()
        assert any(m.is_system for m in ctx)

    def test_prune_truncate(self):
        mgr = ContextManager(max_tokens=4096)
        for i in range(10):
            mgr.add_message("user", f"Message {i} with a bunch of words filling space")
        removed = mgr.prune(strategy="truncate", target_tokens=20)
        assert removed > 0
        assert mgr.total_tokens <= 20

    def test_compress(self):
        mgr = ContextManager(max_tokens=4096)
        mgr.add_message("user", "Hi there friend")
        mgr.add_message("user", "How are you today?")
        mgr.add_message("assistant", "I am fine thanks!")
        before = mgr.message_count
        mgr.compress(target_tokens=5000)
        # With a large target, no compression needed
        assert mgr.message_count <= before

    def test_stats(self):
        mgr = ContextManager(max_tokens=4096)
        mgr.add_message("user", "Hello")
        mgr.add_message("assistant", "Hi there!")
        stats = mgr.stats()
        assert stats["message_count"] == 2
        assert "total_tokens" in stats
        assert "utilization" in stats

    def test_invalid_max_tokens(self):
        with pytest.raises(ValueError):
            ContextManager(max_tokens=0)

    def test_clear(self):
        mgr = ContextManager()
        mgr.add_message("user", "test")
        mgr.clear()
        assert mgr.message_count == 0

    def test_sliding_window_integration(self):
        window = SlidingWindow(max_messages=3)
        mgr = ContextManager(max_tokens=4096, sliding_window=window)
        for i in range(10):
            mgr.add_message("user", f"msg {i}")
        ctx = mgr.get_context()
        assert len(ctx) <= 3

    def test_serialization_round_trip(self):
        mgr = ContextManager(max_tokens=2048)
        mgr.add_message("system", "You are a helpful bot.")
        mgr.add_message("user", "What is 2+2?")
        data = mgr.to_dict()
        mgr2 = ContextManager.from_dict(data)
        assert mgr2.max_tokens == 2048
        assert mgr2.message_count == 2
