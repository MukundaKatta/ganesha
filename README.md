# 🔱 Ganesha — Context Engineering Library

> **Hindu Mythology: Remover of Obstacles** | Token compression and context management for multi-agent systems

[![CI](https://github.com/MukundaKatta/ganesha/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/ganesha/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/🌐_Live_Demo-Visit_Site-blue?style=for-the-badge)](https://MukundaKatta.github.io/ganesha/)

---

## Features

- **Token-aware context management** — Keep conversations within LLM token budgets automatically.
- **Sliding window** — Retain the most recent N messages or N tokens, with system message preservation.
- **Compression strategies** — Truncate, summarize (merge), or priority-based compression.
- **Memory store** — Key-value storage with TTL-based expiry for agent state.
- **Zero dependencies** — Pure Python, no external packages required.
- **CLI included** — Analyze, compress, and prune conversation files from the command line.

## Quickstart

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
from ganesha import ContextManager, SlidingWindow

# Create a context manager with a 4096-token budget
ctx = ContextManager(max_tokens=4096)

# Add messages
ctx.add_message("system", "You are a helpful assistant.")
ctx.add_message("user", "What is context engineering?")
ctx.add_message("assistant", "Context engineering is the practice of managing ...")

# Get messages that fit within a token budget
messages = ctx.get_context(max_tokens=2048)

# Prune old messages using a strategy
ctx.prune(strategy="truncate", target_tokens=1024)
```

### Sliding Window

```python
from ganesha import ContextManager, SlidingWindow

window = SlidingWindow(max_messages=10, preserve_system=True)
ctx = ContextManager(max_tokens=4096, sliding_window=window)

# System messages are always preserved, even when the window slides
ctx.add_message("system", "You are a helpful assistant.")
for i in range(100):
    ctx.add_message("user", f"Question {i}")

# Only the system message + last 9 messages are returned
messages = ctx.get_context()
```

### Memory Store

```python
from ganesha import MemoryStore

memory = MemoryStore(default_ttl=3600)  # 1-hour default expiry

memory.store("user_preference", "dark mode")
memory.store("session_id", "abc123", ttl=300)  # 5-minute TTL

value = memory.recall("user_preference")  # "dark mode"
memory.prune_expired()  # Clean up stale entries
```

### Compression Strategies

```python
from ganesha import ContextManager

ctx = ContextManager(max_tokens=4096)
# ... add many messages ...

# Truncate: drop oldest non-system messages
ctx.prune(strategy="truncate", target_tokens=2048)

# Summarize: merge consecutive same-role messages
ctx.prune(strategy="summarize", target_tokens=2048)

# Priority: keep system + recent user messages, compress assistant
ctx.prune(strategy="priority", target_tokens=2048)
```

### CLI

```bash
# Analyze token usage in a conversation file
ganesha analyze conversation.json --max-tokens 4096

# Compress a conversation
ganesha compress conversation.json --target 2048 -o compressed.json

# Prune with a specific strategy
ganesha prune conversation.json --strategy priority --target 2048
```

## Architecture

```
src/ganesha/
  core.py          — ContextManager, Message, SlidingWindow, TokenCounter
  compression.py   — Compressor protocol + strategies (Truncate, Summarize, Priority)
  memory.py        — MemoryStore with TTL-based expiry
  config.py        — GaneshaConfig dataclass
  cli.py           — Command-line interface
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design documentation.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
make test

# Lint
make lint

# Type check
make typecheck
```

## 🌐 Live Demo

Visit the landing page: **https://MukundaKatta.github.io/ganesha/**

## License

MIT License — (C) 2026 Officethree Technologies. See [LICENSE](LICENSE) for details.

## 🔱 Part of the Mythological Portfolio

This is project **#ganesha** in the [100-project Mythological Portfolio](https://github.com/MukundaKatta) by Officethree Technologies.
