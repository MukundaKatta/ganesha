# Architecture

## Overview

Ganesha is a context engineering library designed for multi-agent systems. It provides token-aware conversation management, compression strategies, and a lightweight memory store.

## Module Structure

```
src/ganesha/
  __init__.py        # Public API exports
  core.py            # ContextManager, Message, SlidingWindow, TokenCounter
  compression.py     # Compressor protocol + TruncateStrategy, SummarizeStrategy, PriorityStrategy
  memory.py          # MemoryStore with TTL-based expiry
  config.py          # GaneshaConfig dataclass
  cli.py             # Command-line interface (analyze, compress, prune)
  __main__.py        # python -m ganesha entry point
```

## Core Concepts

### Token Counting

`TokenCounter` uses a word-based heuristic (default 1.3 tokens per word) to estimate token counts without requiring a tokenizer dependency. This keeps the library lightweight while providing reasonable estimates for English text.

### Context Management

`ContextManager` is the primary entry point. It maintains an ordered message list and provides:

- **Token budgets**: `get_context(max_tokens)` returns messages that fit within a budget.
- **Pruning**: `prune(strategy)` removes messages in-place using a named strategy.
- **Compression**: `compress()` merges consecutive same-role messages.
- **Serialization**: `to_dict()` / `from_dict()` for persistence.

### Sliding Window

`SlidingWindow` enforces limits by message count or token count, always preserving system messages. It is applied automatically in `get_context()` when configured.

### Compression Strategies

All strategies implement the `Compressor` protocol:

| Strategy | Approach |
|---|---|
| `TruncateStrategy` | Drop oldest non-system messages |
| `SummarizeStrategy` | Merge consecutive same-role messages, then truncate |
| `PriorityStrategy` | Keep system + recent user messages, compress assistant responses |

### Memory Store

`MemoryStore` provides key-value storage with optional TTL expiry. Useful for persisting agent state, tool results, or intermediate computations across conversation turns.

## Data Flow

```
User/Agent  -->  ContextManager.add_message()
                      |
                      v
              Internal message list
                      |
         get_context() / prune() / compress()
                      |
                      v
              Trimmed message list  -->  LLM API
```
