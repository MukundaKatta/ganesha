"""
Ganesha - Context Engineering Library for Multi-Agent Systems.

Token compression, conversation context management, memory pruning,
and sliding window support. Named after the Hindu deity Ganesha,
the Remover of Obstacles.
"""

from ganesha.core import (
    ContextManager,
    Message,
    SlidingWindow,
    TokenCounter,
)
from ganesha.compression import (
    Compressor,
    PriorityStrategy,
    SummarizeStrategy,
    TruncateStrategy,
)
from ganesha.memory import MemoryStore
from ganesha.config import GaneshaConfig

__version__ = "0.1.0"

__all__ = [
    "ContextManager",
    "Message",
    "SlidingWindow",
    "TokenCounter",
    "Compressor",
    "PriorityStrategy",
    "SummarizeStrategy",
    "TruncateStrategy",
    "MemoryStore",
    "GaneshaConfig",
    "__version__",
]
