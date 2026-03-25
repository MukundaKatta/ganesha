"""
Command-line interface for Ganesha.

Subcommands:
    analyze   Show token statistics for a conversation JSON file.
    compress  Compress a conversation file and write the result.
    prune     Prune a conversation file using a named strategy.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from ganesha.core import ContextManager, Message


def _load_messages(path: str) -> List[Message]:
    """Load messages from a JSON file (list of {role, content} dicts)."""
    with open(path, "r") as f:
        data = json.load(f)
    messages: List[Message] = []
    for item in data:
        messages.append(Message(
            role=item["role"],
            content=item["content"],
            metadata=item.get("metadata", {}),
        ))
    return messages


def _save_messages(messages: List[Message], path: str) -> None:
    """Write messages to a JSON file."""
    data = [m.to_dict() for m in messages]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Show token statistics for a conversation file."""
    messages = _load_messages(args.file)
    mgr = ContextManager(max_tokens=args.max_tokens)
    for m in messages:
        mgr.add(m)
    stats = mgr.stats()
    print(json.dumps(stats, indent=2))


def cmd_compress(args: argparse.Namespace) -> None:
    """Compress a conversation file."""
    messages = _load_messages(args.file)
    mgr = ContextManager(max_tokens=args.max_tokens)
    for m in messages:
        mgr.add(m)
    removed = mgr.compress(target_tokens=args.target)
    output = args.output or args.file
    _save_messages(mgr.messages, output)
    print(f"Removed {removed} messages. Output written to {output}")


def cmd_prune(args: argparse.Namespace) -> None:
    """Prune a conversation file using the given strategy."""
    messages = _load_messages(args.file)
    mgr = ContextManager(max_tokens=args.max_tokens)
    for m in messages:
        mgr.add(m)
    removed = mgr.prune(strategy=args.strategy, target_tokens=args.target)
    output = args.output or args.file
    _save_messages(mgr.messages, output)
    print(f"Pruned {removed} messages with strategy '{args.strategy}'. Output written to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ganesha",
        description="Ganesha - Context Engineering CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Show token statistics")
    p_analyze.add_argument("file", help="Path to conversation JSON file")
    p_analyze.add_argument("--max-tokens", type=int, default=4096)

    # compress
    p_compress = sub.add_parser("compress", help="Compress a conversation file")
    p_compress.add_argument("file", help="Path to conversation JSON file")
    p_compress.add_argument("--max-tokens", type=int, default=4096)
    p_compress.add_argument("--target", type=int, default=2048)
    p_compress.add_argument("--output", "-o", help="Output file path")

    # prune
    p_prune = sub.add_parser("prune", help="Prune a conversation file")
    p_prune.add_argument("file", help="Path to conversation JSON file")
    p_prune.add_argument("--strategy", default="truncate",
                         choices=["truncate", "summarize", "priority"])
    p_prune.add_argument("--max-tokens", type=int, default=4096)
    p_prune.add_argument("--target", type=int, default=2048)
    p_prune.add_argument("--output", "-o", help="Output file path")

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "analyze": cmd_analyze,
        "compress": cmd_compress,
        "prune": cmd_prune,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
