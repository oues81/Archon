"""Minimal CLI entrypoint (placeholder).

Usage (dev):
    python -m archon.archon.cli --help

We intentionally avoid adding external CLI deps; use argparse only.
"""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="archon", description="Archon CLI (skeleton)")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    args = parser.parse_args(argv)
    if getattr(args, "version", False):
        # Lazy import to avoid import side-effects at module load time
        try:
            from k import __version__  # type: ignore
        except Exception:
            __version__ = "0.0.0-dev"
        print(__version__)
        return 0
    # For now, just display help if no actionable flags were provided
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
