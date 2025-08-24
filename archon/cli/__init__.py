"""Archon CLI package (skeleton).

This package will host user-facing commands in a future Phase.
For now, it exposes a minimal main() for manual invocation without
registering console_scripts to avoid breaking existing workflows.
"""
from __future__ import annotations

from .main import main  # re-export for convenience

__all__ = ["main"]
