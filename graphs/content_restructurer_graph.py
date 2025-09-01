"""Legacy proxy — delegates to canonical graphs.content_restructurer implementation."""
from __future__ import annotations

def get_content_flow():
    from k.graphs.content_restructurer.app.graph import get_content_flow as _impl
    return _impl()
