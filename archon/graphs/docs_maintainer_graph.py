"""Legacy proxy — delegates to canonical graphs.docs_maintainer implementation."""
from __future__ import annotations

def get_docs_flow():
    from archon.archon.graphs.docs_maintainer.app.graph import get_docs_flow as _impl
    return _impl()
