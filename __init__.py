"""Compatibilité après renommage du module archon en k."""

import sys
from importlib import import_module

import k

# Rediriger les imports de `archon` vers `k`
class _ArchonCompatibilityFinder:
    def __getattr__(self, name):
        try:
            return getattr(k, name)
        except AttributeError:
            try:
                module = import_module("k." + name)
                return module
            except ImportError:
                raise AttributeError("Module k has no attribute " + name)

sys.modules["archon"] = _ArchonCompatibilityFinder()
# Avertissement: ce module est un shim de compatibilité. Utilisez `import k` à la place.
