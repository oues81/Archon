# proxy to existing implementation during transition
from __future__ import annotations
"""PreProcessingAgent (skeleton): loads bytes, light text fallback, file meta + hash.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib


def _sha256_bytes(b: bytes) -> str:
    try:
        return hashlib.sha256(b).hexdigest()
    except Exception:
        return ""


def run(file_path: Optional[str]) -> Dict[str, Any]:
    p = Path(file_path) if file_path else None
    data = b""
    try:
        if p and p.exists():
            data = p.read_bytes()
    except Exception:
        data = b""
    text = ""
    # Minimal text extraction: only for .txt
    try:
        if p and p.suffix.lower() == ".txt":
            text = data.decode("utf-8", errors="ignore")
    except Exception:
        text = ""
    out: Dict[str, Any] = {
        "normalized_text": text,
        "sections": {},
        "file_meta": {"NomFichier": p.name if p else "", "UrlFichier": str(p.resolve()) if p else "", "mime_type": "", "pages": 0},
        "ResumeHash": _sha256_bytes(data),
        "DocumentId": "",
    }
    return out
