"""Logging utilities for PII redaction.

This module provides helpers to sanitize structures before logging.
It aims to avoid leaking sensitive CV content while preserving useful metadata.

Python >=3.10, PEP8 compliant, with type hints and docstrings.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping


# Common PII-like field names we should remove or mask entirely
_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        # Keep the list focused to avoid dropping benign fields like 'text'
        "file_base64",
        "parsed",
        "anonymized",
        "raw_text",
        "resume_bytes",
        "body",  # usually raw payload
    }
)


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# A more permissive phone pattern to catch common international and FR formats
# Examples: "+33 1 23 45 67 89", "06 12 34 56 78", "+1 555-123-4567"
_PHONE_RE = re.compile(
    r"(?:(?:\+\d{1,3}[\s\-.]*)?)"  # optional country code
    r"(?:\(?\d{1,4}\)?[\s\-.]*)?"  # optional area code
    r"\d(?:[\s\-.]*\d){6,}"        # at least 7 more digits with optional separators
)


def _mask_string(s: str) -> str:
    """Mask common PII patterns in a string.

    This performs light masking for emails and phone numbers. It is not
    perfect but reduces accidental leakage in logs.
    """
    s = _EMAIL_RE.sub("[REDACTED_EMAIL]", s)
    s = _PHONE_RE.sub("[REDACTED_PHONE]", s)
    return s


def _is_mapping(x: Any) -> bool:
    return isinstance(x, Mapping)


def _is_iterable(x: Any) -> bool:
    return isinstance(x, Iterable) and not isinstance(x, (str, bytes, bytearray))


def redact_pii(obj: Any) -> Any:
    """Return a copy of obj with PII removed or masked.

    - Removes sensitive keys entirely when found in mappings.
    - Recurses into mappings and iterables.
    - Masks emails/phones in strings.

    Parameters
    ----------
    obj: Any
        Arbitrary Python structure to sanitize for logging.

    Returns
    -------
    Any
        A sanitized deep copy suitable for logging.
    """
    if obj is None:
        return None

    if isinstance(obj, str):
        return _mask_string(obj)

    if isinstance(obj, (bytes, bytearray)):
        return "[BINARY]"

    if _is_mapping(obj):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, str) and k in _SENSITIVE_KEYS:
                # drop the field entirely
                continue
            out[k] = redact_pii(v)
        return out

    if _is_iterable(obj):
        return [redact_pii(v) for v in obj]

    return obj
