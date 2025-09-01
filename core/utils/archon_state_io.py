# -*- coding: utf-8 -*-
"""
Archon State IO adapter

Provides utilities to read and write a shared Markdown file used as a
communication bus with the RHCV Vector assistant. The file contains
fenced JSON blocks (```json ... ```). We parse the last JSON fence as the
active state and can update its "outputs" and append timestamped messages.

Rules (per project guidelines):
- Python 3.10+, type hints, PEP 8.
- No secrets persisted. File path is provided by the caller.
- Robust to multiple sections; we always use the LAST fenced JSON block.

Typical layout inside the Markdown file:

---
## Archon Update — <timestamp>
```json
{
  "inputs": { ... },
  "outputs": { ... },
  "messages": [ { "timestamp": "...", "author": "archon", "note": "..." } ]
}
```
---

Functions:
- read_shared_state(md_text) -> dict | None
- replace_last_json_fence(md_text, new_json_text) -> str
- load_state_from_file(path) -> dict | None
- update_outputs_in_file(path, outputs_dict) -> None
- append_message_in_file(path, author, note, iso_ts) -> None

"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

JSON_FENCE_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


@dataclass
class SharedState:
    """Container for parsed shared state."""
    raw: Dict[str, Any]

    @property
    def inputs(self) -> Dict[str, Any]:
        return dict(self.raw.get("inputs") or {})

    @property
    def outputs(self) -> Dict[str, Any]:
        return dict(self.raw.get("outputs") or {})

    @property
    def messages(self) -> list:
        msgs = self.raw.get("messages")
        return list(msgs) if isinstance(msgs, list) else []


def read_shared_state(md_text: str) -> Optional[SharedState]:
    """Parse the last fenced JSON block in the Markdown text.

    Returns SharedState if a valid JSON block is found, else None.
    """
    if not md_text:
        return None
    matches = list(JSON_FENCE_RE.finditer(md_text))
    if not matches:
        return None
    last = matches[-1]
    block = last.group(1)
    try:
        obj = json.loads(block)
        if isinstance(obj, dict):
            return SharedState(raw=obj)
    except Exception:
        return None
    return None


def _replace_span(text: str, span: Tuple[int, int], replacement: str) -> str:
    return text[: span[0]] + replacement + text[span[1] :]


def replace_last_json_fence(md_text: str, new_json_text: str) -> str:
    """Replace the last fenced JSON block with new_json_text (without fences)."""
    matches = list(JSON_FENCE_RE.finditer(md_text))
    if not matches:
        # Append a new fenced block at the end
        suffix = f"\n\n```json\n{new_json_text}\n```\n"
        return (md_text or "") + suffix
    last = matches[-1]
    # Build fenced replacement
    fenced = f"```json\n{new_json_text}\n```"
    return _replace_span(md_text, last.span(), fenced)


def load_state_from_file(path: str) -> Optional[SharedState]:
    """Read file content and parse shared state."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            txt = f.read()
        return read_shared_state(txt)
    except FileNotFoundError:
        return None


def update_outputs_in_file(path: str, outputs: Dict[str, Any]) -> None:
    """Update the "outputs" object inside the last fenced JSON block and write back.

    If no fenced block exists, a minimal block with these outputs will be appended.
    """
    # Read
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            txt = f.read()
    except FileNotFoundError:
        txt = ""
    state = read_shared_state(txt)
    if state is None:
        obj: Dict[str, Any] = {"inputs": {}, "outputs": outputs, "messages": []}
    else:
        obj = state.raw
        obj["outputs"] = outputs
    new_json = json.dumps(obj, ensure_ascii=False, indent=2)
    new_txt = replace_last_json_fence(txt, new_json)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_txt)


def append_message_in_file(path: str, author: str, note: str, iso_ts: str) -> None:
    """Append a timestamped message into the last fenced JSON block and write back."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            txt = f.read()
    except FileNotFoundError:
        txt = ""
    state = read_shared_state(txt)
    if state is None:
        obj: Dict[str, Any] = {"inputs": {}, "outputs": {}, "messages": []}
    else:
        obj = state.raw
    msgs = obj.get("messages")
    if not isinstance(msgs, list):
        msgs = []
    msgs.append({"timestamp": iso_ts, "author": author, "note": note})
    obj["messages"] = msgs
    new_json = json.dumps(obj, ensure_ascii=False, indent=2)
    new_txt = replace_last_json_fence(txt, new_json)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_txt)
