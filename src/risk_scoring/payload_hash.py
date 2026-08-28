"""Input hash tying a logged prediction to the exact scoring event posted.

The hash is SHA-256 over a canonical JSON serialization of the parsed
event object, computed before any pydantic parsing or feature work.
Canonical form: keys sorted lexicographically at every nesting level,
minimal separators, ``ensure_ascii=False``, UTF-8 encoding.

Judgment calls this module fixes:

- The hash covers the parsed JSON object, not the raw request bytes, so
  formatting and key order of the wire text never change the digest;
  duplicate keys in the raw text collapse to the last occurrence.
- Values are hashed unmodified: no unicode normalization (NFC and NFD
  forms of one glyph hash differently) and no type coercion (the string
  "1" and the number 1 hash differently).
- ``allow_nan=False`` rejects non-standard JSON (NaN, infinities) with a
  ``ValueError`` instead of silently emitting unparseable output.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def canonical_event_bytes(event: Mapping[str, object]) -> bytes:
    """Serialize an event object to its canonical UTF-8 JSON bytes."""
    text = json.dumps(
        event, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return text.encode("utf-8")


def payload_hash(event: Mapping[str, object]) -> str:
    """SHA-256 lowercase hex digest of the canonical event bytes."""
    return hashlib.sha256(canonical_event_bytes(event)).hexdigest()
