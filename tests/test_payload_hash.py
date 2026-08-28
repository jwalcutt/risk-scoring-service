"""Tests for the input hash over raw scoring event payloads.

The rules these tests pin:

- The hash is SHA-256 over a canonical JSON serialization of the parsed
  event object: keys sorted at every nesting level, minimal separators,
  UTF-8 encoding with non-ASCII characters kept literal.
- The digest depends only on the parsed object, never on the formatting
  or key order of the JSON text it came from.
- Values are hashed unmodified: string versus number matters, and no
  unicode normalization is applied.
- Non-standard JSON values (NaN, infinities) are rejected loudly.
"""

from __future__ import annotations

import json
import string
import unicodedata

import pytest

from risk_scoring.payload_hash import canonical_event_bytes, payload_hash

_EVENT: dict[str, object] = {
    "event_type": "encounter",
    "payload": {
        "Id": "enc-001",
        "START": "2024-03-01T08:00:00Z",
        "STOP": "2024-03-04T12:30:00Z",
        "PATIENT": "pat-001",
        "ENCOUNTERCLASS": "inpatient",
    },
}

# Computed once at implementation time; freezes the algorithm forever.
_EVENT_DIGEST = "1859774b2636d4593f19dc9aa4d39e8508e0fe8535c0d9ed9ebf47b4aee4168d"


def test_known_vector_pins_algorithm() -> None:
    assert payload_hash(_EVENT) == _EVENT_DIGEST


def test_canonical_bytes_are_sorted_compact_utf8() -> None:
    assert canonical_event_bytes({"b": "2", "a": {"d": "4", "c": "3"}}) == (
        b'{"a":{"c":"3","d":"4"},"b":"2"}'
    )


def test_hash_is_independent_of_key_insertion_order() -> None:
    reordered: dict[str, object] = {
        "payload": {
            "ENCOUNTERCLASS": "inpatient",
            "PATIENT": "pat-001",
            "STOP": "2024-03-04T12:30:00Z",
            "START": "2024-03-01T08:00:00Z",
            "Id": "enc-001",
        },
        "event_type": "encounter",
    }
    assert payload_hash(reordered) == _EVENT_DIGEST


def test_hash_changes_on_any_value_change() -> None:
    altered = json.loads(json.dumps(_EVENT))
    altered["payload"]["Id"] = "enc-002"
    assert payload_hash(altered) != _EVENT_DIGEST


def test_equivalent_json_texts_hash_equal() -> None:
    compact = json.dumps(_EVENT, separators=(",", ":"))
    pretty = json.dumps(_EVENT, indent=4)
    assert payload_hash(json.loads(compact)) == payload_hash(json.loads(pretty)) == _EVENT_DIGEST


def test_string_and_number_values_hash_differently() -> None:
    assert payload_hash({"a": "1"}) != payload_hash({"a": 1})


def test_unicode_values_hash_via_utf8_without_normalization() -> None:
    nfc = {"DESCRIPTION": unicodedata.normalize("NFC", "café")}
    nfd = {"DESCRIPTION": unicodedata.normalize("NFD", "café")}
    assert payload_hash(nfc) == "6465aa5a38543c08dabc568d3736f50764a04e87e7673797a52ff956b789e0e8"
    assert payload_hash(nfd) == "7abcdc85d4310e92f529c52c9c3dec7cacc35ae0cb59adec02d1f8308a17671a"
    assert payload_hash(nfc) != payload_hash(nfd)


def test_nan_rejected_loudly() -> None:
    with pytest.raises(ValueError):
        payload_hash({"a": float("nan")})


def test_digest_shape() -> None:
    digest = payload_hash(_EVENT)
    assert len(digest) == 64
    assert set(digest) <= set(string.hexdigits.lower())
