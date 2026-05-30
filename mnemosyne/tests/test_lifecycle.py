"""Lifecycle tests for mnemosyne supersede shape (Stage 1 DoD)."""

from __future__ import annotations

import json
import pytest

from mnemosyne import store


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    """Redirect DATA_DIR to a temp path so tests never touch real data."""
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)


def test_supersede_filters_from_load():
    """Append A, B, C; supersede A by C; store.load() returns [B, C], not A."""
    id_a = store.append("test", "question A", "answer A", "session-1")
    id_b = store.append("test", "question B", "answer B", "session-1")
    id_c = store.append("test", "question C", "answer C", "session-2")

    store.supersede("test", id_a, id_c, reason="A contradicted by C")

    results = store.load("test")
    ids = [r["id"] for r in results]
    assert id_a not in ids, "superseded pair A must not appear in load()"
    assert id_b in ids, "pair B must still appear"
    assert id_c in ids, "pair C must still appear"
    assert len(results) == 2


def test_supersede_append_only():
    """Supersede must not rewrite the file — original record + tombstone both present in raw file."""
    domain = "test"
    id_a = store.append(domain, "question A", "answer A")
    id_b = store.append(domain, "question B", "answer B")

    store.supersede(domain, id_a, id_b)

    raw_text = store.domain_path(domain).read_text(encoding="utf-8")
    raw_lines = [l for l in raw_text.splitlines() if l.strip()]
    assert len(raw_lines) == 3, "file must have 3 lines: A record, B record, tombstone"

    raw_records = [json.loads(l) for l in raw_lines]
    original_a = next((r for r in raw_records if r.get("id") == id_a), None)
    assert original_a is not None, "original A record must still be in the file"

    tombstone = next((r for r in raw_records if r.get("type") == "supersede"), None)
    assert tombstone is not None, "tombstone must be present"
    assert tombstone["target_id"] == id_a
    assert tombstone["by_id"] == id_b


def test_supersede_idempotent_raises():
    """Superseding the same target twice must raise ValueError."""
    id_a = store.append("test", "question A", "answer A")
    id_b = store.append("test", "question B", "answer B")

    store.supersede("test", id_a, id_b)

    with pytest.raises(ValueError, match="already superseded"):
        store.supersede("test", id_a, id_b)
