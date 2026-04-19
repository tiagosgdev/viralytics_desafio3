"""Refinement semantic safeguards for LNIAGIA.query_parsing.llm_query_parser."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path


def _load_parser_module():
    repo_root = Path(__file__).resolve().parents[1]
    lniagia_dir = repo_root / "LNIAGIA"
    if str(lniagia_dir) not in sys.path:
        sys.path.insert(0, str(lniagia_dir))
    return importlib.import_module("LNIAGIA.query_parsing.llm_query_parser")


def _response(payload: dict):
    content = json.dumps(payload, ensure_ascii=False)
    return types.SimpleNamespace(message=types.SimpleNamespace(content=content))


def test_refine_query_replaces_type_for_instead_phrase(monkeypatch):
    parser = _load_parser_module()
    payload = {
        "query": "I want trousers instead of a t-shirt",
        "filters": {
            "include": {"type": ["trousers", "short_sleeve_top"]},
            "exclude": {},
        },
    }
    monkeypatch.setattr(parser.ollama, "chat", lambda *args, **kwargs: _response(payload))

    result = parser.refine_query(
        previous_query="I want a t-shirt",
        previous_filters={"include": {"type": ["short_sleeve_top"]}, "exclude": {}},
        refinement="Now I want trousers instead of a t-shirt",
        verbose=False,
    )

    assert result["filters"]["include"]["type"] == ["trousers"]


def test_refine_query_set_only_keeps_single_type(monkeypatch):
    parser = _load_parser_module()
    payload = {
        "query": "No, I only want trousers",
        "filters": {
            "include": {"type": ["short_sleeve_top", "trousers"], "color": ["green"]},
            "exclude": {},
        },
    }
    monkeypatch.setattr(parser.ollama, "chat", lambda *args, **kwargs: _response(payload))

    result = parser.refine_query(
        previous_query="I want green tops and trousers",
        previous_filters={
            "include": {"type": ["short_sleeve_top", "trousers"], "color": ["green"]},
            "exclude": {},
        },
        refinement="No, I only want trousers",
        verbose=False,
    )

    assert result["filters"]["include"]["type"] == ["trousers"]


def test_refine_query_additive_phrase_keeps_existing_and_new_types(monkeypatch):
    parser = _load_parser_module()
    payload = {
        "query": "also include trousers",
        "filters": {
            "include": {"type": ["short_sleeve_top", "trousers"]},
            "exclude": {},
        },
    }
    monkeypatch.setattr(parser.ollama, "chat", lambda *args, **kwargs: _response(payload))

    result = parser.refine_query(
        previous_query="I want a t-shirt",
        previous_filters={"include": {"type": ["short_sleeve_top"]}, "exclude": {}},
        refinement="Also include trousers",
        verbose=False,
    )

    assert result["filters"]["include"]["type"] == ["short_sleeve_top", "trousers"]


def test_refine_query_removes_include_exclude_overlap(monkeypatch):
    parser = _load_parser_module()
    payload = {
        "query": "not red",
        "filters": {
            "include": {"color": ["red"]},
            "exclude": {"color": ["red"]},
        },
    }
    monkeypatch.setattr(parser.ollama, "chat", lambda *args, **kwargs: _response(payload))

    result = parser.refine_query(
        previous_query="I want red clothing",
        previous_filters={"include": {"color": ["red"]}, "exclude": {}},
        refinement="Not red",
        verbose=False,
    )

    assert result["filters"].get("include", {}).get("color") in (None, [])
    assert result["filters"].get("exclude", {}).get("color") == ["red"]
