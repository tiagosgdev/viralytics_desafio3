"""Conversation routing tests for LNIAGIA.search_app.run_conversation_model."""

from __future__ import annotations

import importlib
import sys
import types
from copy import deepcopy

import pytest


def _install_optional_dependency_stubs() -> None:
    if "sentence_transformers" not in sys.modules:
        sentence_transformers = types.ModuleType("sentence_transformers")

        class _DummySentenceTransformer:
            def __init__(self, *args, **kwargs):
                pass

            def encode(self, *args, **kwargs):
                return [[0.0]]

            def get_sentence_embedding_dimension(self):
                return 1

        sentence_transformers.SentenceTransformer = _DummySentenceTransformer
        sys.modules["sentence_transformers"] = sentence_transformers

    if "qdrant_client" not in sys.modules:
        qdrant_client = types.ModuleType("qdrant_client")

        class _DummyQdrantClient:
            def __init__(self, *args, **kwargs):
                pass

        qdrant_client.QdrantClient = _DummyQdrantClient
        sys.modules["qdrant_client"] = qdrant_client

    if "qdrant_client.models" not in sys.modules:
        qdrant_models = types.ModuleType("qdrant_client.models")
        for name in [
            "Distance",
            "VectorParams",
            "PointStruct",
            "Filter",
            "FieldCondition",
            "MatchAny",
            "Range",
        ]:
            setattr(qdrant_models, name, type(name, (), {}))
        sys.modules["qdrant_client.models"] = qdrant_models

    if "ollama" not in sys.modules:
        ollama = types.ModuleType("ollama")
        ollama.chat = lambda *args, **kwargs: {"message": {"content": "{}"}}
        sys.modules["ollama"] = ollama


def _load_search_app_module():
    _install_optional_dependency_stubs()
    return importlib.import_module("LNIAGIA.search_app")


@pytest.fixture
def search_app(monkeypatch):
    app = _load_search_app_module()
    strict_calls: list[bool] = []

    def fake_parse_with_mode(mode: str, message: str):
        lowered = message.lower()

        if "trouser" in lowered:
            include = {"type": ["trousers"]}
            if "green" in lowered:
                include["color"] = ["green"]
            return {"include": include, "exclude": {}}, "llm", None

        if "t-shirt" in lowered or "tshirt" in lowered or " tee" in lowered:
            return {"include": {"type": ["short_sleeve_top"]}, "exclude": {}}, "llm", None

        if "green" in lowered:
            return {"include": {"color": ["green"]}, "exclude": {}}, "llm", None

        if "red" in lowered and "not" in lowered:
            return {"include": {}, "exclude": {"color": ["red"]}}, "llm", None

        return {"include": {}, "exclude": {}}, "llm", None

    def fake_update_state_with_message(
        mode: str,
        current_query: str,
        current_filters: dict,
        message: str,
        has_previous_context: bool,
        recent_user_messages: list[str] | None = None,
    ):
        if not has_previous_context:
            parsed, source, warning = fake_parse_with_mode(mode, message)
            return message.strip(), app._normalize_filter_payload(parsed), source, warning

        lowered = message.lower()
        normalized = app._normalize_filter_payload(deepcopy(current_filters))
        include = dict(normalized.get("include", {}))
        exclude = dict(normalized.get("exclude", {}))

        if "t-shirt" in lowered or "tshirt" in lowered:
            include["type"] = ["short_sleeve_top"]
        if "trouser" in lowered:
            include["type"] = ["trousers"]
        if "green" in lowered:
            include["color"] = ["green"]
        if "red" in lowered and ("not" in lowered or "dont" in lowered or "don't" in lowered):
            exclude["color"] = ["red"]

        return (
            f"{current_query} {message}".strip(),
            app._normalize_filter_payload({"include": include, "exclude": exclude}),
            "llm_refine",
            None,
        )

    monkeypatch.setattr(app, "_parse_with_mode", fake_parse_with_mode)
    monkeypatch.setattr(app, "_update_state_with_message", fake_update_state_with_message)
    monkeypatch.setattr(app, "_ensure_vector_db_ready", lambda: (True, None))
    monkeypatch.setattr(app, "_load_model", lambda: object())
    def fake_filtered_search(query, filters, model, strict=False):
        strict_calls.append(bool(strict))
        return [object()]

    monkeypatch.setattr(app, "filtered_search", fake_filtered_search)
    monkeypatch.setattr(
        app,
        "_build_ranked_results",
        lambda hits: [{"rank": 1, "score": 0.9, "item_id": 1, "item": {"id": 1, "type": "short_sleeve_top"}}] if hits else [],
    )
    monkeypatch.setattr(app, "_generate_confirmation_lead", lambda mode, summary, query, user_message, filters: "Lead.")
    monkeypatch.setattr(app, "_generate_persona_reply", lambda mode, scenario, user_message, context: scenario)
    monkeypatch.setattr(app, "_fixed_persona_intro", lambda mode: "intro")

    app._CONVERSATION_MODEL = None
    app._strict_calls = strict_calls
    return app


def _run(search_app, message: str, state: dict | None = None, assistant_mode: str | None = None) -> dict:
    return search_app.run_conversation_model(
        detected_type=None,
        user_input=message,
        conversation_state=state,
        assistant_mode=assistant_mode,
    )


def test_second_consecutive_vague_request_auto_searches_with_cruella_strict(search_app):
    first = _run(search_app, "I want a t-shirt")
    assert first["action"] == "ask_details"
    assert first["state"]["pending_detail_signature"]

    second = _run(search_app, "I really just want a t-shirt", first["state"])
    assert second["action"] == "searched"
    assert second["state"]["strict"] is True
    assert second["state"]["awaiting_confirmation"] is False
    assert search_app._strict_calls[-1] is True


def test_post_search_refinement_auto_searches_and_reuses_cruella_strict(search_app):
    first = _run(search_app, "I want a t-shirt")
    searched = _run(search_app, "I really just want a t-shirt", first["state"])
    refined = _run(search_app, "I dont want anything that is red", searched["state"])

    assert refined["action"] == "searched"
    assert refined["state"]["strict"] is True
    assert refined["state"]["filters"]["exclude"]["color"] == ["red"]
    assert search_app._strict_calls[-1] is True


def test_type_change_after_results_starts_new_query_flow(search_app):
    first = _run(search_app, "I want a t-shirt")
    searched = _run(search_app, "I really just want a t-shirt", first["state"])
    changed = _run(search_app, "Now I want a green pair of trousers", searched["state"])

    assert changed["action"] == "confirm_requirements"
    assert changed["state"]["awaiting_confirmation"] is True
    assert changed["state"]["strict"] is True
    assert changed["state"]["has_completed_search"] is False


def test_non_type_followup_refines_and_searches_immediately(search_app):
    first = _run(search_app, "I want a t-shirt")
    searched = _run(search_app, "I really just want a t-shirt", first["state"])
    refined = _run(search_app, "Now I want it to be green", searched["state"])

    assert refined["action"] == "searched"
    assert refined["state"]["filters"]["include"]["color"] == ["green"]
    assert refined["state"]["strict"] is True
    assert search_app._strict_calls[-1] is True


def test_legacy_state_without_strictness_fields_defaults_to_cruella_strict(search_app):
    prior_state = {
        "started": True,
        "assistant_mode": "cruella",
        "query": "I need formal work clothing",
        "filters": {"include": {"style": ["formal"], "occasion": ["work"]}, "exclude": {}},
        "awaiting_confirmation": False,
        "parser_source": "llm_refine",
        "has_completed_search": True,
    }

    refined = _run(search_app, "anything, just not red", prior_state)
    assert refined["action"] == "searched"
    assert refined["state"]["strict"] is True
    assert search_app._strict_calls[-1] is True


def test_edna_mode_always_searches_with_flexible_matching(search_app):
    first = _run(search_app, "I want a t-shirt", assistant_mode="edna")
    assert first["action"] == "ask_details"

    second = _run(search_app, "I really just want a t-shirt", first["state"], assistant_mode="edna")
    assert second["action"] == "searched"
    assert second["state"]["strict"] is False
    assert search_app._strict_calls[-1] is False


def test_short_detailed_revision_during_confirmation_is_parsed(search_app):
    first = _run(search_app, "I want a t-shirt")
    searched = _run(search_app, "I really just want a t-shirt", first["state"])
    changed = _run(search_app, "Now I want a green pair of trousers", searched["state"])

    assert changed["action"] == "confirm_requirements"
    assert changed["state"]["awaiting_confirmation"] is True

    revised = _run(search_app, "No, only trousers please", changed["state"])

    assert revised["action"] == "confirm_requirements"
    assert revised["state"]["awaiting_confirmation"] is True
    assert revised["state"]["query"].endswith("No, only trousers please")
