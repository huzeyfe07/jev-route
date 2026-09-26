"""Tests for the FastAPI example: its HTTP surface and the gate it enforces.

The example loads offline (canned Jev answers through an ``httpx.MockTransport``)
and is driven through FastAPI's ``TestClient``, so the routing, escalation and
audit behaviour are asserted without a server or an API key.

    python -m pytest tests/test_fastapi_service.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from collections.abc import Iterator
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE_PATH = PROJECT_ROOT / "examples" / "fastapi_service.py"


def load_example() -> ModuleType:
    """Import ``examples/fastapi_service.py`` without starting a server."""
    spec = importlib.util.spec_from_file_location("jev_route_fastapi_example", EXAMPLE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Cannot load the example from {EXAMPLE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example() -> ModuleType:
    """The loaded example module, shared by every test."""
    return load_example()


@pytest.fixture(autouse=True)
def offline_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the example into its offline mock-transport mode."""
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


@pytest.fixture
def api(example: ModuleType) -> Iterator[TestClient]:
    """A client for a fresh app instance, with the lifespan running."""
    with TestClient(example.create_app()) as client:
        yield client


def test_healthz_and_options_describe_the_router(api: TestClient) -> None:
    """Readiness reports the model and the gate; options list what Jev may pick."""
    health = api.get("/healthz").json()

    assert health["status"] == "ok"
    assert health["model"] == "typesafe/jev-1.13"
    assert health["confidence_threshold"] == pytest.approx(0.6)
    assert health["fallback_label"] == "human_handoff"
    assert health["offline_mock"] is True

    options = api.get("/options").json()
    assert [option["label"] for option in options] == [
        "refund_tool",
        "order_status_tool",
        "knowledge_base_agent",
    ]
    assert {option["kind"] for option in options} == {"tool", "agent"}


def test_confident_message_reaches_the_privileged_tool(
    api: TestClient,
    example: ModuleType,
) -> None:
    """A 0.91 decision passes the gate and the refund handler records the call."""
    response = api.post(
        "/route",
        json={"message": example.REFUND_PROMPT, "context": {"order_id": "A-1042"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "refund_tool"
    assert body["accepted"] is True
    assert body["escalated"] is False
    assert body["outcome"] == "routed"
    assert body["score"] == pytest.approx(0.91)
    assert body["value"]["issued"] is True

    ledger = api.get("/refunds").json()
    assert ledger["count"] == 1
    assert ledger["calls"][0]["order_id"] == "A-1042"


def test_low_confidence_message_never_reaches_the_privileged_tool(
    api: TestClient,
    example: ModuleType,
) -> None:
    """The same handler at 0.44 confidence is blocked and escalated instead."""
    body = api.post("/route", json={"message": example.AMBIGUOUS_PROMPT}).json()

    assert body["winner"] == "refund_tool"  # the engine did lean that way
    assert body["score"] == pytest.approx(0.44)
    assert body["accepted"] is False
    assert body["escalated"] is True
    assert body["outcome"] == "fallback"
    assert body["label"] == "human_handoff"
    assert body["value"]["candidate"] == "refund_tool"
    assert api.get("/refunds").json() == {"calls": [], "count": 0}


def test_engine_abstention_escalates(api: TestClient, example: ModuleType) -> None:
    """A noul answer (no winner at all) is handled like a rejected winner."""
    body = api.post("/route", json={"message": example.OUT_OF_SCOPE_PROMPT}).json()

    assert body["winner"] is None
    assert body["accepted"] is False
    assert body["outcome"] == "fallback"
    assert body["label"] == "human_handoff"
    assert api.get("/refunds").json()["count"] == 0


def test_audit_middleware_records_every_decision(
    api: TestClient,
    example: ModuleType,
) -> None:
    """The middleware trail shows the winner and the handler that got the call."""
    for message, context in (
        (example.REFUND_PROMPT, {"order_id": "A-1042"}),
        (example.AMBIGUOUS_PROMPT, {}),
        (example.OUT_OF_SCOPE_PROMPT, {}),
    ):
        assert api.post("/route", json={"message": message, "context": context}).status_code == 200

    audit = api.get("/audit").json()
    assert [entry["target"] for entry in audit] == [
        "refund_tool",
        "human_handoff",
        "human_handoff",
    ]
    assert [entry["accepted"] for entry in audit] == [True, False, False]
    assert all(entry["reason"] for entry in audit)
    assert all(entry["latency_ms"] is not None for entry in audit)


def test_empty_message_is_rejected_by_the_request_model(api: TestClient) -> None:
    """Pydantic validates the payload before any routing happens."""
    assert api.post("/route", json={"message": ""}).status_code == 422
