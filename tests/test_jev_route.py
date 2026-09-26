"""Smoke tests for JevRoute, driven by the runnable example.

The suite stays deliberately small: it loads ``examples/basic_routing.py``
(which runs offline through an ``httpx.MockTransport``) and asserts the routing
behaviour that the example demonstrates, so the example keeps working while the
library stays green.

    python -m pytest
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

from jev_route import (
    DecisionOutcome,
    IntentRequest,
    IntentRouter,
    JevClient,
    JevSettings,
    MiddlewarePipeline,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLE_PATH = PROJECT_ROOT / "examples" / "basic_routing.py"


def load_example() -> ModuleType:
    """Import ``examples/basic_routing.py`` without running ``main()``."""
    spec = importlib.util.spec_from_file_location("jev_route_example", EXAMPLE_PATH)
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


async def route_once(
    example: ModuleType,
    prompt: str,
    context: dict[str, Any] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Route one prompt through a fresh router and return the result + audit log."""
    audit: list[dict[str, Any]] = []
    client = example.build_client()
    router = example.build_router(client, audit)
    async with client:
        result = await router.route(prompt, context=context)
    return result, audit


async def test_weather_prompt_routes_to_the_weather_agent(example: ModuleType) -> None:
    """The happy path (the example's headline scenario) reaches the right agent."""
    result, audit = await route_once(example, example.WEATHER_PROMPT, {"city": "Ankara"})

    decision = result.decision
    assert result.handled is True
    assert result.label == "weather_agent"
    assert result.value["agent"] == "weather_agent"
    assert result.value["city"] == "Ankara"
    assert decision.winner == "weather_agent"
    assert decision.accepted is True
    assert decision.outcome.value == "routed"
    assert decision.score == pytest.approx(0.94)
    assert decision.target == "weather_agent"
    assert decision.response is not None
    assert decision.response.request_id.startswith("mock-")
    assert len(audit) == 1
    assert audit[0]["target"] == "weather_agent"


async def test_math_prompt_routes_to_the_calculator_tool(example: ModuleType) -> None:
    """A deterministic tool wins when the input is plain arithmetic."""
    result, _ = await route_once(example, example.MATH_PROMPT)

    assert result.handled is True
    assert result.label == "calculator_tool"
    assert result.value["result"] == 96
    assert result.decision.accepted is True


async def test_low_confidence_prompt_triggers_the_fallback(example: ModuleType) -> None:
    """A weak winner below the threshold must trip the confidence gate."""
    result, _ = await route_once(example, example.AMBIGUOUS_PROMPT)

    decision = result.decision
    assert decision.winner == "search_agent"
    assert decision.score == pytest.approx(0.42)
    assert decision.accepted is False
    assert decision.outcome.value == "fallback"
    assert "below the threshold" in (decision.reason or "")
    assert result.label == "human_handoff"
    assert result.value["escalated"] is True


async def test_out_of_scope_prompt_abstains_into_the_fallback(example: ModuleType) -> None:
    """An engine abstention (noul) also lands on the fallback handler."""
    result, _ = await route_once(example, example.OUT_OF_SCOPE_PROMPT)

    decision = result.decision
    assert decision.winner is None
    assert decision.accepted is False
    assert decision.score == pytest.approx(0.0)
    assert result.label == "human_handoff"
    assert result.handled is True


async def test_audit_middleware_records_every_decision(example: ModuleType) -> None:
    """The middleware pipeline sees and records each routing decision."""
    audit: list[dict[str, Any]] = []
    client = example.build_client()
    router = example.build_router(client, audit)
    prompts = [
        example.WEATHER_PROMPT,
        example.MATH_PROMPT,
        example.AMBIGUOUS_PROMPT,
        example.OUT_OF_SCOPE_PROMPT,
    ]

    async with client:
        for prompt in prompts:
            await router.route(prompt)

    assert [entry["target"] for entry in audit] == [
        "weather_agent",
        "calculator_tool",
        "human_handoff",
        "human_handoff",
    ]
    assert all(entry["input"] for entry in audit)
    assert "latency_ms" in audit[0]


def test_router_registers_the_demo_options(example: ModuleType) -> None:
    """Registration exposes labels, kinds and the confidence configuration."""
    client = example.build_client()
    router = example.build_router(client, [])

    assert router.labels == ("weather_agent", "calculator_tool", "search_agent")
    kinds = {option.label: option.kind for option in router.options}
    assert kinds["weather_agent"] == "agent"
    assert kinds["calculator_tool"] == "tool"
    assert router.confidence_threshold == pytest.approx(0.6)
    assert router.fallback_label == "human_handoff"


async def test_agent_loop_demo_runs(
    example: ModuleType,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The same pipeline machinery wraps a plain agent loop."""
    await example.run_agent_loop_demo(["plan", "act"])

    captured = capsys.readouterr().out
    assert "plan" in captured
    assert "handled" in captured


async def test_confidence_gate_never_reaches_a_weak_handler(example: ModuleType) -> None:
    """The same weak winner executes without a gate, and is blocked with one.

    The gate is the only difference between the two runs: identical client,
    identical engine answer (`search_agent` at 0.42), identical handler.
    """
    prompt = example.AMBIGUOUS_PROMPT
    executed: list[str] = []

    def build_gated_router(threshold: float) -> tuple[JevClient, IntentRouter]:
        client = example.build_client()
        router = IntentRouter(
            client,
            name="gate-demo",
            confidence_threshold=threshold,
            fallback_label="human_handoff",
        )

        @router.tool("search_agent", description="web search that costs money per call")
        async def search_agent(request: IntentRequest) -> dict[str, str]:
            executed.append(request.input)
            return {"tool": "search_agent"}

        @router.fallback(label="human_handoff")
        async def human_handoff(request: IntentRequest) -> dict[str, object]:
            return {"escalated": True, "candidate": request.decision.winner}

        return client, router

    # BEFORE: the gate is disabled, so the 42% guess reaches the handler.
    client, router = build_gated_router(0.0)
    async with client:
        ungated = await router.route(prompt)

    assert ungated.handled is True
    assert ungated.label == "search_agent"
    assert ungated.decision.accepted is True
    assert executed == [prompt]

    # AFTER: the gate rejects the very same guess and escalates instead.
    executed.clear()
    client, router = build_gated_router(0.6)
    async with client:
        gated = await router.route(prompt)

    decision = gated.decision
    assert decision.winner == "search_agent"
    assert decision.accepted is False
    assert decision.outcome is DecisionOutcome.FALLBACK
    assert decision.reason == "confidence 0.420 is below the threshold 0.600"
    assert gated.label == "human_handoff"
    assert gated.handled is True
    assert executed == []


async def test_middleware_pipeline_uses_onion_ordering() -> None:
    """First registered runs first on the way in and last on the way out."""
    events: list[str] = []

    async def outer(call: str, call_next: Callable[[str], Any]) -> str:
        events.append("outer:in")
        result = await call_next(call)
        events.append("outer:out")
        return result

    async def inner(call: str, call_next: Callable[[str], Any]) -> str:
        events.append("inner:in")
        result = await call_next(call)
        events.append("inner:out")
        return result

    async def terminal(call: str) -> str:
        events.append("terminal")
        return f"handled {call}"

    pipeline: MiddlewarePipeline[str, str] = MiddlewarePipeline([outer, inner])
    assert len(pipeline) == 2
    assert await pipeline.run(terminal, "ping") == "handled ping"
    assert events == ["outer:in", "inner:in", "terminal", "inner:out", "outer:out"]


def test_settings_defaults() -> None:
    """The client defaults target the Jev model on OpenRouter."""
    settings = JevSettings()

    assert settings.model == "typesafe/jev-1.13"
    assert settings.confidence_threshold == pytest.approx(0.6)
    assert settings.has_credentials is False
    assert settings.completions_url.endswith("/chat/completions")

