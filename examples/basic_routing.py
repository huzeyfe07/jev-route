"""Runnable example: routing a user prompt to the right AI agent with JevRoute.

Two modes:

* **offline (default)** -- an :class:`httpx.MockTransport` replays canned Jev
  answers, so the example runs without an API key.
* **live** -- when ``OPENROUTER_API_KEY`` (or ``JEV_API_KEY``) is set, the very
  same code path calls the real Jev engine ``typesafe/jev-1.13`` on OpenRouter.

Run it with::

    python examples/basic_routing.py
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import Any

import httpx

from jev_route import (
    IntentCall,
    IntentRequest,
    IntentResult,
    IntentRouter,
    JevClient,
    JevDecision,
    JevRouteError,
    JevSettings,
    LoggingMiddleware,
    MiddlewarePipeline,
    NextStep,
)

WEATHER_PROMPT = "Hava durumu nasıl?"
MATH_PROMPT = "12 çarpı 8 kaç eder?"
AMBIGUOUS_PROMPT = "Şirketimizin cirosu ne kadar?"
OUT_OF_SCOPE_PROMPT = "Bana kısa bir bilim kurgu hikayesi yaz."

#: Canned Jev decisions keyed by user input, replayed by the offline transport.
MOCK_DECISIONS: dict[str, dict[str, Any]] = {
    # A confident, well separated winner: routed straight to the weather agent.
    WEATHER_PROMPT: {
        "winner": "weather_agent",
        "confidence": 0.94,
        "reason": "the input asks for the current weather, which the weather agent owns",
        "choices": [
            {"label": "weather_agent", "score": 0.94},
            {"label": "search_agent", "score": 0.38},
            {"label": "calculator_tool", "score": 0.02},
        ],
        "noul": None,
    },
    MATH_PROMPT: {
        "winner": "calculator_tool",
        "confidence": 0.91,
        "reason": "the input is plain arithmetic",
        "choices": [
            {"label": "calculator_tool", "score": 0.91},
            {"label": "search_agent", "score": 0.24},
            {"label": "weather_agent", "score": 0.01},
        ],
        "noul": None,
    },
    # A weak winner: the confidence gate must trip and trigger the fallback.
    AMBIGUOUS_PROMPT: {
        "winner": "search_agent",
        "confidence": 0.42,
        "reason": "revenue is mentioned but no financial data source is registered",
        "choices": [
            {"label": "search_agent", "score": 0.42},
            {"label": "calculator_tool", "score": 0.27},
            {"label": "weather_agent", "score": 0.03},
        ],
        "noul": None,
    },
    # The engine abstains itself, reporting a noul (null outcome).
    OUT_OF_SCOPE_PROMPT: {
        "winner": None,
        "confidence": 0.0,
        "reason": "none of the registered options can author fiction",
        "choices": [
            {"label": "search_agent", "score": 0.18},
            {"label": "weather_agent", "score": 0.05},
            {"label": "calculator_tool", "score": 0.01},
        ],
        "noul": {"reason": "no option fits a creative writing request"},
    },
    "__default__": {
        "winner": "search_agent",
        "confidence": 0.71,
        "reason": "general question, best served by a web search",
        "choices": [
            {"label": "search_agent", "score": 0.71},
            {"label": "weather_agent", "score": 0.11},
            {"label": "calculator_tool", "score": 0.04},
        ],
        "noul": None,
    },
}


def _extract_input(request_body: Mapping[str, Any]) -> str:
    """Read the user input back out of the completion request the client built."""
    for message in request_body.get("messages", []):
        if message.get("role") != "user":
            continue
        content = str(message.get("content", ""))
        if '"""' in content:
            return content.split('"""')[1].strip()
        return content.strip()
    return ""


def build_mock_transport() -> httpx.MockTransport:
    """Replay canned Jev answers so the example runs without credentials."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        prompt = _extract_input(body)
        decision = MOCK_DECISIONS.get(prompt, MOCK_DECISIONS["__default__"])
        payload = {
            "id": f"mock-{abs(hash(prompt)) % 100000:05d}",
            "model": body.get("model", "typesafe/jev-1.13"),
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps(decision)},
                }
            ],
            "usage": {"prompt_tokens": 128, "completion_tokens": 32, "total_tokens": 160},
        }
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


def build_client() -> JevClient:
    """Call the real engine when an API key exists, otherwise use the mock."""
    settings = JevSettings.from_env()
    if settings.has_credentials:
        return JevClient(settings)
    return JevClient(settings, transport=build_mock_transport())


class AuditMiddleware:
    """A tiny async middleware that records every decision it sees.

    It shows two things at once: middlewares are plain coroutines, and they can
    annotate the :class:`IntentCall` (here the cache flag and the latency stored
    by :class:`LoggingMiddleware`) for downstream consumers.
    """

    def __init__(self, sink: list[dict[str, Any]]) -> None:
        self._sink = sink

    async def __call__(
        self,
        call: IntentCall,
        call_next: NextStep[IntentCall, JevDecision],
    ) -> JevDecision:
        decision = await call_next(call)
        self._sink.append(
            {
                "input": call.text,
                "winner": decision.winner,
                "outcome": decision.outcome.value,
                "target": decision.target,
                "score": round(decision.score, 3),
                "cache_hit": call.annotations.get("cache_hit"),
                "latency_ms": call.annotations.get("latency_ms"),
            }
        )
        return decision


def build_router(client: JevClient, audit_log: list[dict[str, Any]]) -> IntentRouter:
    """Register the agents and tools of the demo, then wire up the middleware."""
    middleware: MiddlewarePipeline[IntentCall, JevDecision] = MiddlewarePipeline(
        [AuditMiddleware(audit_log), LoggingMiddleware()]
    )
    router = IntentRouter(
        client,
        name="support-desk",
        confidence_threshold=0.6,
        middleware=middleware,
        fallback_label="human_handoff",
    )

    @router.intent(
        "weather_agent",
        description="current weather and short forecasts for a city",
        examples=(WEATHER_PROMPT, "Yarın yağmur var mı?"),
    )
    async def weather_agent(request: IntentRequest) -> dict[str, Any]:
        """Serve a weather question (the forecast itself is faked here)."""
        city = request.context.get("city", "Istanbul")
        return {
            "agent": "weather_agent",
            "city": city,
            "forecast": "partly cloudy, 21 C",
            "confidence": round(request.score, 3),
        }

    @router.tool(
        "calculator_tool",
        description="arithmetic on plain numbers; needs no external data",
        examples=(MATH_PROMPT,),
    )
    def calculator_tool(request: IntentRequest) -> dict[str, Any]:
        """A synchronous handler: plain functions are supported as well."""
        return {"tool": "calculator_tool", "expression": "12 * 8", "result": 96}

    @router.tool(
        "search_agent",
        description="web search for facts, news and general knowledge",
    )
    async def search_agent(request: IntentRequest) -> dict[str, Any]:
        """Serve a general question with a (faked) list of web results."""
        return {
            "agent": "search_agent",
            "query": request.input,
            "results": ["https://example.com/first-hit", "https://example.com/second-hit"],
        }

    @router.fallback(label="human_handoff")
    async def human_handoff(request: IntentRequest) -> dict[str, Any]:
        """Escalate when the engine abstains or the confidence gate trips."""
        return {
            "agent": "human_handoff",
            "escalated": True,
            "why": request.decision.reason,
        }

    return router


async def run_agent_loop_demo(turns: list[str]) -> None:
    """Show the same pipeline wrapping an agent loop instead of intent routing.

    :class:`MiddlewarePipeline` is generic over the call and result types, so
    policies such as logging, budgets, tracing or retries are written once and
    reused for both stages.
    """

    async def step(turn: str) -> str:
        return f"handled {turn!r}"

    async def trace(turn: str, call_next: NextStep[str, str]) -> str:
        result = await call_next(turn)
        print(f"    [loop middleware] {turn!r} -> {result!r}")
        return result

    pipeline: MiddlewarePipeline[str, str] = MiddlewarePipeline([trace])
    for turn in turns:
        await pipeline.run(step, turn)


async def main() -> None:
    """Route a few prompts through Jev and print every decision."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    audit_log: list[dict[str, Any]] = []
    client = build_client()
    mode = "live OpenRouter" if client.settings.has_credentials else "offline mock transport"
    print(f"JevRoute example -- {mode}, model={client.model}\n")

    router = build_router(client, audit_log)
    print("Registered options:")
    for option in router:
        print(f"  - {option.label} [{option.kind}] {option.description}")
    print(
        f"Confidence threshold: {router.confidence_threshold:.2f}, "
        f"fallback label: {router.fallback_label}\n"
    )

    prompts = [WEATHER_PROMPT, MATH_PROMPT, AMBIGUOUS_PROMPT, OUT_OF_SCOPE_PROMPT]
    context = {"city": "Istanbul", "user_tier": "pro"}

    async with client:
        for prompt in prompts:
            print(f"--> user: {prompt}")
            try:
                result: IntentResult = await router.route(prompt, context=context)
            except JevRouteError as error:
                print(f"    routing error: {error}\n")
                continue

            decision = result.decision
            print(
                f"    winner={decision.winner!r} score={decision.score:.3f} "
                f"threshold={decision.threshold:.2f} accepted={decision.accepted}"
            )
            print(f"    outcome={decision.outcome.value} reason={decision.reason}")
            print(f"    ranking={[str(choice) for choice in decision.choices]}")
            print(f"    executed={result.label!r} handled={result.handled}")
            print(f"    handler returned: {result.value}")
            if decision.response is not None:
                print(
                    f"    engine latency={decision.response.latency_ms:.1f} ms "
                    f"request_id={decision.response.request_id}"
                )
            print()

    print("Audit trail collected by the middleware:")
    for entry in audit_log:
        print(f"  {entry}")

    print("\nAgent loop wrapped with MiddlewarePipeline[str, str]:")
    await run_agent_loop_demo(["plan", "act", "observe"])


if __name__ == "__main__":
    asyncio.run(main())