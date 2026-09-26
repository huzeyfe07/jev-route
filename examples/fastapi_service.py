"""Runnable example: embedding JevRoute in a FastAPI service.

The service decides *which* handler should serve an incoming message and only
then runs it:

* ``GET  /healthz`` -- readiness, the model in use and the confidence gate.
* ``GET  /options`` -- the options the engine is allowed to choose from.
* ``POST /route``   -- route one message and return the gate-checked decision.
* ``GET  /audit``   -- every decision seen by the middleware, oldest first.
* ``GET  /refunds`` -- the fake ledger: which refund calls reached the handler.

``refund_tool`` moves money, so it is the one handler that must never see a weak
decision. ``POST /route`` with a confident message (0.91) executes it; the same
endpoint with an ambiguous message whose winner is the *same* tool at 0.44 trips
the confidence gate, escalates to ``human_handoff`` and leaves the ledger empty.

No API key is required: :func:`build_client` replays canned Jev answers through
an :class:`httpx.MockTransport` whenever no credentials are configured, while the
routing, gate and escalation logic stay identical.

Run it with::

    python examples/fastapi_service.py

or::

    uvicorn --app-dir examples fastapi_service:app --reload
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from jev_route import (
    CacheMiddleware,
    DecisionOutcome,
    IntentCall,
    IntentRequest,
    IntentRouter,
    JevClient,
    JevDecision,
    JevRouteError,
    JevSettings,
    LoggingMiddleware,
    MiddlewarePipeline,
    NextStep,
)

logger = logging.getLogger("jev_route.example.fastapi")

#: Confident request: the engine is sure the refund tool owns this message.
REFUND_PROMPT = "Siparişimi iade etmek istiyorum"
#: Ambiguous request: the engine leans towards the refund tool, but only weakly.
AMBIGUOUS_PROMPT = "Müşteri şikayetini bir an önce çözer misin?"
#: A read-only lookup that the same endpoint may route to.
ORDER_STATUS_PROMPT = "A-1042 numaralı siparişim nerede?"
#: Nothing registered can serve this, so the engine abstains itself (noul).
OUT_OF_SCOPE_PROMPT = "Bana kısa bir bilim kurgu hikayesi yaz."

#: Canned Jev decisions replayed by the offline transport, keyed by user input.
MOCK_DECISIONS: dict[str, dict[str, Any]] = {
    REFUND_PROMPT: {
        "winner": "refund_tool",
        "confidence": 0.91,
        "reason": "the user explicitly asks for a refund of their own order",
        "choices": [
            {"label": "refund_tool", "score": 0.91},
            {"label": "order_status_tool", "score": 0.33},
            {"label": "knowledge_base_agent", "score": 0.12},
        ],
        "noul": None,
    },
    AMBIGUOUS_PROMPT: {
        "winner": "refund_tool",
        "confidence": 0.44,
        "reason": "the complaint mentions money, but no order id is available",
        "choices": [
            {"label": "refund_tool", "score": 0.44},
            {"label": "knowledge_base_agent", "score": 0.39},
            {"label": "order_status_tool", "score": 0.17},
        ],
        "noul": None,
    },
    ORDER_STATUS_PROMPT: {
        "winner": "order_status_tool",
        "confidence": 0.86,
        "reason": "the input asks for the delivery status of a specific order id",
        "choices": [
            {"label": "order_status_tool", "score": 0.86},
            {"label": "knowledge_base_agent", "score": 0.21},
            {"label": "refund_tool", "score": 0.08},
        ],
        "noul": None,
    },
    OUT_OF_SCOPE_PROMPT: {
        "winner": None,
        "confidence": 0.0,
        "reason": "none of the registered options author fiction",
        "choices": [
            {"label": "knowledge_base_agent", "score": 0.19},
            {"label": "order_status_tool", "score": 0.04},
            {"label": "refund_tool", "score": 0.02},
        ],
        "noul": {"reason": "no registered option fits a creative writing request"},
    },
    "__default__": {
        "winner": "knowledge_base_agent",
        "confidence": 0.72,
        "reason": "a general support question, best served by the knowledge base",
        "choices": [
            {"label": "knowledge_base_agent", "score": 0.72},
            {"label": "order_status_tool", "score": 0.19},
            {"label": "refund_tool", "score": 0.06},
        ],
        "noul": None,
    },
}


def _extract_input(request_body: Mapping[str, Any]) -> str:
    """Read the routed message back out of the completion request."""
    for message in request_body.get("messages", []):
        if message.get("role") != "user":
            continue
        content = str(message.get("content", ""))
        if '"""' in content:
            return content.split('"""')[1].strip()
        return content.strip()
    return ""


def build_mock_transport() -> httpx.MockTransport:
    """Replay canned Jev answers so the service runs without credentials."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        prompt = _extract_input(body)
        decision = MOCK_DECISIONS.get(prompt, MOCK_DECISIONS["__default__"])
        return httpx.Response(
            200,
            json={
                "id": f"mock-{abs(hash(prompt)) % 100000:05d}",
                "model": body.get("model", "typesafe/jev-1.13"),
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(decision)},
                    }
                ],
                "usage": {"prompt_tokens": 164, "completion_tokens": 28, "total_tokens": 192},
            },
        )

    return httpx.MockTransport(handler)


def build_client() -> JevClient:
    """Talk to the real engine when a key exists, otherwise replay canned answers."""
    settings = JevSettings.from_env()
    if settings.has_credentials:
        logger.info("Using the live Jev engine at %s", settings.completions_url)
        return JevClient(settings)
    logger.info("No API key configured: replaying canned Jev answers through MockTransport")
    return JevClient(settings, transport=build_mock_transport())


class AuditMiddleware:
    """Async middleware that records every gate-checked decision.

    ``target`` is the label the router is about to execute: the winner when the
    gate accepted it, the fallback label otherwise. Recording it here keeps the
    audit trail independent of the endpoint code.
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
                "message": call.text,
                "winner": decision.winner,
                "accepted": decision.accepted,
                "outcome": decision.outcome.value,
                "target": decision.target,
                "score": round(decision.score, 3),
                "threshold": decision.threshold,
                "reason": decision.reason,
                "cache_hit": call.annotations.get("cache_hit"),
                "latency_ms": call.annotations.get("latency_ms"),
            }
        )
        return decision


class RouteRequest(BaseModel):
    """One message to route, plus the state handed to the chosen handler."""

    message: str = Field(min_length=1, max_length=4000)
    context: dict[str, Any] = Field(default_factory=dict)


class RouteResponse(BaseModel):
    """The gate-checked outcome of one routing call."""

    label: str | None
    handled: bool
    winner: str | None
    accepted: bool
    escalated: bool
    outcome: str
    score: float
    threshold: float
    reason: str | None = None
    value: Any = None


def create_app(
    client: JevClient | None = None,
    *,
    confidence_threshold: float = 0.6,
) -> FastAPI:
    """Build the service: Jev client, router, middleware and endpoints.

    Args:
        client: The Jev client to route with; a fresh one is built when omitted.
        confidence_threshold: The gate. A winner below it escalates to the
            fallback handler instead of being executed.

    Returns:
        A ready-to-serve FastAPI application.
    """
    jev = client if client is not None else build_client()
    audit: list[dict[str, Any]] = []
    # The demo ledger: calls that actually reached the privileged handler.
    refund_calls: list[dict[str, Any]] = []

    middleware: MiddlewarePipeline[IntentCall, JevDecision] = MiddlewarePipeline(
        [AuditMiddleware(audit), CacheMiddleware(ttl=30.0), LoggingMiddleware()]
    )
    router = IntentRouter(
        jev,
        name="support-desk",
        confidence_threshold=confidence_threshold,
        middleware=middleware,
        fallback_label="human_handoff",
    )

    @router.tool(
        "refund_tool",
        description="issue a refund for an order id; this moves money",
        examples=(REFUND_PROMPT,),
    )
    async def refund_tool(request: IntentRequest) -> dict[str, Any]:
        """Privileged handler: the confidence gate is what protects it."""
        order_id = str(request.context.get("order_id", "")).strip()
        refund_calls.append({"order_id": order_id, "confidence": round(request.score, 3)})
        if not order_id:
            return {"tool": "refund_tool", "issued": False, "error": "no order_id in context"}
        return {
            "tool": "refund_tool",
            "issued": True,
            "order_id": order_id,
            "amount": 149.90,
            "currency": "TRY",
        }

    @router.tool(
        "order_status_tool",
        description="read-only lookup of the delivery status of an order id",
        examples=(ORDER_STATUS_PROMPT,),
    )
    async def order_status_tool(request: IntentRequest) -> dict[str, Any]:
        """A read-only tool: harmless, but the same gate applies to it."""
        order_id = str(request.context.get("order_id", "A-1042"))
        return {"tool": "order_status_tool", "order_id": order_id, "status": "in_transit"}

    @router.intent(
        "knowledge_base_agent",
        description="answer general support questions from the help centre",
    )
    async def knowledge_base_agent(request: IntentRequest) -> dict[str, Any]:
        """The catch-all agent for questions that fit no tool."""
        return {
            "agent": "knowledge_base_agent",
            "answer": "See https://help.example.com/ for the matching article.",
        }

    @router.fallback(label="human_handoff")
    async def human_handoff(request: IntentRequest) -> dict[str, Any]:
        """Escalation: the gate rejected the winner, or the engine abstained."""
        return {
            "agent": "human_handoff",
            "escalated": True,
            "candidate": request.decision.winner,
            "reason": request.decision.reason,
        }

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Open the HTTP connection pool for the lifetime of the service."""
        async with jev:
            yield

    app = FastAPI(title="jev-route FastAPI example", version="0.2.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        """Readiness plus the routing configuration in use."""
        return {
            "status": "ok",
            "model": jev.model,
            "confidence_threshold": router.confidence_threshold,
            "fallback_label": router.fallback_label,
            "offline_mock": not jev.settings.has_credentials,
        }

    @app.get("/options")
    async def options() -> list[dict[str, str]]:
        """The options the engine may choose from."""
        return [
            {"label": option.label, "kind": option.kind, "description": option.description}
            for option in router
        ]

    @app.post("/route", response_model=RouteResponse)
    async def route_message(payload: RouteRequest) -> RouteResponse:
        """Route one message; a weak decision escalates instead of executing."""
        try:
            result = await router.route(payload.message, context=payload.context)
        except JevRouteError as exc:
            logger.warning("Routing failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc

        decision = result.decision
        return RouteResponse(
            label=result.label,
            handled=result.handled,
            winner=decision.winner,
            accepted=decision.accepted,
            escalated=decision.outcome is DecisionOutcome.FALLBACK,
            outcome=decision.outcome.value,
            score=round(decision.score, 3),
            threshold=decision.threshold,
            reason=decision.reason,
            value=result.value,
        )

    @app.get("/audit")
    async def read_audit() -> list[dict[str, Any]]:
        """Every decision the middleware saw, oldest first."""
        return audit

    @app.get("/refunds")
    async def read_refunds() -> dict[str, Any]:
        """The demo ledger: calls that reached the privileged refund handler."""
        return {"calls": refund_calls, "count": len(refund_calls)}

    return app


#: Module-level application, so ``uvicorn fastapi_service:app`` works directly.
app = create_app()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print("Serving the jev-route FastAPI example on http://127.0.0.1:8000")
    print("  POST /route  {'message': 'Siparişimi iade etmek istiyorum'}")
    print("  GET  /options /audit /refunds /healthz")
    uvicorn.run(app, host="127.0.0.1", port=8000)
