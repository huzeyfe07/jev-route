"""Asynchronous middleware for intent resolution and agent loops.

The pipeline in this module is deliberately generic: a middleware wraps any
awaitable step and may inspect or replace both the call it receives and the
result it returns. The same machinery therefore serves the two stages JevRoute
cares about:

* **Intent resolution** -- wrapping ``IntentCall -> JevDecision``, so that
  caching, auditing, budget checks or tracing can short-circuit the Jev call.
* **Agent loops** -- wrapping ``AgentTurn -> AgentTurnResult``, so that the very
  same policies apply to every hop of a multi-step agent.

Middlewares are coroutines invoked in registration order: the first one
registered is the outermost layer, so it observes the call first and the result
last (the classic "onion" model).
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable, Iterator, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import (
    Any,
    Protocol,
    TypeVar,
    overload,
    runtime_checkable,
)

from .core import JevDecision, OptionSpec

__all__ = [
    "CacheMiddleware",
    "IntentCall",
    "IntentMiddleware",
    "LoggingMiddleware",
    "Middleware",
    "MiddlewarePipeline",
    "NextStep",
]

CallT = TypeVar("CallT")
ResultT = TypeVar("ResultT")

#: An awaitable step that receives a call and produces a result.
NextStep = Callable[[CallT], Awaitable[ResultT]]
#: An async middleware: the call, plus the step it delegates to.
Middleware = Callable[[CallT, NextStep[CallT, ResultT]], Awaitable[ResultT]]


@dataclass
class IntentCall:
    """The envelope that flows through the intent resolution pipeline.

    Middlewares may rewrite :attr:`text` or :attr:`options` before delegating,
    short-circuit the chain with a cached decision, or annotate the call for
    observability.

    Attributes:
        text: The natural-language input being routed.
        options: The options offered to the engine for this call.
        context: Optional state forwarded to the engine (agent state, user
            profile, conversation summary, ...).
        annotations: A scratch pad for middlewares (timings, cache flags, guard
            verdicts). It never reaches the engine.
    """

    text: str
    options: tuple[OptionSpec, ...] = ()
    context: MutableMapping[str, Any] = field(default_factory=dict)
    annotations: MutableMapping[str, Any] = field(default_factory=dict)

    def annotate(self, key: str, value: Any) -> IntentCall:
        """Record a middleware annotation and return ``self`` for chaining."""
        self.annotations[key] = value
        return self

    def with_text(self, text: str) -> IntentCall:
        """Return a shallow copy of this call carrying a different input."""
        return IntentCall(
            text=text,
            options=self.options,
            context=dict(self.context),
            annotations=dict(self.annotations),
        )

    @property
    def labels(self) -> tuple[str, ...]:
        """The labels of the options offered in this call."""
        return tuple(option.label for option in self.options)


@runtime_checkable
class IntentMiddleware(Protocol):
    """Structural type for a middleware that operates on intent decisions."""

    async def __call__(
        self,
        call: IntentCall,
        call_next: NextStep[IntentCall, JevDecision],
    ) -> JevDecision:
        ...  # pragma: no cover - protocol definition


class MiddlewarePipeline(Sequence[Middleware[CallT, ResultT]]):
    """An ordered collection of asynchronous middlewares.

    The pipeline is generic over the call and result types, so a single
    implementation serves intent resolution (``IntentCall -> JevDecision``) and
    agent loops (``AgentTurn -> AgentTurnResult``) alike.

    Example:
        >>> pipeline: MiddlewarePipeline[IntentCall, JevDecision] = MiddlewarePipeline(
        ...     [CacheMiddleware(ttl=30.0), LoggingMiddleware()]
        ... )
        >>> decision = await pipeline.run(decide, IntentCall(text="How is the weather?"))
    """

    def __init__(self, middlewares: Iterable[Middleware[CallT, ResultT]] = ()) -> None:
        """Create a pipeline from *middlewares*, in outermost-first order."""
        self._middlewares: list[Middleware[CallT, ResultT]] = list(middlewares)

    def add(
        self,
        middleware: Middleware[CallT, ResultT],
    ) -> MiddlewarePipeline[CallT, ResultT]:
        """Append *middleware* to the pipeline and return ``self`` for chaining."""
        self._middlewares.append(middleware)
        return self

    def extend(
        self,
        middlewares: Iterable[Middleware[CallT, ResultT]],
    ) -> MiddlewarePipeline[CallT, ResultT]:
        """Append every item of *middlewares* and return ``self``."""
        self._middlewares.extend(middlewares)
        return self

    @staticmethod
    def _bind(
        middleware: Middleware[CallT, ResultT],
        next_step: NextStep[CallT, ResultT],
    ) -> NextStep[CallT, ResultT]:
        """Bind one middleware to the step that follows it."""

        async def bound(call: CallT) -> ResultT:
            return await middleware(call, next_step)

        return bound

    def wrap(self, terminal: NextStep[CallT, ResultT]) -> NextStep[CallT, ResultT]:
        """Wrap *terminal* with every middleware of this pipeline.

        Args:
            terminal: The innermost step, e.g. the Jev call itself or the final
                hop of an agent loop.

        Returns:
            The composed step, ready to be awaited with a call object.
        """
        next_step: NextStep[CallT, ResultT] = terminal
        for middleware in reversed(self._middlewares):
            next_step = self._bind(middleware, next_step)
        return next_step

    async def run(self, terminal: NextStep[CallT, ResultT], call: CallT) -> ResultT:
        """Wrap *terminal*, await it with *call* and return the final result."""
        return await self.wrap(terminal)(call)

    def __len__(self) -> int:
        return len(self._middlewares)

    def __iter__(self) -> Iterator[Middleware[CallT, ResultT]]:
        return iter(self._middlewares)

    @overload
    def __getitem__(self, index: int) -> Middleware[CallT, ResultT]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Middleware[CallT, ResultT]]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> Middleware[CallT, ResultT] | Sequence[Middleware[CallT, ResultT]]:
        return self._middlewares[index]

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        names = ", ".join(type(middleware).__name__ for middleware in self._middlewares)
        return f"{type(self).__name__}([{names}])"


class LoggingMiddleware:
    """Async middleware that logs every resolved intent.

    This is the canonical "wrapper" middleware: it delegates to the next step,
    measures the latency, records the outcome and re-raises failures after
    logging them. It is typed for intent resolution
    (``IntentCall -> JevDecision``).
    """

    def __init__(self, logger: logging.Logger | None = None, level: int = logging.INFO) -> None:
        """Create the middleware.

        Args:
            logger: The logger to emit to; defaults to ``logging.getLogger("jev_route")``.
            level: The level used for the success record.
        """
        self._logger = logger or logging.getLogger("jev_route")
        self._level = level

    @property
    def logger(self) -> logging.Logger:
        """The underlying logger instance."""
        return self._logger

    async def __call__(
        self,
        call: IntentCall,
        call_next: NextStep[IntentCall, JevDecision],
    ) -> JevDecision:
        started_at = time.perf_counter()
        try:
            decision = await call_next(call)
        except Exception:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            self._logger.exception(
                "Intent routing failed after %.2f ms for input %r", elapsed_ms, call.text[:80]
            )
            raise
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        call.annotate("latency_ms", round(elapsed_ms, 3))
        self._logger.log(
            self._level,
            "Resolved %r to %s (outcome=%s, score=%.3f, threshold=%.3f, %.2f ms)",
            call.text[:80],
            decision.target or "-",
            decision.outcome.value,
            decision.score,
            decision.threshold,
            elapsed_ms,
        )
        return decision


class CacheMiddleware:
    """Async middleware that caches intent decisions for a short time to live.

    Routing the same input twice is common (retries, agent loops, tests), so an
    in-process LRU with a TTL removes an entire engine round trip. A cache hit
    short-circuits the chain entirely and annotates the call with
    ``cache_hit=True``.

    Note:
        The default cache key combines the trimmed input text with the offered
        option labels. The context is deliberately ignored because it is rarely
        hashable and often noisy; pass a ``key_builder`` when it matters.
    """

    def __init__(
        self,
        ttl: float = 60.0,
        max_entries: int = 128,
        *,
        key_builder: Callable[[IntentCall], str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create the middleware.

        Args:
            ttl: Seconds a cached decision stays valid.
            max_entries: Hard cap on cached decisions (least recently used first).
            key_builder: Optional callable computing the cache key for a call.
            clock: Monotonic clock, injectable for deterministic tests.

        Raises:
            ValueError: If *ttl* or *max_entries* is not positive.
        """
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._ttl = float(ttl)
        self._max_entries = int(max_entries)
        self._key_builder = key_builder
        self._clock = clock
        self._entries: OrderedDict[str, tuple[float, JevDecision]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def ttl(self) -> float:
        """The time to live of a cached decision, in seconds."""
        return self._ttl

    def build_key(self, call: IntentCall) -> str:
        """Compute the cache key for *call*."""
        if self._key_builder is not None:
            return self._key_builder(call)
        return "|".join((call.text.strip(), *call.labels))

    def clear(self) -> None:
        """Drop every cached decision."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    async def __call__(
        self,
        call: IntentCall,
        call_next: NextStep[IntentCall, JevDecision],
    ) -> JevDecision:
        key = self.build_key(call)
        now = self._clock()
        cached_entry = self._entries.get(key)
        if cached_entry is not None:
            expires_at, cached_decision = cached_entry
            if expires_at > now:
                self._entries.move_to_end(key)
                self.hits += 1
                call.annotate("cache_hit", True)
                return cached_decision.model_copy(deep=True)
            del self._entries[key]

        self.misses += 1
        call.annotate("cache_hit", False)
        decision = await call_next(call)
        self._entries[key] = (now + self._ttl, decision.model_copy(deep=True))
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
        return decision