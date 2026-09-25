"""Intent routing for :mod:`jev_route`.

:class:`IntentRouter` is the primary entry point of the library. Register the
intents, agents and tools you own, then hand it the input of an agent loop: the
router asks the Jev engine which option fits best, applies the confidence gate
and invokes the matching handler.

Nothing here deals with URLs -- an "intent" is a routing decision target, not a
path segment.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from .core import (
    ConfidenceTooLowError,
    DecisionOutcome,
    IntentHandlerError,
    IntentNotRegisteredError,
    JevClient,
    JevConfigurationError,
    JevDecision,
    JevResponse,
    JevRouteError,
    OptionSpec,
    coerce_options,
)
from .middleware import IntentCall, MiddlewarePipeline

__all__ = ["IntentOption", "IntentRequest", "IntentResult", "IntentRouter"]

logger = logging.getLogger("jev_route.router")

#: An intent handler. It may be a plain function or a coroutine function.
IntentHandler = Callable[["IntentRequest"], Any]
#: The pipeline shape used for intent resolution.
IntentPipeline = MiddlewarePipeline[IntentCall, JevDecision]


@dataclass(frozen=True)
class IntentOption:
    """A routing target registered on an :class:`IntentRouter`.

    Attributes:
        label: The identifier the engine returns as ``winner``.
        handler: The callable serving the input. ``None`` marks a decision-only
            option, where the router reports the decision without running code.
        description: One line explaining when this option applies. It is sent to
            the engine, so keep it short and discriminative.
        examples: Example inputs that should route here.
        metadata: Free-form extras forwarded to the prompt and to the handlers.
    """

    label: str
    handler: IntentHandler | None = None
    description: str = ""
    examples: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_spec(self) -> OptionSpec:
        """Convert this option into the description offered to the engine."""
        return OptionSpec(
            label=self.label,
            description=self.description,
            examples=self.examples,
            metadata=self.metadata,
        )

    @property
    def executable(self) -> bool:
        """Return ``True`` when a handler is attached to this option."""
        return self.handler is not None

    @property
    def kind(self) -> str:
        """The option kind (``agent``, ``tool``, ``router``, ...)."""
        return str(self.metadata.get("kind", "intent"))


@dataclass
class IntentRequest:
    """The payload handed to an intent handler.

    Attributes:
        input: The original user input that was routed.
        decision: The full decision, including scores and runner-up choices.
        option: The option whose handler is being invoked.
        context: The state that travelled with the call.
    """

    input: str
    decision: JevDecision
    option: IntentOption | None = None
    context: MutableMapping[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str | None:
        """The label of the option being served, when known."""
        return self.option.label if self.option is not None else self.decision.target

    @property
    def score(self) -> float:
        """The confidence the engine reported for this input."""
        return self.decision.score

    @property
    def metadata(self) -> Mapping[str, Any]:
        """The metadata of the option being served."""
        return self.option.metadata if self.option is not None else {}


@dataclass
class IntentResult:
    """The outcome of :meth:`IntentRouter.route`.

    Attributes:
        decision: The gated decision produced by the engine.
        value: Whatever the handler returned (``None`` when nothing was run).
        handled: ``True`` when a handler served the input.
        label: The label that served the input, or was supposed to.
    """

    decision: JevDecision
    value: Any = None
    handled: bool = False
    label: str | None = None

    @property
    def outcome(self) -> DecisionOutcome:
        """The outcome of the underlying decision."""
        return self.decision.outcome

    @property
    def score(self) -> float:
        """The confidence of the underlying decision."""
        return self.decision.score

    def describe(self) -> str:
        """Return a one-line human readable summary of the result."""
        state = "handled" if self.handled else "not handled"
        return f"{self.label or '-'} ({self.outcome.value}, {state}, score={self.score:.3f})"


class IntentRouter:
    """Routes natural-language inputs to intents, agents, tools and sub-routers.

    The router owns the option registry, the confidence gate and the optional
    middleware pipeline. It is transport agnostic: give it a
    :class:`~jev_route.core.JevClient` -- real or backed by a mock transport --
    and the behaviour is identical.

    Example:
        >>> router = IntentRouter(JevClient(), confidence_threshold=0.6)
        >>> @router.tool("weather_agent", description="current weather and forecasts")
        ... async def weather_agent(request: IntentRequest) -> dict[str, str]:
        ...     return {"forecast": "sunny"}
        >>> result = await router.route("How is the weather?")
        >>> result.value
        {'forecast': 'sunny'}
    """

    def __init__(
        self,
        client: JevClient,
        *,
        confidence_threshold: float | None = None,
        middleware: IntentPipeline | None = None,
        fallback_label: str = "fallback",
        offer_fallback: bool = False,
        system_prompt: str | None = None,
        name: str = "",
    ) -> None:
        """Create an empty router.

        Args:
            client: The Jev client used to obtain decisions.
            confidence_threshold: Minimum score accepted as a routing decision;
                defaults to ``client.settings.confidence_threshold``.
            middleware: Pipeline wrapped around every decision; an empty
                pipeline is created when omitted.
            fallback_label: The label used when the confidence gate rejects a
                winner.
            offer_fallback: When ``True`` the fallback is also offered to the
                engine, so it can abstain on its own terms. When ``False`` (the
                default) the fallback stays invisible and only the gate decides.
            system_prompt: Overrides the engine's built-in routing system prompt.
            name: A human readable router name used in diagnostics and by
                :meth:`include_router`.
        """
        self.client = client
        self.name = name
        self.system_prompt = system_prompt
        self.offer_fallback = offer_fallback
        threshold = confidence_threshold
        if threshold is None:
            threshold = client.settings.confidence_threshold
        self.confidence_threshold = float(threshold)
        self.middleware: IntentPipeline = (
            middleware if middleware is not None else MiddlewarePipeline()
        )
        self._options: dict[str, IntentOption] = {}
        self._fallback_label = fallback_label
        self._fallback_handler: IntentHandler | None = None

    @property
    def options(self) -> tuple[IntentOption, ...]:
        """The registered options, in registration order."""
        return tuple(self._options.values())

    @property
    def labels(self) -> tuple[str, ...]:
        """The labels of the registered options."""
        return tuple(self._options)

    @property
    def specs(self) -> tuple[OptionSpec, ...]:
        """The option descriptions offered to the engine."""
        return tuple(option.to_spec() for option in self._options.values())

    @property
    def fallback_label(self) -> str:
        """The label used when the confidence gate rejects a winner."""
        return self._fallback_label

    @property
    def fallback_handler(self) -> IntentHandler | None:
        """The handler invoked on the fallback path, when one is configured."""
        return self._fallback_handler

    def option(self, label: str) -> IntentOption:
        """Return the registered option named *label*.

        Raises:
            IntentNotRegisteredError: If *label* is unknown.
        """
        try:
            return self._options[label]
        except KeyError as exc:
            raise IntentNotRegisteredError(label, self.labels) from exc

    def __contains__(self, label: object) -> bool:
        return isinstance(label, str) and label in self._options

    def __len__(self) -> int:
        return len(self._options)

    def __iter__(self) -> Iterator[IntentOption]:
        return iter(self._options.values())

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"IntentRouter(name={self.name!r}, intents={len(self._options)}, "
            f"threshold={self.confidence_threshold:.2f})"
        )

    def register(
        self,
        label: str,
        handler: IntentHandler | None = None,
        *,
        description: str = "",
        examples: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Register a routing option, directly or as a decorator.

        Direct call::

            router.register("weather_agent", weather_agent, description="...")

        Decorator::

            @router.register("weather_agent", description="...")
            def weather_agent(request: IntentRequest) -> dict[str, str]:
                ...

        Args:
            label: The identifier the engine returns as ``winner``.
            handler: The callable serving the input.
            description: Short explanation sent to the engine.
            examples: Example inputs that should route here.
            metadata: Free-form extras (kind, cost, owner, ...).

        Returns:
            The created :class:`IntentOption`, or the decorator when no handler
            was passed.

        Raises:
            JevConfigurationError: If the label is invalid or already registered.
        """
        if handler is None:

            def decorator(func: IntentHandler) -> IntentHandler:
                self._add(
                    label,
                    func,
                    description=description,
                    examples=examples,
                    metadata=metadata,
                )
                return func

            return decorator

        return self._add(
            label,
            handler,
            description=description,
            examples=examples,
            metadata=metadata,
        )

    def _add(
        self,
        label: str,
        handler: IntentHandler | None,
        *,
        description: str,
        examples: Sequence[str] | None,
        metadata: Mapping[str, Any] | None,
    ) -> IntentOption:
        """Validate and store a new option."""
        try:
            spec = OptionSpec(
                label=label,
                description=description,
                examples=tuple(examples or ()),
                metadata=dict(metadata or {}),
            )
        except ValidationError as exc:
            raise JevConfigurationError(f"Invalid intent label {label!r}: {exc}") from exc

        if spec.label in self._options:
            raise JevConfigurationError(f"Intent {spec.label!r} is already registered")

        option = IntentOption(
            label=spec.label,
            handler=handler,
            description=spec.description,
            examples=spec.examples,
            metadata=spec.metadata,
        )
        self._options[option.label] = option
        return option

    def unregister(self, label: str) -> IntentOption:
        """Remove *label* from the registry and return the removed option.

        Raises:
            IntentNotRegisteredError: If *label* is unknown.
        """
        self.option(label)
        return self._options.pop(label)

    def intent(
        self,
        label: str,
        *,
        description: str = "",
        examples: Sequence[str] | None = None,
    ) -> Any:
        """Decorator registering a specialised *agent* intent."""
        return self.register(
            label,
            description=description,
            examples=examples,
            metadata={"kind": "agent"},
        )

    def tool(
        self,
        label: str,
        *,
        description: str = "",
        examples: Sequence[str] | None = None,
    ) -> Any:
        """Decorator registering a deterministic *tool* the engine may call."""
        return self.register(
            label,
            description=description,
            examples=examples,
            metadata={"kind": "tool"},
        )

    def set_fallback(
        self,
        handler: IntentHandler | str,
        *,
        label: str | None = None,
        description: str = "",
    ) -> str:
        """Configure the target used when the confidence gate rejects a winner.

        Args:
            handler: A handler to invoke, or the label of an already registered
                intent that should take over.
            label: The label to register the handler under; defaults to the
                router's current fallback label.
            description: Description used when the fallback is offered to the
                engine (see ``offer_fallback``).

        Returns:
            The resolved fallback label.

        Raises:
            IntentNotRegisteredError: If *handler* is a label that is not registered.
        """
        if isinstance(handler, str):
            self.option(handler)
            self._fallback_label = handler
            self._fallback_handler = None
            return handler

        resolved = label or self._fallback_label
        self._fallback_label = resolved
        self._fallback_handler = handler
        if self.offer_fallback and resolved not in self._options:
            self._add(
                resolved,
                handler,
                description=description
                or "abstain and escalate to a human operator or a general-purpose agent",
                examples=None,
                metadata={"kind": "fallback"},
            )
        return resolved

    def fallback(
        self,
        handler: IntentHandler | None = None,
        *,
        label: str | None = None,
        description: str = "",
    ) -> Any:
        """Decorator (or direct call) configuring the fallback handler.

        ::

            @router.fallback(label="human_handoff")
            async def human_handoff(request: IntentRequest) -> dict[str, str]:
                return {"escalated": True}

        Args:
            handler: The fallback handler when called directly.
            label: The label to register the handler under.
            description: Description used when the fallback is offered to the engine.
        """
        if handler is None:

            def decorator(func: IntentHandler) -> IntentHandler:
                self.set_fallback(func, label=label, description=description)
                return func

            return decorator

        self.set_fallback(handler, label=label, description=description)
        return handler

    def _resolve_threshold(self, override: float | None) -> float:
        """Return the effective confidence gate for a single call.

        Raises:
            JevConfigurationError: If the override is outside ``0.0``-``1.0``.
        """
        threshold = self.confidence_threshold if override is None else float(override)
        if not 0.0 <= threshold <= 1.0:
            raise JevConfigurationError(
                f"confidence_threshold must be between 0.0 and 1.0, got {threshold}"
            )
        return threshold

    def _resolve_specs(
        self,
        options: Sequence[Any] | None,
    ) -> tuple[OptionSpec, ...]:
        """Normalise the options offered for one call.

        Raises:
            JevConfigurationError: If nothing can be offered to the engine.
        """
        if options is None:
            specs = self.specs
            if not specs:
                raise JevConfigurationError(
                    "Register at least one intent (agent, tool or router) before routing"
                )
            return specs
        candidates = tuple(
            option.to_spec() if isinstance(option, IntentOption) else option for option in options
        )
        specs = coerce_options(candidates)
        if not specs:
            raise JevConfigurationError("At least one option must be offered to the Jev engine")
        return specs

    async def _decide_terminal(self, call: IntentCall) -> JevDecision:
        """The innermost step of the pipeline: ask Jev, then apply the gate."""
        threshold = float(call.annotations.get("threshold", self.confidence_threshold))
        response: JevResponse = await self.client.decide(
            call.text,
            call.options,
            context=call.context,
            system_prompt=self.system_prompt,
        )
        decision = JevDecision.evaluate(
            text=call.text,
            response=response,
            threshold=threshold,
            fallback_label=self._fallback_label,
        )
        if decision.routed and decision.winner not in self._options:
            # Hallucinated labels must never reach a handler: degrade the
            # decision to the fallback path instead of routing blindly.
            logger.warning(
                "The Jev engine returned unregistered intent %r for input %r; degrading to %s",
                decision.winner,
                call.text[:80],
                "fallback" if self._fallback_label else "abstention",
            )
            outcome = (
                DecisionOutcome.FALLBACK if self._fallback_label else DecisionOutcome.ABSTAINED
            )
            return decision.model_copy(
                update={
                    "outcome": outcome,
                    "accepted": False,
                    "reason": f"the engine returned unregistered intent {decision.winner!r}",
                    "fallback_label": (
                        self._fallback_label if outcome is DecisionOutcome.FALLBACK else None
                    ),
                }
            )
        return decision

    async def decide(
        self,
        text: str,
        *,
        context: Mapping[str, Any] | None = None,
        threshold: float | None = None,
        options: Sequence[Any] | None = None,
    ) -> JevDecision:
        """Ask the engine for a gated decision without running a handler.

        Args:
            text: The natural-language input to route.
            context: Optional state forwarded to the engine (agent state, user
                profile, conversation summary, ...).
            threshold: Per-call override of the confidence gate.
            options: A subset of options to offer; every registered option is
                offered by default.

        Returns:
            The gated :class:`~jev_route.core.JevDecision`.

        Raises:
            JevConfigurationError: If no option can be offered.
            JevAPIError: If the engine cannot be reached.
        """
        specs = self._resolve_specs(options)
        call = IntentCall(
            text=text,
            options=specs,
            context=dict(context or {}),
            annotations={"threshold": self._resolve_threshold(threshold)},
        )
        return await self.middleware.run(self._decide_terminal, call)

    async def route(
        self,
        text: str,
        *,
        context: Mapping[str, Any] | None = None,
        threshold: float | None = None,
        options: Sequence[Any] | None = None,
        raise_on_fallback: bool = False,
    ) -> IntentResult:
        """Decide *text* and run the matching handler.

        Args:
            text: The natural-language input to route.
            context: Optional state forwarded to the engine and to the handler.
            threshold: Per-call override of the confidence gate.
            options: A subset of options to offer; every registered option by default.
            raise_on_fallback: Raise instead of returning a fallback or abstained
                result, for callers that must not continue silently.

        Returns:
            An :class:`IntentResult` carrying the decision and the handler value.

        Raises:
            ConfidenceTooLowError: When *raise_on_fallback* is set and the gate
                did not accept a winner.
            IntentNotRegisteredError: If the selected option is unknown.
            IntentHandlerError: If the handler fails.
        """
        decision = await self.decide(text, context=context, threshold=threshold, options=options)
        if raise_on_fallback and not decision.routed:
            raise ConfidenceTooLowError(
                decision.winner or self._fallback_label,
                decision.score,
                decision.threshold,
            )
        return await self.execute(decision, text=text, context=context)

    async def execute(
        self,
        decision: JevDecision,
        *,
        text: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> IntentResult:
        """Run the handler selected by *decision*.

        Args:
            decision: A decision produced by :meth:`decide`.
            text: Overrides the input stored on the decision.
            context: Optional state handed to the handler.

        Returns:
            An :class:`IntentResult`. ``handled`` stays ``False`` when the
            decision abstained or when the target is a decision-only option
            without a handler.

        Raises:
            IntentNotRegisteredError: If the target is not registered.
            IntentHandlerError: If the handler raises.
        """
        result = IntentResult(decision=decision, label=decision.target)
        target = decision.target
        if target is None:
            return result

        option = self._options.get(target)
        if option is None:
            if target == self._fallback_label and self._fallback_handler is not None:
                option = IntentOption(label=target, handler=self._fallback_handler)
            else:
                raise IntentNotRegisteredError(target, self.labels)
        if option.handler is None:
            # A decision-only option: the caller inspects the decision itself.
            return result

        request = IntentRequest(
            input=text if text is not None else decision.input,
            decision=decision,
            option=option,
            context=dict(context or {}),
        )
        result.value = await self._invoke(option, request)
        result.handled = True
        return result

    @staticmethod
    async def _invoke(option: IntentOption, request: IntentRequest) -> Any:
        """Invoke an option handler, awaiting it when it is a coroutine function.

        Raises:
            IntentHandlerError: If the handler raises anything but a
                :class:`~jev_route.core.JevRouteError`.
        """
        handler = option.handler
        if handler is None:  # pragma: no cover - guarded by the caller
            raise IntentNotRegisteredError(option.label, ())
        try:
            value = handler(request)
            if inspect.isawaitable(value):
                value = await value
        except Exception as exc:
            if isinstance(exc, JevRouteError):
                raise
            raise IntentHandlerError(option.label, exc) from exc
        return value

    def include_router(
        self,
        router: IntentRouter,
        *,
        label: str,
        description: str = "",
        examples: Sequence[str] | None = None,
    ) -> IntentOption:
        """Mount *router* as a single option of this router.

        Hierarchical routing: the outer engine picks the domain ("support" or
        "billing") and the inner engine picks the concrete tool. The nested
        router keeps its own options, confidence gate and middleware.

        Args:
            router: The router to delegate to.
            label: The label the outer engine returns to reach it.
            description: Description offered to the outer engine.
            examples: Example inputs offered to the outer engine.

        Returns:
            The option that now publishes *router*.

        Raises:
            JevConfigurationError: If *label* is invalid or already registered.
        """

        async def _delegate(request: IntentRequest) -> IntentResult:
            return await router.route(request.input, context=request.context)

        return self._add(
            label,
            _delegate,
            description=description
            or f"delegate to the {router.name or label!r} intent router",
            examples=examples,
            metadata={"kind": "router", "nested": router.name or label},
        )