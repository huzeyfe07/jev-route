"""Core primitives of :mod:`jev_route` -- the AI agent intent & tool router.

JevRoute is **not** a web URL router. It sits in front of AI agents, tools and
model calls: it hands an incoming natural-language input (plus optional context
and state) to the *Jev* decision engine and asks which registered option -- an
intent, an agent or a tool -- should handle it.

This module owns the domain model of that idea:

* Wire level data structures returned by the engine: :class:`Score`,
  :class:`Choice`, :class:`Noul`, :class:`JevResponse`.
* :class:`JevDecision` -- the router's interpretation of a response, including
  the confidence gate that turns a weak winner into a fallback or an abstention.
* :class:`OptionSpec` -- the description of a routing option offered to Jev.
* :class:`JevSettings` and :class:`JevClient` -- configuration plus the
  asynchronous HTTP client that talks to the Jev engine through an
  OpenRouter-compatible ``/chat/completions`` endpoint using
  :class:`httpx.AsyncClient`.
* The exception hierarchy raised across the package.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, TypeGuard

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_COMPLETIONS_PATH",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_MODEL",
    "DEFAULT_SYSTEM_PROMPT",
    "Choice",
    "ConfidenceTooLowError",
    "DecisionOutcome",
    "IntentHandlerError",
    "IntentNotRegisteredError",
    "JevAPIError",
    "JevClient",
    "JevConfigurationError",
    "JevDecision",
    "JevResponse",
    "JevResponseError",
    "JevRouteError",
    "JevSettings",
    "Noul",
    "OptionSpec",
    "Score",
    "coerce_options",
    "extract_json_object",
]

#: Default Jev decision model hosted on OpenRouter.
DEFAULT_MODEL = "typesafe/jev-1.13"
#: Default OpenRouter API root (OpenAI-compatible chat completions protocol).
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
#: Path of the completions endpoint, appended to :data:`DEFAULT_BASE_URL`.
DEFAULT_COMPLETIONS_PATH = "/chat/completions"
#: A winner below this confidence is not trusted and triggers the fallback path.
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
#: HTTP status codes that are worth retrying with an exponential backoff.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

DEFAULT_SYSTEM_PROMPT = """\
You are Jev, an intent routing engine that runs in front of a fleet of AI agents and tools.
You receive one user input and a closed list of candidate options. You must either pick the
single best option or explicitly abstain.

Answer with one JSON object and nothing else:
{
  "winner": "<option label>" or null,
  "confidence": <number between 0 and 1>,
  "reason": "<short justification>",
  "choices": [{"label": "<option label>", "score": <number between 0 and 1>}, ...],
  "noul": {"reason": "<why no option fits>"} or null
}

Rules:
- "winner" must be one of the provided option labels, or null.
- Never invent option labels and never answer with prose.
- List every provided option exactly once in "choices", ordered by score descending.
- Use "noul" (null outcome) when no option can serve the input or you are unsure.
"""


logger = logging.getLogger("jev_route")


class DecisionOutcome(str, Enum):
    """The three possible results of an intent decision."""

    #: The engine picked an option and its confidence passed the gate.
    ROUTED = "routed"
    #: The winner was too weak, so the configured fallback takes over.
    FALLBACK = "fallback"
    #: No usable winner exists (engine abstention or gate without fallback).
    ABSTAINED = "abstained"


class JevRouteError(Exception):
    """Base class for every error raised by :mod:`jev_route`."""


class JevConfigurationError(JevRouteError):
    """Raised when the client or a router is misconfigured."""


class JevAPIError(JevRouteError):
    """Raised when the Jev engine cannot be reached or answers with an error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        url: str | None = None,
        payload: Any = None,
    ) -> None:
        self.status_code = status_code
        self.url = url
        self.payload = payload
        location = f" [{status_code}]" if status_code is not None else ""
        super().__init__(f"{message}{location}")


class JevResponseError(JevRouteError):
    """Raised when the engine answer cannot be parsed into a decision."""


class IntentNotRegisteredError(JevRouteError):
    """Raised when the engine names an intent that the router does not know."""

    def __init__(self, label: str, registered: Sequence[str]) -> None:
        self.label = label
        self.registered = tuple(registered)
        known = ", ".join(self.registered) if self.registered else "<none>"
        super().__init__(f"Intent {label!r} is not registered. Registered intents: {known}")


class IntentHandlerError(JevRouteError):
    """Raised when an intent handler fails while serving an input."""

    def __init__(self, label: str, original: BaseException) -> None:
        self.label = label
        self.original = original
        detail = f"{type(original).__name__}: {original}"
        super().__init__(f"Handler for intent {label!r} raised {detail}")


class ConfidenceTooLowError(JevRouteError):
    """Raised when a decision must be accepted but its score misses the gate."""

    def __init__(self, label: str, score: float, threshold: float) -> None:
        self.label = label
        self.score = score
        self.threshold = threshold
        super().__init__(
            f"Confidence {score:.3f} for intent {label!r} is below the threshold {threshold:.3f}"
        )


class Score(BaseModel):
    """A confidence value reported by the Jev engine, normalised to ``0.0``-``1.0``.

    Engines are not always well behaved: some answer with ``42`` or ``-1`` when
    asked for a probability. :meth:`Score.clamp` therefore normalises anything
    numeric into the documented range instead of rejecting the whole response.
    """

    model_config = ConfigDict(frozen=True)

    value: float = Field(ge=0.0, le=1.0)
    label: str | None = None

    @classmethod
    def clamp(cls, value: Any, *, label: str | None = None) -> Score:
        """Build a score from a possibly out-of-range engine value.

        Args:
            value: A number, or a string containing a number.
            label: The option label the score belongs to, when known.

        Raises:
            JevResponseError: If *value* is not numeric at all.
        """
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise JevResponseError(f"Invalid score value {value!r} for label {label!r}") from exc
        return cls(value=min(1.0, max(0.0, numeric)), label=label)

    @property
    def percentage(self) -> float:
        """The score expressed in percent, rounded to two decimals."""
        return round(self.value * 100, 2)

    def meets(self, threshold: float) -> bool:
        """Return ``True`` when this score reaches *threshold* (inclusive)."""
        return self.value >= threshold

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.value:.3f}"


class Choice(BaseModel):
    """One candidate option considered by the engine, together with its score."""

    model_config = ConfigDict(frozen=True)

    label: str
    score: Score
    reason: str | None = None
    index: int | None = None

    @classmethod
    def of(
        cls,
        label: str,
        score: Any,
        *,
        reason: str | None = None,
        index: int | None = None,
    ) -> Choice:
        """Build a :class:`Choice`, clamping the raw engine *score*."""
        return cls(label=label, score=Score.clamp(score, label=label), reason=reason, index=index)

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.label} ({self.score.value:.3f})"


class Noul(BaseModel):
    """The engine's explicit abstention, mirroring the ``noul`` response field.

    A *noul* (null outcome) means "no option fits this input". It is a first
    class answer, not a failure: it lets a router escalate to a human or a
    general-purpose agent instead of guessing.
    """

    model_config = ConfigDict(frozen=True)

    reason: str | None = None
    fallback_label: str | None = None
    raw: Mapping[str, Any] = Field(default_factory=dict)


class OptionSpec(BaseModel):
    """A routing option offered to the Jev engine.

    An option is anything the engine may choose: an *intent* (a specialised
    agent), a *tool* (a deterministic function) or a decision-only branch.

    Attributes:
        label: The identifier used by the engine in its ``winner`` field.
        description: One line explaining when the option applies.
        examples: Example inputs that should route to this option.
        metadata: Free-form extras used while building the engine prompt.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    description: str = ""
    examples: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("label")
    @classmethod
    def _validate_label(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("An option label must not be empty")
        if any(character.isspace() for character in cleaned):
            raise ValueError(f"An option label must not contain whitespace: {value!r}")
        return cleaned

    @field_validator("examples", mode="before")
    @classmethod
    def _coerce_examples(cls, value: Any) -> Any:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        return tuple(value)

    def prompt_line(self) -> str:
        """Render the option as a single bullet point for the engine prompt."""
        line = f"- {self.label}"
        if self.description:
            line += f": {self.description}"
        if self.examples:
            examples = "; ".join(str(example) for example in self.examples)
            line += f" (examples: {examples})"
        return line


def coerce_options(options: Sequence[Any]) -> tuple[OptionSpec, ...]:
    """Normalise a mixed sequence of options into :class:`OptionSpec` objects.

    Accepted inputs are plain strings, mappings, :class:`OptionSpec` instances
    and any object exposing ``label`` / ``description`` / ``examples``
    attributes (such as :class:`jev_route.router.IntentOption`).

    Raises:
        JevConfigurationError: If an option cannot be interpreted.
    """
    specs: list[OptionSpec] = []
    for option in options:
        if isinstance(option, OptionSpec):
            specs.append(option)
        elif isinstance(option, str):
            specs.append(OptionSpec(label=option))
        elif isinstance(option, Mapping):
            specs.append(OptionSpec.model_validate(dict(option)))
        elif hasattr(option, "label"):
            specs.append(
                OptionSpec(
                    label=str(option.label),
                    description=str(getattr(option, "description", "") or ""),
                    examples=tuple(getattr(option, "examples", ()) or ()),
                    metadata=dict(getattr(option, "metadata", {}) or {}),
                )
            )
        else:
            raise JevConfigurationError(
                f"Cannot use {option!r} as a routing option; expected a label, a mapping "
                "or an object with a 'label' attribute"
            )
    return tuple(specs)


#: Matches a markdown fenced code block, e.g. ```json { ... } ```.
_FENCED_JSON_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _is_sequence(value: Any) -> TypeGuard[Sequence[Any]]:
    """Return ``True`` for real sequences, narrowing the value to one.

    Strings and byte buffers are excluded, so a caller that passes this guard
    may safely treat the value as a container of items.
    """
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def extract_json_object(content: str) -> Mapping[str, Any]:
    """Extract the first JSON object from a possibly chatty engine answer.

    Handles plain JSON, markdown fenced JSON and answers that wrap the decision
    object in prose.

    Args:
        content: The raw ``message.content`` returned by the engine.

    Returns:
        The decoded JSON object.

    Raises:
        JevResponseError: If no JSON object can be found.
    """
    if not content or not content.strip():
        raise JevResponseError("The Jev engine returned an empty answer")

    candidates: list[str] = [content]
    fenced = _FENCED_JSON_PATTERN.search(content)
    if fenced:
        candidates.insert(0, fenced.group(1))

    decoder = json.JSONDecoder()
    for candidate in candidates:
        stripped = candidate.strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
            return parsed
        # Fall back to scanning for the first decodable object in the text.
        for start, character in enumerate(stripped):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(stripped[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, Mapping):
                return value

    snippet = content.strip()[:200]
    raise JevResponseError(f"No JSON decision object in the engine answer: {snippet!r}")


class JevResponse(BaseModel):
    """The parsed answer of the Jev engine for a single decision request.

    Attributes:
        winner: The option label chosen by the engine (``None`` on abstention).
        choices: Every option the engine scored, ordered by descending score.
        noul: The explicit null outcome, when the engine abstained.
        model: The model identifier that produced the answer.
        request_id: The upstream completion id, useful for tracing and billing.
        latency_ms: Round-trip time of the completion call, in milliseconds.
        usage: Token usage and cost as reported by the gateway.
        raw: The untouched gateway payload.
    """

    winner: str | None = None
    choices: tuple[Choice, ...] = ()
    noul: Noul | None = None
    model: str = DEFAULT_MODEL
    request_id: str | None = None
    latency_ms: float = 0.0
    usage: Mapping[str, Any] = Field(default_factory=dict)
    raw: Any = None

    @property
    def is_abstention(self) -> bool:
        """Return ``True`` when the engine refused to pick a winner."""
        return self.winner is None

    @property
    def winner_score(self) -> Score | None:
        """The score attached to :attr:`winner`, when it was reported."""
        return None if self.winner is None else self.score_for(self.winner)

    def score_for(self, label: str) -> Score | None:
        """Return the score recorded for *label*, or ``None`` when unknown."""
        for choice in self.choices:
            if choice.label == label:
                return choice.score
        return None

    @property
    def ranking(self) -> tuple[Choice, ...]:
        """The recorded choices ordered by descending score."""
        return tuple(sorted(self.choices, key=lambda choice: choice.score.value, reverse=True))

    def describe(self) -> str:
        """Return a one-line human readable summary of the engine answer."""
        if self.winner is None:
            reason = self.noul.reason if self.noul is not None else None
            return f"abstained ({reason})" if reason else "abstained"
        score = self.winner_score
        return f"{self.winner} @ {score.value:.3f}" if score is not None else str(self.winner)


class JevDecision(BaseModel):
    """The router's interpretation of a :class:`JevResponse`.

    This is where the **confidence gate** lives: a winner whose score does not
    reach :attr:`threshold` is not trusted. When a fallback is configured the
    decision becomes :attr:`DecisionOutcome.FALLBACK`, otherwise the router
    abstains instead of routing a weak guess to an agent.
    """

    input: str
    outcome: DecisionOutcome
    winner: str | None = None
    score: float = 0.0
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    accepted: bool = False
    reason: str | None = None
    fallback_label: str | None = None
    choices: tuple[Choice, ...] = ()
    response: JevResponse | None = None

    @classmethod
    def evaluate(
        cls,
        *,
        text: str,
        response: JevResponse,
        threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        fallback_label: str | None = None,
    ) -> JevDecision:
        """Apply the confidence gate to *response* and build a decision.

        Args:
            text: The input that was routed.
            response: The answer of the Jev engine.
            threshold: The minimum confidence required to accept a winner.
            fallback_label: The label used when the gate rejects the winner.
        """
        winner_score = response.winner_score
        value = winner_score.value if winner_score is not None else 0.0
        noul_reason = response.noul.reason if response.noul is not None else None

        if response.winner is None:
            # The engine itself refused to choose: escalate or abstain.
            outcome = DecisionOutcome.FALLBACK if fallback_label else DecisionOutcome.ABSTAINED
            accepted = False
            reason = noul_reason or "the engine abstained"
        elif value >= threshold:
            outcome = DecisionOutcome.ROUTED
            accepted = True
            reason = noul_reason or "the winner passed the confidence gate"
        else:
            # A weak winner is treated as no winner at all.
            outcome = DecisionOutcome.FALLBACK if fallback_label else DecisionOutcome.ABSTAINED
            accepted = False
            reason = f"confidence {value:.3f} is below the threshold {threshold:.3f}"

        return cls(
            input=text,
            outcome=outcome,
            winner=response.winner,
            score=value,
            threshold=threshold,
            accepted=accepted,
            reason=reason,
            fallback_label=fallback_label if outcome is DecisionOutcome.FALLBACK else None,
            choices=response.ranking,
            response=response,
        )

    @property
    def target(self) -> str | None:
        """The label that should handle the input, or ``None`` on abstention."""
        if self.outcome is DecisionOutcome.ROUTED:
            return self.winner
        if self.outcome is DecisionOutcome.FALLBACK:
            return self.fallback_label
        return None

    @property
    def routed(self) -> bool:
        """Return ``True`` when the winner passed the confidence gate."""
        return self.outcome is DecisionOutcome.ROUTED

    @property
    def fallback(self) -> bool:
        """Return ``True`` when the fallback path was triggered."""
        return self.outcome is DecisionOutcome.FALLBACK

    @property
    def abstained(self) -> bool:
        """Return ``True`` when no handler can serve the input."""
        return self.outcome is DecisionOutcome.ABSTAINED

    def describe(self) -> str:
        """Return a one-line human readable summary of the decision."""
        return (
            f"{self.outcome.value}: target={self.target!r} score={self.score:.3f} "
            f"(threshold={self.threshold:.3f}, reason={self.reason})"
        )


class JevSettings(BaseModel):
    """Immutable configuration for :class:`JevClient`.

    Attributes:
        api_key: Bearer token for the gateway (an OpenRouter or Jev API key).
        base_url: API root including the version prefix.
        endpoint: Completions path appended to :attr:`base_url`.
        model: The Jev decision model, ``typesafe/jev-1.13`` by default.
        timeout: Per-request timeout in seconds.
        max_retries: Extra attempts for timeouts, rate limits and ``5xx`` answers.
        retry_backoff: Base delay of the exponential retry backoff, in seconds.
        confidence_threshold: Default gate used by
            :class:`jev_route.router.IntentRouter`.
        temperature: Sampling temperature; ``0.0`` keeps routing deterministic.
        max_tokens: Optional cap on the answer length.
        json_mode: Ask the gateway for a JSON response format. Disabled by
            default because not every provider supports it; the prompt already
            demands JSON and the parser tolerates chatty answers.
        site_url: Optional ``HTTP-Referer`` attribution header for OpenRouter.
        app_title: Optional ``X-Title`` attribution header for OpenRouter.
        extra_headers: Additional headers merged into every request.
    """

    model_config = ConfigDict(frozen=True)

    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    endpoint: str = DEFAULT_COMPLETIONS_PATH
    model: str = DEFAULT_MODEL
    timeout: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    retry_backoff: float = Field(default=0.25, ge=0)
    confidence_threshold: float = Field(default=DEFAULT_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    json_mode: bool = False
    site_url: str | None = None
    app_title: str | None = "JevRoute"
    extra_headers: Mapping[str, str] = Field(default_factory=dict)

    @field_validator("base_url")
    @classmethod
    def _normalise_base_url(cls, value: str) -> str:
        cleaned = value.strip().rstrip("/")
        if not cleaned:
            raise ValueError("base_url must not be empty")
        return cleaned

    @field_validator("endpoint")
    @classmethod
    def _normalise_endpoint(cls, value: str) -> str:
        cleaned = value.strip().strip("/")
        return f"/{cleaned}" if cleaned else DEFAULT_COMPLETIONS_PATH

    @property
    def completions_url(self) -> str:
        """The absolute URL of the completions endpoint."""
        return f"{self.base_url}{self.endpoint}"

    @property
    def has_credentials(self) -> bool:
        """Return ``True`` when an API key is configured."""
        return bool(self.api_key.strip())

    @property
    def is_openrouter(self) -> bool:
        """Return ``True`` when the endpoint points at OpenRouter."""
        return "openrouter.ai" in self.base_url

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        **overrides: Any,
    ) -> JevSettings:
        """Build settings from environment variables plus explicit *overrides*.

        Recognised variables: ``JEV_API_KEY`` (falls back to
        ``OPENROUTER_API_KEY``), ``JEV_BASE_URL``, ``JEV_ENDPOINT``,
        ``JEV_MODEL``, ``JEV_TIMEOUT``, ``JEV_MAX_RETRIES``,
        ``JEV_CONFIDENCE_THRESHOLD``, ``JEV_TEMPERATURE``, ``JEV_JSON_MODE``,
        ``JEV_SITE_URL`` and ``JEV_APP_TITLE``.

        Args:
            env: The mapping to read from; defaults to :data:`os.environ`.
            **overrides: Values that win over the environment. ``None`` values
                are ignored so callers can forward optional arguments directly.
        """
        source = os.environ if env is None else env
        values: dict[str, Any] = {
            "api_key": source.get("JEV_API_KEY") or source.get("OPENROUTER_API_KEY") or "",
            "base_url": source.get("JEV_BASE_URL") or DEFAULT_BASE_URL,
            "endpoint": source.get("JEV_ENDPOINT") or DEFAULT_COMPLETIONS_PATH,
            "model": source.get("JEV_MODEL") or DEFAULT_MODEL,
            "site_url": source.get("JEV_SITE_URL") or None,
            "app_title": source.get("JEV_APP_TITLE") or None,
        }

        numeric_variables = (
            ("JEV_TIMEOUT", "timeout", float),
            ("JEV_MAX_RETRIES", "max_retries", int),
            ("JEV_CONFIDENCE_THRESHOLD", "confidence_threshold", float),
            ("JEV_TEMPERATURE", "temperature", float),
        )
        for variable, field_name, caster in numeric_variables:
            raw_value = source.get(variable)
            if not raw_value:
                continue
            try:
                values[field_name] = caster(raw_value)
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid value for %s: %r", variable, raw_value)

        json_mode = source.get("JEV_JSON_MODE")
        if json_mode is not None:
            values["json_mode"] = json_mode.strip().lower() in {"1", "true", "yes", "on"}

        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)


class JevClient:
    """Asynchronous client for the Jev intent decision engine.

    The engine is reached through an OpenAI-compatible chat completions call,
    which is what OpenRouter exposes, so the defaults target
    ``https://openrouter.ai/api/v1`` with the ``typesafe/jev-1.13`` model. The
    transport is :class:`httpx.AsyncClient`, which keeps the client easy to test:
    pass ``transport=httpx.MockTransport(handler)`` to replay canned answers
    without a network round trip or an API key.

    Example:
        >>> async with JevClient(api_key="sk-...") as client:
        ...     response = await client.decide(
        ...         "How is the weather in Berlin?",
        ...         ["weather_agent", "search_agent"],
        ...     )
        ...     print(response.winner)  # 'weather_agent'
    """

    def __init__(
        self,
        settings: JevSettings | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a client.

        Args:
            settings: Full configuration; environment variables are read when
                this is omitted.
            api_key: Overrides ``settings.api_key``.
            base_url: Overrides ``settings.base_url``.
            model: Overrides ``settings.model``.
            timeout: Overrides ``settings.timeout``.
            transport: Optional :mod:`httpx` transport. Injecting one (for
                example an :class:`httpx.MockTransport`) waives the API key
                requirement, because it exists to run the engine offline.
            headers: Extra headers merged on top of the generated ones.
            client: A ready-made :class:`httpx.AsyncClient` to reuse. It stays
                owned by the caller, so JevRoute never closes it.
        """
        base = settings if settings is not None else JevSettings.from_env()
        overrides: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "timeout": timeout,
        }
        active_overrides = {key: value for key, value in overrides.items() if value is not None}
        self._settings = base.model_copy(update=active_overrides) if active_overrides else base
        self._client = client
        self._transport = transport
        self._extra_headers = dict(headers or {})
        self._owns_client = client is None
        self._logger = logging.getLogger("jev_route.client")

    @property
    def settings(self) -> JevSettings:
        """The immutable configuration in use."""
        return self._settings

    @property
    def model(self) -> str:
        """The Jev model identifier used for decisions."""
        return self._settings.model

    @property
    def is_closed(self) -> bool:
        """Return ``True`` when no live HTTP client is available."""
        return self._client is None or self._client.is_closed

    async def __aenter__(self) -> JevClient:
        """Enter the client's async context and open the HTTP connection pool."""
        self._ensure_client()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        """Close the owned HTTP client when leaving the async context."""
        await self.aclose()

    def _build_headers(self) -> dict[str, str]:
        """Compose the headers sent with every completion request."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._settings.has_credentials:
            headers["Authorization"] = f"Bearer {self._settings.api_key.strip()}"
        if self._settings.site_url:
            headers["HTTP-Referer"] = self._settings.site_url
        if self._settings.app_title:
            headers["X-Title"] = self._settings.app_title
        headers.update(self._settings.extra_headers)
        headers.update(self._extra_headers)
        return headers

    def _ensure_client(self) -> httpx.AsyncClient:
        """Return a live HTTP client, creating (and owning) one when needed."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._settings.timeout,
                transport=self._transport,
                headers=self._build_headers(),
            )
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP client when this instance owns it."""
        if self._client is not None and self._owns_client and not self._client.is_closed:
            await self._client.aclose()
        if self._owns_client:
            self._client = None

    def _check_credentials(self) -> None:
        """Raise when a remote call would go out without an API key.

        Injected transports and borrowed clients are exempt: they exist to run
        the engine offline. Non-OpenRouter endpoints are exempt as well, since a
        self-hosted Jev gateway may not require a bearer token.

        Raises:
            JevConfigurationError: If the call would certainly fail with ``401``.
        """
        if self._settings.has_credentials or self._transport is not None:
            return
        if self._client is not None and not self._owns_client:
            return
        if not self._settings.is_openrouter:
            return
        raise JevConfigurationError(
            "No Jev API key configured. Set the JEV_API_KEY (or OPENROUTER_API_KEY) "
            "environment variable, pass api_key=..., or inject a transport for offline use."
        )

    @staticmethod
    def build_user_prompt(
        text: str,
        options: Sequence[OptionSpec],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        """Render the decision task as the user message of the completion.

        Args:
            text: The natural-language input that must be routed.
            options: The options the engine may choose from.
            context: Optional state handed to the engine (conversation summary,
                user profile, feature flags, ...).
        """
        lines = ["Input:", '"""', text.strip(), '"""', "", "Options:"]
        lines.extend(option.prompt_line() for option in options)
        if context:
            lines.extend(
                [
                    "",
                    "Additional context (JSON):",
                    json.dumps(dict(context), ensure_ascii=False, default=str, sort_keys=True),
                ]
            )
        lines.extend(["", "Respond with the JSON object only."])
        return "\n".join(lines)

    def build_messages(
        self,
        text: str,
        options: Sequence[OptionSpec],
        *,
        context: Mapping[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Build the ``messages`` array for the completions request."""
        return [
            {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": self.build_user_prompt(text, options, context=context)},
        ]

    def build_payload(
        self,
        text: str,
        options: Sequence[OptionSpec],
        *,
        context: Mapping[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Build the JSON body posted to the completions endpoint."""
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "messages": self.build_messages(
                text, options, context=context, system_prompt=system_prompt
            ),
            "temperature": self._settings.temperature,
            "stream": False,
        }
        if self._settings.max_tokens is not None:
            payload["max_tokens"] = self._settings.max_tokens
        if self._settings.json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    async def _backoff(self, attempt: int) -> None:
        """Sleep before retrying attempt number *attempt* (zero based)."""
        delay = self._settings.retry_backoff * (2**attempt)
        if delay > 0:
            await asyncio.sleep(delay)

    @staticmethod
    def _decode_json(response: httpx.Response) -> Any:
        """Decode a response body, turning decode failures into a JevRoute error."""
        try:
            return response.json()
        except ValueError as exc:
            snippet = response.text.strip()[:200]
            raise JevResponseError(f"The Jev gateway returned invalid JSON: {snippet!r}") from exc

    @staticmethod
    def _maybe_json(response: httpx.Response) -> Any:
        """Decode a response body, returning ``None`` instead of raising."""
        try:
            return response.json()
        except ValueError:
            return None

    @classmethod
    def _describe_api_error(cls, response: httpx.Response) -> str:
        """Extract a readable message from an error response."""
        body = cls._maybe_json(response)
        if isinstance(body, Mapping):
            error = body.get("error")
            if isinstance(error, Mapping) and error.get("message"):
                return f"The Jev engine rejected the request: {error['message']}"
            if isinstance(error, str):
                return f"The Jev engine rejected the request: {error}"
            if body.get("message"):
                return f"The Jev engine rejected the request: {body['message']}"
        snippet = response.text.strip()[:200] or response.reason_phrase
        return f"The Jev engine rejected the request: {snippet}"

    async def _post(self, payload: Mapping[str, Any]) -> tuple[Any, float]:
        """POST *payload* to the completions endpoint with retries.

        Timeouts, transport errors, ``429`` and ``5xx`` answers are retried with
        an exponential backoff.

        Returns:
            A ``(decoded_body, latency_ms)`` pair for the successful attempt.

        Raises:
            JevAPIError: When the request keeps failing.
            JevResponseError: When the gateway answers with invalid JSON.
        """
        client = self._ensure_client()
        url = self._settings.completions_url
        attempts = self._settings.max_retries + 1
        last_error: BaseException | None = None

        for attempt in range(attempts):
            started_at = time.perf_counter()
            try:
                response = await client.post(url, json=dict(payload))
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    self._logger.debug("Retrying %s after transport error: %s", url, exc)
                    await self._backoff(attempt)
                    continue
                raise JevAPIError(
                    f"Unable to reach the Jev engine at {url}: {exc}", url=url
                ) from exc

            latency_ms = (time.perf_counter() - started_at) * 1000
            if response.status_code in RETRYABLE_STATUS_CODES and attempt + 1 < attempts:
                self._logger.debug("Retrying %s after HTTP %s", url, response.status_code)
                await self._backoff(attempt)
                continue
            if response.status_code >= 400:
                raise JevAPIError(
                    self._describe_api_error(response),
                    status_code=response.status_code,
                    url=url,
                    payload=self._maybe_json(response),
                )
            return self._decode_json(response), latency_ms

        raise JevAPIError(
            f"Exhausted {attempts} attempt(s) against {url}: {last_error}",
            url=url,
        )

    @staticmethod
    def _extract_content(payload: Mapping[str, Any]) -> str:
        """Pull the assistant text out of a chat completions payload.

        Raises:
            JevResponseError: If the payload carries no usable message.
        """
        choices = payload.get("choices")
        if not _is_sequence(choices) or not choices:
            raise JevResponseError("The Jev gateway returned no completion choices")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise JevResponseError(f"Unexpected completion choice: {first!r}")

        message = first.get("message")
        if not isinstance(message, Mapping):
            # Tolerate the legacy ``text`` completion shape.
            legacy_text = first.get("text")
            if isinstance(legacy_text, str):
                return legacy_text
            raise JevResponseError(f"The completion choice carries no message: {first!r}")

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if _is_sequence(content):
            parts = [part.get("text", "") for part in content if isinstance(part, Mapping)]
            joined = "".join(str(part) for part in parts)
            if joined.strip():
                return joined
        raise JevResponseError("The Jev engine answered with an empty message")

    def parse_response(
        self,
        payload: Mapping[str, Any],
        *,
        latency_ms: float = 0.0,
        model: str | None = None,
    ) -> JevResponse:
        """Turn a raw completions payload into a :class:`JevResponse`.

        Args:
            payload: The decoded gateway payload.
            latency_ms: Measured round-trip time of the call.
            model: Model override; taken from the payload when omitted.

        Raises:
            JevResponseError: When the payload cannot be interpreted.
        """
        if not isinstance(payload, Mapping):
            raise JevResponseError(f"Unexpected gateway payload: {payload!r}")

        decision_payload = extract_json_object(self._extract_content(payload))
        request_id = payload.get("id")
        reported_model = payload.get("model") or model or self._settings.model
        usage = payload.get("usage")
        try:
            return self._build_response(
                decision_payload,
                request_id=str(request_id) if request_id else None,
                model=str(reported_model),
                latency_ms=float(latency_ms),
                usage=usage if isinstance(usage, Mapping) else {},
                raw=payload,
            )
        except ValidationError as exc:
            raise JevResponseError(f"The Jev engine returned an unusable decision: {exc}") from exc

    @staticmethod
    def _build_choice(item: Any, index: int) -> Choice | None:
        """Normalise one entry of the engine's ``choices`` array."""
        if isinstance(item, Mapping):
            label = (
                item.get("label") or item.get("intent") or item.get("name") or item.get("option")
            )
            if not label:
                return None
            raw_score = item.get("score", item.get("confidence", item.get("probability", 0.0)))
            raw_reason = item.get("reason") or item.get("explanation")
        elif isinstance(item, str) and item.strip():
            label = item.strip()
            raw_score = 0.0
            raw_reason = None
        else:
            return None
        return Choice.of(
            str(label),
            raw_score,
            reason=str(raw_reason) if raw_reason else None,
            index=index,
        )

    @staticmethod
    def _build_noul(decision: Mapping[str, Any]) -> Noul | None:
        """Normalise the engine's ``noul`` (null outcome) field."""
        raw_noul = decision.get("noul", decision.get("null_outcome"))
        if raw_noul is None or raw_noul is False:
            return None
        if isinstance(raw_noul, Mapping):
            raw_reason = raw_noul.get("reason") or raw_noul.get("message")
            raw_fallback = raw_noul.get("fallback_label") or raw_noul.get("fallback")
            return Noul(
                reason=str(raw_reason) if raw_reason else None,
                fallback_label=str(raw_fallback) if raw_fallback else None,
                raw=dict(raw_noul),
            )
        if raw_noul is True:
            return Noul(reason="the engine abstained")
        return Noul(reason=str(raw_noul))

    @classmethod
    def _build_response(
        cls,
        decision: Mapping[str, Any],
        *,
        request_id: str | None,
        model: str,
        latency_ms: float,
        usage: Mapping[str, Any],
        raw: Any,
    ) -> JevResponse:
        """Normalise the engine's decision object into a :class:`JevResponse`.

        The mapping is deliberately tolerant: engines answer with ``label``,
        ``intent``, ``name`` or ``option`` interchangeably, and may omit the
        ``choices`` list when they only report a winner. A missing ``winner`` is
        never guessed from the ranking; it is treated as an abstention so that
        the confidence gate stays conservative.
        """
        choices: list[Choice] = []
        raw_choices = decision.get("choices")
        if _is_sequence(raw_choices):
            for index, item in enumerate(raw_choices):
                choice = cls._build_choice(item, index)
                if choice is not None:
                    choices.append(choice)

        raw_winner = decision.get("winner", decision.get("label", decision.get("intent")))
        winner = str(raw_winner).strip() if raw_winner is not None else None
        winner = winner or None
        raw_confidence = decision.get("confidence", decision.get("score", 0.0))

        if winner is not None and all(choice.label != winner for choice in choices):
            choices.insert(
                0,
                Choice.of(winner, raw_confidence, reason=decision.get("reason"), index=0),
            )

        noul = cls._build_noul(decision)
        if noul is None and winner is None:
            noul = Noul(reason="the engine did not report a winner")

        return JevResponse(
            winner=winner,
            choices=tuple(choices),
            noul=noul,
            model=model,
            request_id=request_id,
            latency_ms=latency_ms,
            usage=dict(usage),
            raw=raw,
        )

    async def decide(
        self,
        text: str,
        options: Sequence[Any],
        *,
        context: Mapping[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> JevResponse:
        """Ask the Jev engine which option best matches *text*.

        Args:
            text: The natural-language input to route.
            options: The candidate options; see :func:`coerce_options` for every
                accepted shape.
            context: Optional state forwarded to the engine (agent state,
                conversation summary, user profile, ...).
            system_prompt: Overrides the built-in routing system prompt.

        Returns:
            The parsed engine answer. Apply
            :meth:`jev_route.core.JevDecision.evaluate` to turn it into a gated
            decision.

        Raises:
            JevConfigurationError: When no option was offered or no API key is set.
            JevAPIError: When the engine cannot be reached or answers with an error.
            JevResponseError: When the answer cannot be parsed.
        """
        specs = coerce_options(options)
        if not specs:
            raise JevConfigurationError(
                "At least one option must be offered to the Jev engine before routing"
            )
        self._check_credentials()
        payload = self.build_payload(text, specs, context=context, system_prompt=system_prompt)
        body, latency_ms = await self._post(payload)
        return self.parse_response(body, latency_ms=latency_ms)