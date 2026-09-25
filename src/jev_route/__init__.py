"""JevRoute -- an AI agent intent & tool router for Python.

JevRoute sits in front of AI agents, tools and model calls. It asks the Jev
decision engine (``typesafe/jev-1.13``, reached through OpenRouter) which
registered option should handle an incoming input, applies a confidence gate and
only then invokes the matching handler.

Example:
    from jev_route import IntentRouter, JevClient

    router = IntentRouter(JevClient(), confidence_threshold=0.6)

    @router.tool("weather_agent", description="current weather and forecasts")
    async def weather_agent(request):
        return {"forecast": "sunny"}

    result = await router.route("How is the weather?")
"""

from __future__ import annotations

from .core import (
    DEFAULT_BASE_URL,
    DEFAULT_COMPLETIONS_PATH,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    Choice,
    ConfidenceTooLowError,
    DecisionOutcome,
    IntentHandlerError,
    IntentNotRegisteredError,
    JevAPIError,
    JevClient,
    JevConfigurationError,
    JevDecision,
    JevResponse,
    JevResponseError,
    JevRouteError,
    JevSettings,
    Noul,
    OptionSpec,
    Score,
    coerce_options,
    extract_json_object,
)
from .middleware import (
    CacheMiddleware,
    IntentCall,
    IntentMiddleware,
    LoggingMiddleware,
    Middleware,
    MiddlewarePipeline,
    NextStep,
)
from .router import IntentOption, IntentRequest, IntentResult, IntentRouter

__version__ = "0.2.0"
__title__ = "jev-route"

__all__ = [
    "__version__",
    "__title__",
    # core: constants
    "DEFAULT_BASE_URL",
    "DEFAULT_COMPLETIONS_PATH",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_MODEL",
    "DEFAULT_SYSTEM_PROMPT",
    # core: data structures
    "Choice",
    "DecisionOutcome",
    "JevDecision",
    "JevResponse",
    "Noul",
    "OptionSpec",
    "Score",
    # core: client and configuration
    "JevClient",
    "JevSettings",
    # core: errors
    "ConfidenceTooLowError",
    "IntentHandlerError",
    "IntentNotRegisteredError",
    "JevAPIError",
    "JevConfigurationError",
    "JevResponseError",
    "JevRouteError",
    # core: helpers
    "coerce_options",
    "extract_json_object",
    # middleware
    "CacheMiddleware",
    "IntentCall",
    "IntentMiddleware",
    "LoggingMiddleware",
    "Middleware",
    "MiddlewarePipeline",
    "NextStep",
    # router
    "IntentOption",
    "IntentRequest",
    "IntentResult",
    "IntentRouter",
]