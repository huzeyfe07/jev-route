<div align="center">

# jev-route

**An AI agent intent & tool router for Python.**
JevRoute sits in front of your agents, tools and model calls: it asks the Jev
decision engine *which* capability should handle an input, applies a confidence
gate, and only then invokes the matching handler.

[![CI](https://github.com/huzeyfe07/jev-route/actions/workflows/ci.yml/badge.svg)](https://github.com/huzeyfe07/jev-route/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-261230)](https://docs.astral.sh/ruff/)
[![Types: mypy strict](https://img.shields.io/badge/types-mypy%20strict-blue)](https://mypy-lang.org/)

</div>

---

## Why jev-route?

The Jev ecosystem is dominated by **harness adapters**: CLI tools, IDE plugins,
local proxies and agent skills that add a decision step to a runner you did not
write. JevRoute is deliberately the other shape — a plain Python dependency that
lives **inside your own agent**. No CLI, no daemon, no IDE, no manifest: you
register your own callables and the library decides which one runs.

| | Harness adapter (CLI / IDE / proxy) | jev-route (embedding library) |
| --- | --- | --- |
| Where it runs | in the harness, in its process | in your process, in front of your handlers |
| Unit of extension | plugin, skill, hook or manifest | `@router.intent` / `@router.tool` on a Python callable |
| Who owns the loop | the harness | your agent; the router is one `await` inside it |
| When a decision is weak | depends on the harness honouring the model | code decides: gate → fallback handler, or raise |

That last row is the substantive difference, and JevRoute enforces it in the
routing layer instead of requesting it in a prompt:

```python
@router.tool("issue_refund_tool", description="issue a refund for an order")
async def issue_refund_tool(request: IntentRequest) -> dict[str, Any]:
    return await refunds.issue(request.context["order_id"])
```

1. `JevClient.decide()` returns a typed `JevResponse`: `winner`, `confidence`,
   per-option `choices`, and the engine's own `noul` abstention.
2. `JevDecision.evaluate()` compares the winner's score against your threshold.
   Below it, the winner is discarded and the decision becomes a fallback — or a
   plain abstention when no fallback is configured. The handler is never
   invoked.
3. The winning label is then resolved against the router's **own registry**. A
   label the engine invented is degraded to the same fallback path and logged
   with a warning, rather than executed.
4. `IntentNotRegisteredError` is raised when you hand `execute()` a decision
   whose target is not registered — for example one built by hand or cached
   against an older registry.

A weak or invented decision therefore cannot reach a privileged tool: it either
lands on your fallback handler or raises. The check is a comparison in
`core.JevDecision.evaluate()`, not an instruction a model is asked to respect —
which is why it holds for low-confidence guesses, hallucinated labels and
abstentions alike.

None of this means harness adapters are unsafe; they solve a different problem,
namely improving a tool you do not control. When you *do* control the code — a
service, a worker, a graph node, a notebook — routing belongs in it. The same
design keeps JevRoute testable: exact thresholds, an injectable
`httpx.MockTransport`, `mypy --strict` types (`py.typed` shipped) and no
framework lock-in.

## Architecture

JevRoute never answers the user itself — it decides **who should**, and hands the
work over. The Jev engine (`typesafe/jev-1.13`) is reached through OpenRouter's
OpenAI-compatible `/chat/completions` endpoint.

```mermaid
flowchart TD
    U(["User input / agent state"]) --> IR["IntentRouter<br/>intent registry + confidence gate"]
    IR --> MW["MiddlewarePipeline<br/>logging, cache, audit, guards"]
    MW --> JC["JevClient<br/>httpx.AsyncClient"]
    JC -->|"POST /chat/completions<br/>input + candidate options"| JEV
    JEV[["Jev engine - typesafe/jev-1.13<br/>via OpenRouter API"]] -->|"JSON: winner, confidence,<br/>choices, noul"| GATE{"score >= threshold ?"}
    GATE -->|"accepted"| H["Matching handler<br/>agent (async) or tool (sync)"]
    GATE -->|"too weak, unregistered or noul"| FB["Fallback handler<br/>human_handoff / escalation"]
    H --> OUT(["IntentResult<br/>decision + handler value"])
    FB --> OUT
```

### Why a decision layer?

Most agent stacks hard-code routing with `if/elif` chains or let the main model
free-style tool selection. JevRoute separates that concern:

| Step | What happens | Where |
| --- | --- | --- |
| 1. Offer | You register the intents, agents and tools you own | `IntentRouter.register()` / `.intent()` / `.tool()` |
| 2. Ask | The input plus the option list goes to the Jev engine | `JevClient.decide()` |
| 3. Gate | A winner below your threshold is **not** trusted | `JevDecision.evaluate()` |
| 4. Act | The winner (or the fallback) is executed | `IntentRouter.route()` |

Because the gate lives in the routing layer, a weak or hallucinated answer can
never silently reach a privileged tool: it escalates to your fallback instead.

## Features

- Async `JevClient` on `httpx.AsyncClient` (OpenRouter-compatible, retries included).
- Typed Jev data structures: `Score`, `Choice`, `Noul`, `JevResponse`, `JevDecision`.
- `IntentRouter` with **confidence gate** and fallback/escalation handling.
- Async **middleware pipeline** shared by intent resolution and agent loops
  (`LoggingMiddleware`, `CacheMiddleware`, plus your own).
- Transport-agnostic decision models: no framework lock-in, testable offline.
- Strict quality gates: `ruff`, `mypy --strict`, `pytest` (all green in CI).

## Installation & Quickstart

**Requirements:** Python 3.10+ and an OpenRouter API key (or any Jev-compatible
gateway exposing the OpenAI chat completions protocol).

### 1. Install

```bash
# from PyPI (once released)
pip install jev-route

# or, editable for development (ruff, mypy, pytest included)
git clone https://github.com/huzeyfe07/jev-route.git
cd jev-route
python -m pip install -e ".[dev]"
```

### 2. Provide credentials

JevRoute reads the key from the environment — never hard-code it.

```powershell
# Windows PowerShell
$env:OPENROUTER_API_KEY = "sk-or-..."
```

```bash
# bash / zsh
export OPENROUTER_API_KEY="sk-or-..."
```

### 3. Route your first prompt

```python
import asyncio

from jev_route import IntentRequest, IntentRouter, JevClient


async def main() -> None:
    # 1. The Jev engine, reached through OpenRouter.
    async with JevClient() as client:
        # 2. Build the router: options + confidence gate + fallback.
        router = IntentRouter(
            client,
            name="support-desk",
            confidence_threshold=0.6,
            fallback_label="human_handoff",
        )

        @router.intent(
            "weather_agent",
            description="current weather and short forecasts for a city",
            examples=("How is the weather?",),
        )
        async def weather_agent(request: IntentRequest) -> dict[str, str]:
            return {
                "agent": "weather_agent",
                "city": request.context.get("city", "Istanbul"),
                "forecast": "partly cloudy, 21 C",
            }

        @router.tool("calculator_tool", description="plain arithmetic")
        def calculator_tool(request: IntentRequest) -> dict[str, int]:
            return {"result": 96}

        @router.fallback(label="human_handoff")
        async def human_handoff(request: IntentRequest) -> dict[str, str]:
            return {"escalated": True, "why": request.decision.reason}

        # 3. Decide *and* execute in a single call.
        result = await router.route("How is the weather?", context={"city": "Berlin"})
        print(result.label)            # weather_agent
        print(result.value)            # {'agent': 'weather_agent', 'city': 'Berlin', ...}
        print(result.decision.score)   # 0.94
        print(result.decision.reason)  # the winner passed the confidence gate


asyncio.run(main())
```

Prefer to decide and act separately? Use `await router.decide(text)` for a
gate-checked `JevDecision`, then `await router.execute(decision)` to run the
selected handler.

### 4. Try it with zero tokens

The bundled example replays canned Jev answers through an `httpx.MockTransport`
whenever no API key is set, so the whole flow runs offline:

```bash
python examples/basic_routing.py
```

It routes four prompts — a confident match, a tool call, a weak winner that trips
the confidence gate, and an out-of-scope request that abstains — then prints an
audit trail collected by the middleware.

### The confidence gate: before and after

The example routes an ambiguous prompt — *"Şirketimizin cirosu ne kadar?"* —
that Jev answers with `search_agent` at **0.42** confidence. Rename that option to
a privileged handler (`issue_refund_tool`, `send_email_tool`, `sql_executor`) and
the gate is the only thing standing between a guess and a side effect:

```python
# BEFORE — gate disabled: whatever the engine returns gets executed.
router = IntentRouter(client, confidence_threshold=0.0, fallback_label="human_handoff")
result = await router.route("Şirketimizin cirosu ne kadar?")

# decision.winner   -> "search_agent"
# decision.score    -> 0.42
# decision.accepted -> True         (0.42 >= 0.0)
# result.handled    -> True         the handler ran on a 42%-confidence guess

# AFTER — gate at 0.6: same input, same engine answer.
router = IntentRouter(client, confidence_threshold=0.6, fallback_label="human_handoff")
result = await router.route("Şirketimizin cirosu ne kadar?")

# decision.winner   -> "search_agent"  (kept, so the audit trail shows the guess)
# decision.accepted -> False
# decision.outcome  -> DecisionOutcome.FALLBACK
# decision.reason   -> "confidence 0.420 is below the threshold 0.600"
# result.label      -> "human_handoff"  the weak handler was never invoked
```

Both branches are asserted by
[`test_confidence_gate_never_reaches_a_weak_handler`](tests/test_jev_route.py):
one test routes the same prompt through both configurations and checks that the
handler call list stays empty when the gate is on.

<details>
<summary><strong>Recording the terminal demo (asciinema / GIF)</strong></summary>

The demo is four short prompts, so 20–30 seconds of terminal output is enough.
`asciinema` is POSIX-only — on Windows run it inside WSL2
(`wsl --install`, then open Ubuntu) or use one of the alternatives below.

```bash
# 1. Check the timing first; the recording must not wait on anything.
python examples/basic_routing.py

# 2. Record a cast: --idle-time-limit caps long pauses, --cols keeps it narrow.
mkdir -p docs
asciinema rec docs/demo.cast --cols 100 --rows 32 --idle-time-limit 1.5 \
  --title "jev-route: confidence-gated intent routing" \
  -c "python examples/basic_routing.py"

# 3. Render a GIF from the cast (agg: cargo install agg, or brew install agg).
agg docs/demo.cast docs/demo.gif --font-size 14 --speed 1.5 --theme monokai

# 4. Optional: publish the replayable text version and link that instead.
asciinema upload docs/demo.cast
```

Windows-only option, if you would rather not install WSL:

```powershell
npm install -g terminalizer     # requires Node.js
terminalizer record demo        # stop the recording with Ctrl+D
terminalizer render demo        # -> demo.gif
```

Or record the terminal window with [ScreenToGif](https://www.screentogif.com/):
crop to the window, 12–15 fps, trim the first and last seconds, export a GIF
under ~2 MB. Keep either artefact in `docs/` and embed the GIF at the top of this
README:

```markdown
[![jev-route demo](docs/demo.gif)](docs/demo.cast)
```

Commit both files: the GIF is what people see on the repository page, the `.cast`
is what they can replay at their own speed.

</details>

### Embed it in a web service (FastAPI)

[`examples/fastapi_service.py`](examples/fastapi_service.py) puts the same router
behind HTTP: the service registers a (faked) refund tool, a read-only order
lookup and a knowledge-base agent, and only a high-confidence decision reaches
the refund handler — everything else is escalated.

```bash
python -m pip install "jev-route[fastapi]"
python examples/fastapi_service.py        # serves http://127.0.0.1:8000

curl -X POST http://127.0.0.1:8000/route \
  -H "content-type: application/json" \
  -d '{"message": "Siparişimi iade etmek istiyorum", "context": {"order_id": "A-1042"}}'

curl -X POST http://127.0.0.1:8000/route \
  -H "content-type: application/json" \
  -d '{"message": "Müşteri şikayetini çözer misin?"}'
```

The endpoints are `GET /healthz` (readiness, model, threshold),
`GET /options` (what the engine may choose from), `POST /route` (route one
message) and `GET /audit` (every decision, with the cache and latency
annotations). The example runs offline through an `httpx.MockTransport` when no
API key is set, and
[`tests/test_fastapi_service.py`](tests/test_fastapi_service.py) drives it
through FastAPI's `TestClient`, including the assertion that a low-confidence
message never reaches the refund handler.

### Run the tests

```bash
python -m pytest          # 19 offline tests, no API key required
```

### Configuration

JevRoute is configured through `JevSettings.from_env()`; every value can also be
passed explicitly (e.g. `JevClient(api_key=..., model=...)`).

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_KEY` / `JEV_API_KEY` | *(empty)* | Bearer token for the gateway |
| `JEV_MODEL` | `typesafe/jev-1.13` | Decision model identifier |
| `JEV_BASE_URL` | `https://openrouter.ai/api/v1` | API root (any OpenAI-compatible Jev gateway) |
| `JEV_CONFIDENCE_THRESHOLD` | `0.6` | Default confidence gate |
| `JEV_TIMEOUT` | `30` | Per-request timeout in seconds |
| `JEV_MAX_RETRIES` | `2` | Extra attempts for timeouts, `429` and `5xx` |

## Project layout

```
jev-route/
├── .github/workflows/ci.yml   # ruff + mypy + pytest + build (Python 3.10-3.12)
├── LICENSE                    # MIT
├── README.md
├── documentation.md           # roadmap, architecture decisions, live status
├── pyproject.toml             # packaging + ruff/mypy/pytest configuration
├── examples/
│   ├── basic_routing.py       # offline-capable end-to-end demo
│   └── fastapi_service.py     # the same router behind HTTP endpoints
├── tests/
│   ├── test_jev_route.py      # smoke tests driven by basic_routing.py
│   ├── test_fastapi_service.py # TestClient tests driven by fastapi_service.py
│   └── test_packaging.py      # version and PyPI metadata stay in sync
└── src/
    └── jev_route/
        ├── __init__.py        # public API
        ├── core.py            # Jev data structures, JevClient, confidence gate
        ├── router.py          # IntentRouter: registration, decide, execute
        ├── middleware.py      # async middleware pipeline
        └── py.typed           # PEP 561 marker: the package ships its types
```

## Development

```bash
python -m pip install -e ".[dev]"

python -m ruff check .     # lint          -> All checks passed!
python -m mypy src         # type check    -> Success: no issues found
python -m pytest           # tests         -> 19 passed

python -m build            # sdist + wheel -> dist/
python -m twine check dist/*   # -> PASSED for both artefacts
```

All three gates run in CI on every push and pull request; see
[`documentation.md`](documentation.md) for the roadmap and current status.

## License

MIT — see [LICENSE](LICENSE).


