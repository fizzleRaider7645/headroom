<![CDATA[<div align="center">

```
██╗  ██╗███████╗ █████╗ ██████╗ ██████╗  ██████╗  ██████╗ ███╗   ███╗
██║  ██║██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔═══██╗██╔═══██╗████╗ ████║
███████║█████╗  ███████║██║  ██║██████╔╝██║   ██║██║   ██║██╔████╔██║
██╔══██║██╔══╝  ██╔══██║██║  ██║██╔══██╗██║   ██║██║   ██║██║╚██╔╝██║
██║  ██║███████╗██║  ██║██████╔╝██║  ██║╚██████╔╝╚██████╔╝██║ ╚═╝ ██║
╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝     ╚═╝
```

**Context-aware token management for the Claude API**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square)](https://python.org)
[![Anthropic SDK](https://img.shields.io/badge/anthropic-≥0.49-orange?style=flat-square)](https://github.com/anthropic/anthropic-sdk-python)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

</div>

---

## What is headroom?

**headroom** is a Python library, CLI, and web dashboard that helps you keep long Claude conversations alive without hitting context limits. It wraps the Anthropic SDK with an automatic optimization pipeline — summarizing, filtering, and caching your message history so you always have room to think.

The name says it all: how much headroom do you have left before the context window fills up?

```
[headroom: 142,871 / 200,000 | 71% free | cache hits: 8]
```

---

## Features

| | |
|---|---|
| **Token budgeting** | Track usage in real time. Get warnings at 80%, automatic action at 90%, emergency truncation at 100% |
| **Summarization** | Compresses old history into summaries. Old messages → one compact TrackedMessage, marked pinned |
| **Relevance filtering** | Drops messages unrelated to the current task using keyword overlap (or embeddings if you want) |
| **Prompt caching** | Injects `cache_control` breakpoints at stable history prefixes — cuts API costs significantly |
| **Budget guard** | Last-resort fallback: drops oldest messages in batches to get back under the limit |
| **Session persistence** | Export and resume any conversation from a JSON file |
| **Web dashboard** | Real-time browser UI with token bar, message history, and strategy toggles |
| **CLI** | Interactive REPL, one-shot sends, token counting, session inspection |

---

## Installation

```bash
pip install headroom
```

With embedding-based relevance filtering (uses `sentence-transformers`):

```bash
pip install "headroom[embeddings]"
```

For development:

```bash
git clone https://github.com/you/headroom
cd headroom
pip install -e ".[dev]"
```

---

## Quickstart

### As a library

```python
import asyncio
from headroom import Session, TokenBudget

async def main():
    session = Session(
        model="claude-opus-4-6",
        budget=TokenBudget(limit=200_000),
        system="You are a helpful assistant.",
    )

    response = await session.send("Explain quantum entanglement simply.")
    print(response.content[0].text)

    usage = session.token_usage
    print(f"Used: {usage.used:,} / {usage.limit:,} — Headroom: {usage.headroom:,}")

asyncio.run(main())
```

### Synchronous

```python
from headroom import Session, TokenBudget

session = Session(model="claude-sonnet-4-6", budget=TokenBudget(limit=200_000))
response = session.send_sync("Tell me something interesting.")
print(response.content[0].text)
```

### With callbacks

```python
from headroom import Session, TokenBudget

def on_warn(event):
    print(f"⚠️  Context at {event['used_pct']}% — strategies activating")

def on_trim(event):
    print(f"✂️  Dropped {event['dropped']} messages to free space")

session = Session(
    model="claude-opus-4-6",
    budget=TokenBudget(limit=200_000, warn_at=0.75, act_at=0.85),
    on_warning=on_warn,
    on_trim=on_trim,
)
```

---

## CLI

```
Usage: headroom [OPTIONS] COMMAND [ARGS]...

Commands:
  chat       Interactive chat with live headroom tracking
  send       Send a single message
  count      Count tokens in text or a file
  inspect    Inspect a saved session file
  dashboard  Launch the web dashboard
```

### `headroom chat`

Start an interactive REPL. Headroom is shown after every response:

```bash
headroom chat --model claude-opus-4-6 --show-tokens --save session.json
```

```
You: What's the difference between TCP and UDP?
Assistant: TCP (Transmission Control Protocol) provides reliable, ordered...

[headroom: 198,441 / 200,000 | 99% free | cache hits: 0]
You: ▌
```

Flags:

| Flag | Description |
|------|-------------|
| `--model` | Claude model to use (default: `claude-opus-4-6`) |
| `--budget` | Token limit (default: model max) |
| `--system` | System prompt, or `@file.txt` to load from file |
| `--session-file` | Resume a saved session |
| `--save` | Auto-save session on exit |
| `--show-tokens` | Display headroom line after each turn |

### `headroom send`

One-shot message:

```bash
headroom send "Summarize the Rust ownership model" --show-tokens
headroom send "List 5 Python tips" --format json
```

### `headroom count`

Count tokens without sending a message:

```bash
headroom count "The quick brown fox"
headroom count --file context.txt --model claude-sonnet-4-6
cat prompt.txt | headroom count
```

### `headroom inspect`

View stats and message history for any saved session:

```bash
headroom inspect session.json
```

```
Session: my-chat  |  Model: claude-opus-4-6  |  Turns: 14
Used: 87,234 / 200,000  |  Headroom: 112,766  |  Cache hits: 6

 #   Role       Tokens   Flags
 1   user          142
 2   assistant     891   [cached]
 3   user          203
 4   assistant    1204   [cached]
 5   user          312
 6   assistant    4102   [summary of 4]  [pinned]
...
```

### `headroom dashboard`

Launch the browser UI:

```bash
headroom dashboard                        # new session
headroom dashboard session.json           # resume saved session
headroom dashboard --port 9000 --reload   # custom port with auto-reload
```

---

## Web Dashboard

A zero-JS-framework browser UI built with FastAPI + HTMX.

```
┌─────────────────────────────────────────────────────────────┐
│  ████████████████████████░░░░░░░░░░░░░  71% used           │
│  Headroom: 142,871  ·  Used: 57,129  ·  Limit: 200,000     │
├──────────────────────────────┬──────────────────────────────┤
│                              │  Strategies                  │
│  user                        │  ┌──────────────────────┐   │
│  ┌──────────────────────┐    │  │ SummarizationStrategy│ON │
│  │ Explain recursion... │    │  │ priority: 30         │   │
│  └──────────────────────┘    │  └──────────────────────┘   │
│                              │  ┌──────────────────────┐   │
│  assistant                   │  │ RelevanceFilter      │ON │
│  ┌──────────────────────┐    │  │ priority: 20         │   │
│  │ Recursion is when a  │    │  └──────────────────────┘   │
│  │ function calls...    │    │                              │
│  │             142 tkns │    │  Actions                     │
│  └──────────────────────┘    │  [ Export Session ]          │
│                              │  [ Clear Chat    ]           │
│  ┌──────────────────────┐    │                              │
│  │ Type a message...    │    │  Status                      │
│  └────────────[ Send ]──┘    │  Status: ok                  │
│                              │  Turns: 3                    │
│                              │  Cache hits: 4               │
└──────────────────────────────┴──────────────────────────────┘
```

**Features:**
- Budget bar refreshes every 2 seconds — color shifts green → amber → red as you approach the limit
- Message history shows token counts, `[pinned]`, `[summary of N]`, and `[cached]` badges
- Strategy toggles enable/disable each optimization strategy live
- Spinner and disabled Send button while a request is in-flight
- Input clears and message panel resets on each new exchange

---

## How the strategy pipeline works

Before every API call, `Session` runs the strategy pipeline in priority order:

```
User message received
        │
        ▼
┌───────────────────┐   priority 10 — only on overflow
│  BudgetGuard      │   Drop oldest non-pinned messages
└────────┬──────────┘
         │
         ▼
┌───────────────────┐   priority 20 — fires at ≥90% usage
│  RelevanceFilter  │   Score messages by keyword overlap
└────────┬──────────┘   with recent context; drop low scorers
         │
         ▼
┌───────────────────┐   priority 30 — fires at ≥90% usage
│  Summarizer       │   Chunk old messages, call Claude to
└────────┬──────────┘   summarize; replace with pinned summary
         │
         ▼
┌───────────────────┐   priority 90 — always runs
│  CacheInjector    │   Mark stable message prefix with
└────────┬──────────┘   cache_control breakpoints (up to 4)
         │
         ▼
  API call → response
```

Each strategy is a **pure function** over the message list — independently testable, composable, and replaceable.

---

## Pinned messages

Mark any message as immune to trimming:

```python
session.pin(message_id)          # pin by ID
session.add_context("...", pin=True)   # add and pin in one step
```

Pinned messages survive all strategies, including the last-resort budget guard.

---

## Session persistence

```python
# Save
session.export("my-chat.json")

# Resume
session = Session.load("my-chat.json")
response = await session.send("Where were we?")
```

The CLI also handles this automatically:

```bash
headroom chat --session-file my-chat.json --save
```

---

## Budget thresholds

```python
budget = TokenBudget(
    limit=200_000,   # context window size
    warn_at=0.80,    # 80% → on_warning callback fires
    act_at=0.90,     # 90% → strategies activate
    reserve=1024,    # tokens always held back for the response
)
```

| Status | Meaning |
|--------|---------|
| `ok` | Below `warn_at` — everything fine |
| `warn` | 80–90% used — warning callback fired |
| `act` | ≥90% used — strategies running |
| `overflow` | Over limit — BudgetGuard drops messages |

---

## Custom strategies

```python
from headroom.strategies.base import BaseStrategy, SessionContext
from headroom.core.message import TrackedMessage
from headroom.core.budget import TokenBudget

class MyStrategy(BaseStrategy):
    priority = 25  # runs between RelevanceFilter (20) and Summarizer (30)

    def apply(
        self,
        messages: list[TrackedMessage],
        budget: TokenBudget,
        used_tokens: int,
        ctx: SessionContext,
    ) -> list[TrackedMessage]:
        # your logic here — return the modified message list
        return [m for m in messages if not should_drop(m)]

session = Session(
    model="claude-opus-4-6",
    strategies=[MyStrategy(), *default_strategies()],
)
```

---

## REST API

When the dashboard is running, a JSON API is available alongside the UI:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/status` | Token usage snapshot |
| `GET` | `/api/messages` | Full message history as JSON |
| `POST` | `/api/message` | Send a message (form: `message=...`) |
| `GET` | `/api/session/export` | Download session as JSON |
| `POST` | `/api/strategy/{name}/toggle` | Enable/disable a strategy |
| `GET` | `/api/events` | Last 50 logged events |

---

## Environment

Set your API key before running:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Or pass it directly:

```python
session = Session(model="claude-opus-4-6", api_key="sk-ant-...")
```

The dashboard shows a warning banner if no key is detected.

---

## Testing

```bash
pip install -e ".[dev]"
pytest
```

All tests mock the Anthropic client — no real API calls, no quota burned in CI. Tests marked `@pytest.mark.real_api` are skipped unless you pass `--run-live` with a valid key.

---

## Project layout

```
headroom/
├── src/headroom/
│   ├── core/
│   │   ├── session.py        # Session — primary user-facing class
│   │   ├── message.py        # TrackedMessage dataclass
│   │   └── budget.py         # TokenBudget + TokenUsage
│   ├── strategies/
│   │   ├── budget.py         # BudgetGuardStrategy   (priority 10)
│   │   ├── relevance.py      # RelevanceFilterStrategy (priority 20)
│   │   ├── summarizer.py     # SummarizationStrategy  (priority 30)
│   │   └── cache.py          # CacheInjectionStrategy (priority 90)
│   ├── counting/
│   │   └── counter.py        # TokenCounter (exact SDK + heuristic)
│   ├── cli/
│   │   └── main.py           # Click CLI entry point
│   └── dashboard/
│       ├── app.py            # FastAPI application factory
│       ├── state.py          # Singleton DashboardState
│       ├── routes/
│       │   ├── api.py        # JSON REST endpoints
│       │   └── views.py      # HTMX HTML endpoints
│       └── templates/        # Jinja2 + HTMX templates
├── tests/
│   ├── unit/                 # Pure unit tests (no mocks needed for strategies)
│   └── integration/          # Dashboard tests via FastAPI TestClient
└── examples/
    ├── basic_chat.py
    └── long_context.py
```

---

<div align="center">

Built for developers who talk to Claude a lot.

</div>
]]>