# headroom

**Context-aware token management for the Claude API**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square)](https://python.org)
[![Anthropic SDK](https://img.shields.io/badge/anthropic-%E2%89%A50.49-orange?style=flat-square)](https://github.com/anthropics/anthropic-sdk-python)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

---

**headroom** is a Python library, CLI, and web dashboard that keeps long Claude conversations alive without hitting context limits. It wraps the Anthropic SDK with an automatic optimization pipeline — summarizing, filtering, and caching your message history so you always have room to think.

```
[headroom: 142,871 / 200,000 | 71% free | cache hits: 8]
```

---

## Features

| | |
|---|---|
| **Token budgeting** | Track usage in real time. Warnings at 80%, strategies activate at 90%, emergency truncation at overflow |
| **Summarization** | Compresses old history into summaries — old messages become one pinned TrackedMessage |
| **Relevance filtering** | Drops messages unrelated to the current task using keyword overlap (or embeddings, opt-in) |
| **Prompt caching** | Injects `cache_control` breakpoints at stable prefixes — cuts API costs significantly |
| **Budget guard** | Last-resort fallback: drops oldest non-pinned messages until back under the limit |
| **Session persistence** | Export and resume any conversation from a JSON file |
| **Web dashboard** | Real-time browser UI with token bar, message history, and strategy toggles |
| **CLI** | Interactive REPL, one-shot sends, token counting, session inspection |

---

## Installation

```bash
pip install headroom
```

With embedding-based relevance filtering:

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

### Library

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

session = Session(
    model="claude-opus-4-6",
    budget=TokenBudget(limit=200_000, warn_at=0.75, act_at=0.85),
    on_warning=lambda e: print(f"Warning: context at {e['used_pct']}%"),
    on_trim=lambda e: print(f"Trimmed {e['dropped']} messages"),
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

```bash
headroom chat --model claude-opus-4-6 --show-tokens --save session.json
```

```
You: What's the difference between TCP and UDP?
Assistant: TCP (Transmission Control Protocol) provides reliable, ordered...

[headroom: 198,441 / 200,000 | 99% free | cache hits: 0]
You:
```

| Flag | Description |
|------|-------------|
| `--model` | Claude model (default: `claude-opus-4-6`) |
| `--budget` | Token limit (default: model max) |
| `--system` | System prompt, or `@file.txt` to load from file |
| `--session-file` | Resume a saved session |
| `--save` | Auto-save session on exit |
| `--show-tokens` | Display headroom line after each turn |

### `headroom send`

```bash
headroom send "Summarize the Rust ownership model" --show-tokens
headroom send "List 5 Python tips" --format json
```

### `headroom count`

```bash
headroom count "The quick brown fox"
headroom count --file context.txt --model claude-sonnet-4-6
cat prompt.txt | headroom count
```

### `headroom inspect`

```bash
headroom inspect session.json
```

```
Session: my-chat  |  Model: claude-opus-4-6  |  Turns: 14
Used: 87,234 / 200,000  |  Headroom: 112,766  |  Cache hits: 6

 #   Role        Tokens   Flags
 1   user           142
 2   assistant      891   [cached]
 3   user           203
 4   assistant     1204   [cached]
 5   assistant     4102   [summary of 4]  [pinned]
```

### `headroom dashboard`

```bash
headroom dashboard                       # new session
headroom dashboard session.json          # resume saved session
headroom dashboard --port 9000 --reload  # custom port, auto-reload
```

---

## Web Dashboard

A zero-JS-framework browser UI built with FastAPI + HTMX.

```
┌─────────────────────────────────────────────────────────────────┐
│  ██████████████████████░░░░░░░░░  71% used                      │
│  Headroom: 142,871  ·  Used: 57,129  ·  Limit: 200,000         │
├───────────────────────────────────┬─────────────────────────────┤
│                                   │  Strategies                 │
│  user                             │                             │
│  ┌─────────────────────────────┐  │  Summarization       [ ON ] │
│  │ Explain recursion...        │  │  priority: 30               │
│  └─────────────────────────────┘  │                             │
│                                   │  RelevanceFilter     [ ON ] │
│  assistant                        │  priority: 20               │
│  ┌─────────────────────────────┐  │                             │
│  │ Recursion is when a         │  │  CacheInjection      [ ON ] │
│  │ function calls itself...    │  │  priority: 90               │
│  │                   142 tkns  │  │                             │
│  └─────────────────────────────┘  │  Actions                    │
│                                   │  [ Export Session ]         │
│  ┌──────────────────────┐         │  [ Clear Chat      ]        │
│  │ Type a message...    │ [Send]  │                             │
│  └──────────────────────┘         │  Status: ok                 │
│                                   │  Turns: 3 · Cache hits: 4   │
└───────────────────────────────────┴─────────────────────────────┘
```

- Budget bar refreshes every 2 seconds — color shifts green → amber → red as the limit approaches
- Each message shows token count plus `[pinned]`, `[summary of N]`, and `[cached]` badges
- Strategy toggles enable/disable each optimization live
- Loading spinner and disabled Send button while a request is in-flight
- Input clears and message panel resets on each new exchange

---

## How the strategy pipeline works

Before every API call, `Session` runs the strategy pipeline in priority order:

```
User message received
        │
        ▼
┌────────────────────┐   priority 10 — only on overflow
│  BudgetGuard       │   Drop oldest non-pinned messages
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐   priority 20 — fires at ≥90% usage
│  RelevanceFilter   │   Score messages by keyword overlap
└─────────┬──────────┘   with recent context; drop low scorers
          │
          ▼
┌────────────────────┐   priority 30 — fires at ≥90% usage
│  Summarizer        │   Chunk old messages, call Claude to
└─────────┬──────────┘   summarize; replace with pinned summary
          │
          ▼
┌────────────────────┐   priority 90 — always runs
│  CacheInjector     │   Mark stable prefix with cache_control
└─────────┬──────────┘   breakpoints (up to 4)
          │
          ▼
   API call → response
```

Each strategy is a **pure function** over the message list — independently testable, composable, and replaceable.

---

## Budget thresholds

```python
budget = TokenBudget(
    limit=200_000,   # context window size
    warn_at=0.80,    # 80%  → on_warning callback fires
    act_at=0.90,     # 90%  → strategies activate
    reserve=1024,    # tokens always held back for the response
)
```

| Status | Range | Behavior |
|--------|-------|----------|
| `ok` | < 80% | Everything fine |
| `warn` | 80–90% | `on_warning` callback fires |
| `act` | ≥ 90% | Strategies run |
| `overflow` | > 100% | BudgetGuard drops messages |

---

## Pinned messages

Pinned messages are immune to all trimming strategies, including the last-resort budget guard:

```python
session.pin(message_id)
session.add_context("Important context...", pin=True)
```

---

## Session persistence

```python
# Save
session.export("my-chat.json")

# Resume
session = Session.load("my-chat.json")
response = await session.send("Where were we?")
```

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
        return [m for m in messages if not my_filter(m)]

from headroom import default_strategies
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
| `POST` | `/api/message` | Send a message (`form: message=...`) |
| `GET` | `/api/session/export` | Download session as JSON |
| `POST` | `/api/strategy/{name}/toggle` | Enable / disable a strategy |
| `GET` | `/api/events` | Last 50 logged events |

---

## Environment

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Or pass directly: `Session(model="...", api_key="sk-ant-...")`

The dashboard shows a warning banner if no key is detected.

---

## Testing

```bash
pip install -e ".[dev]"
pytest
```

All tests mock the Anthropic client — no real API calls, no quota burned in CI.

---

## Project layout

```
headroom/
├── src/headroom/
│   ├── core/
│   │   ├── session.py          # Session — primary user-facing class
│   │   ├── message.py          # TrackedMessage dataclass
│   │   └── budget.py           # TokenBudget + TokenUsage
│   ├── strategies/
│   │   ├── budget.py           # BudgetGuardStrategy    (priority 10)
│   │   ├── relevance.py        # RelevanceFilterStrategy (priority 20)
│   │   ├── summarizer.py       # SummarizationStrategy   (priority 30)
│   │   └── cache.py            # CacheInjectionStrategy  (priority 90)
│   ├── counting/
│   │   └── counter.py          # TokenCounter (exact SDK + heuristic)
│   ├── cli/
│   │   └── main.py             # Click CLI entry point
│   └── dashboard/
│       ├── app.py              # FastAPI application factory
│       ├── state.py            # Singleton DashboardState
│       ├── routes/
│       │   ├── api.py          # JSON REST endpoints
│       │   └── views.py        # HTMX HTML endpoints
│       └── templates/          # Jinja2 + HTMX, no build step
├── tests/
│   ├── unit/
│   └── integration/
└── examples/
    ├── basic_chat.py
    └── long_context.py
```

---

Built for developers who talk to Claude a lot.
