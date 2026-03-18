from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from headroom.core.budget import TokenBudget
from headroom.core.session import Session


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_session(
    model: str,
    budget: int | None,
    system: str | None,
    session_file: str | None,
    api_key: str | None,
) -> Session:
    kwargs: dict = {"model": model}
    if api_key:
        kwargs["api_key"] = api_key
    if system:
        kwargs["system"] = system
    if budget:
        kwargs["budget"] = TokenBudget(limit=budget)

    if session_file and Path(session_file).exists():
        session = Session.load(session_file, **{k: v for k, v in kwargs.items() if k != "model"})
        session.model = model
        return session

    return Session(**kwargs)


def _headroom_line(session: Session) -> str:
    u = session.token_usage
    bar_width = 20
    filled = int(u.used_fraction * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)

    # Color
    if u.used_fraction >= 0.9:
        color = "\033[31m"  # red
    elif u.used_fraction >= 0.8:
        color = "\033[33m"  # yellow
    else:
        color = "\033[32m"  # green
    reset = "\033[0m"

    return (
        f"{color}[{bar}]{reset} "
        f"{u.used:,} / {u.limit:,} tokens | "
        f"headroom: {u.headroom:,} ({u.headroom_pct}% free) | "
        f"cache hits: {u.cache_hits} | "
        f"turns: {u.turns}"
    )


# ------------------------------------------------------------------
# CLI group
# ------------------------------------------------------------------

@click.group()
@click.version_option(package_name="headroom")
def cli():
    """headroom — optimize context and token usage for Claude conversations."""


# ------------------------------------------------------------------
# headroom chat
# ------------------------------------------------------------------

@cli.command()
@click.option("--model", default="claude-opus-4-6", show_default=True)
@click.option("--budget", type=int, default=None, help="Context window limit (tokens)")
@click.option("--system", default=None, help="System prompt text or @file.txt")
@click.option("--session-file", default=None, help="Resume from saved session file")
@click.option("--save", default=None, help="Auto-save session to this file on exit")
@click.option("--show-tokens", is_flag=True, default=False)
@click.option("--api-key", envvar="ANTHROPIC_API_KEY", default=None, hidden=True)
def chat(model, budget, system, session_file, save, show_tokens, api_key):
    """Start an interactive chat session with live headroom tracking."""
    # Handle @file.txt system prompt
    if system and system.startswith("@"):
        system_path = Path(system[1:])
        if not system_path.exists():
            click.echo(f"Error: system prompt file not found: {system_path}", err=True)
            sys.exit(1)
        system = system_path.read_text()

    session = _make_session(model, budget, system, session_file, api_key)

    click.echo(f"headroom chat | model: {model} | budget: {session.budget.limit:,} tokens")
    click.echo("Type 'exit' or Ctrl+C to quit.\n")

    try:
        while True:
            try:
                user_input = click.prompt("You", prompt_suffix="> ")
            except click.Abort:
                break

            if user_input.strip().lower() in ("exit", "quit", "/exit", "/quit"):
                break

            try:
                response = session.send_sync(user_input)
                assistant_text = response.content[0].text if response.content else ""
                click.echo(f"\nAssistant: {assistant_text}\n")
                if show_tokens:
                    click.echo(_headroom_line(session) + "\n")
            except Exception as e:
                click.echo(f"\nError: {e}\n", err=True)

    except KeyboardInterrupt:
        pass

    click.echo("\nGoodbye!")
    if save:
        session.export(save)
        click.echo(f"Session saved to {save}")


# ------------------------------------------------------------------
# headroom send
# ------------------------------------------------------------------

@cli.command()
@click.argument("message")
@click.option("--model", default="claude-opus-4-6", show_default=True)
@click.option("--system", default=None)
@click.option("--show-tokens", is_flag=True, default=False)
@click.option("--format", "fmt", type=click.Choice(["plain", "json"]), default="plain")
@click.option("--api-key", envvar="ANTHROPIC_API_KEY", default=None, hidden=True)
def send(message, model, system, show_tokens, fmt, api_key):
    """Send a single message and print the response."""
    session = _make_session(model, None, system, None, api_key)
    try:
        response = session.send_sync(message)
        text = response.content[0].text if response.content else ""
        if fmt == "json":
            u = session.token_usage
            click.echo(json.dumps({
                "response": text,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "headroom": u.headroom,
                },
            }))
        else:
            click.echo(text)
            if show_tokens:
                click.echo("\n" + _headroom_line(session))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# ------------------------------------------------------------------
# headroom count
# ------------------------------------------------------------------

@cli.command()
@click.argument("text", required=False)
@click.option("--file", "file_path", type=click.Path(exists=True), default=None)
@click.option("--model", default="claude-opus-4-6", show_default=True)
@click.option("--api-key", envvar="ANTHROPIC_API_KEY", default=None, hidden=True)
def count(text, file_path, model, api_key):
    """Count tokens in text or a file."""
    import anthropic as _anthropic
    from headroom.core.message import TrackedMessage
    from headroom.counting.counter import TokenCounter

    if file_path:
        content = Path(file_path).read_text()
    elif text:
        content = text
    else:
        content = click.get_text_stream("stdin").read()

    client = _anthropic.Anthropic(api_key=api_key)
    counter = TokenCounter(client, model)
    msg = TrackedMessage(role="user", content=content)
    total = counter.count_exact([msg])
    click.echo(f"{total:,} tokens")


# ------------------------------------------------------------------
# headroom inspect
# ------------------------------------------------------------------

@cli.command()
@click.argument("session_file", type=click.Path(exists=True))
def inspect(session_file):
    """Show stats and message list for a saved session."""
    session = Session.load(session_file)
    u = session.token_usage

    click.echo(f"\nSession: {session_file}")
    click.echo(f"  Model:       {session.model}")
    click.echo(f"  Turns:       {u.turns}")
    click.echo(f"  Used tokens: {u.used:,}")
    click.echo(f"  Limit:       {u.limit:,}")
    click.echo(f"  Headroom:    {u.headroom:,} ({u.headroom_pct}% free)")
    click.echo(f"  Cache hits:  {u.cache_hits}")
    click.echo(f"  Cache read:  {u.cache_read_tokens:,} tokens")
    click.echo(f"  Cache write: {u.cache_write_tokens:,} tokens")

    click.echo(f"\nMessages ({len(session.history)}):")
    for msg in session.history:
        badges = []
        if msg.pinned:
            badges.append("[pinned]")
        if msg.summary_of:
            badges.append(f"[summary of {len(msg.summary_of)}]")
        if msg.cache_breakpoint:
            badges.append("[cached]")
        badge_str = " ".join(badges)
        content_preview = (
            msg.content[:80] + "..."
            if isinstance(msg.content, str) and len(msg.content) > 80
            else (msg.content if isinstance(msg.content, str) else "[structured content]")
        )
        click.echo(
            f"  [{msg.id:>4}] {msg.role:<10} {msg.token_count:>6} tok  "
            f"{badge_str:<25} {content_preview}"
        )


# ------------------------------------------------------------------
# headroom dashboard
# ------------------------------------------------------------------

@cli.command()
@click.argument("session_file", required=False, type=click.Path())
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True)
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload")
@click.option("--api-key", envvar="ANTHROPIC_API_KEY", default=None, hidden=True)
def dashboard(session_file, host, port, reload, api_key):
    """Launch the web dashboard."""
    try:
        import uvicorn
    except ImportError:
        click.echo("Error: uvicorn is required. Run: pip install headroom", err=True)
        sys.exit(1)

    from headroom.dashboard.state import get_state
    from headroom.dashboard.app import create_app

    state = get_state()

    if session_file and Path(session_file).exists():
        state.session = Session.load(session_file, api_key=api_key)
        state.session_name = Path(session_file).stem
    else:
        state.session = Session(model="claude-opus-4-6", api_key=api_key)
        state.session_name = "new-session"

    click.echo(f"Starting headroom dashboard at http://{host}:{port}")

    if reload:
        uvicorn.run(
            "headroom.dashboard.app:create_app",
            host=host, port=port, reload=True, factory=True,
        )
    else:
        uvicorn.run(create_app(), host=host, port=port)
