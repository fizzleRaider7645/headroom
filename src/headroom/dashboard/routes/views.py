from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from headroom.dashboard.state import get_state

router = APIRouter()
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _status_color(used_fraction: float) -> str:
    if used_fraction >= 0.9:
        return "#ef4444"  # red
    if used_fraction >= 0.8:
        return "#f59e0b"  # amber
    return "#22c55e"  # green


def _api_key_warning() -> str | None:
    """Return a warning string if no API key is configured."""
    session = get_state().session
    if session is None:
        return None
    # Check if the client has an api_key set (non-empty)
    try:
        key = session._client.api_key
        if not key:
            return "ANTHROPIC_API_KEY is not set. Messages will fail until you restart with a valid key."
    except Exception:
        pass
    return None


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    state = get_state()
    session = state.session

    if session:
        u = session.token_usage
        status_data = {
            "used": u.used,
            "limit": u.limit,
            "headroom": u.headroom,
            "headroom_pct": u.headroom_pct,
            "used_pct": round(u.used_fraction * 100),
            "color": _status_color(u.used_fraction),
            "cache_hits": u.cache_hits,
            "turns": u.turns,
            "model": session.model,
            "session_name": state.session_name,
            "status": session.budget.status(u.used),
        }
        messages = session.history
        strategies = [
            {
                "name": s.name,
                "priority": s.priority,
                "enabled": state.strategy_enabled.get(s.name, True),
                "params": s.params,
            }
            for s in sorted(session._strategies, key=lambda x: x.priority)
        ]
    else:
        status_data = {"error": "No session loaded"}
        messages = []
        strategies = []

    # Pass budget bar variables flat so {% include "partials/budget_bar.html" %} works
    ctx: dict = {
        "request": request,
        "status": status_data,
        "messages": messages,
        "strategies": strategies,
        "api_key_warning": _api_key_warning(),
    }
    ctx.update(status_data)  # exposes headroom, used, limit, color, etc. at top level
    return templates.TemplateResponse(request, "index.html", ctx)


@router.get("/partials/budget_bar", response_class=HTMLResponse)
async def budget_bar_partial(request: Request):
    state = get_state()
    if not state.session:
        return HTMLResponse('<div class="error">No session</div>')
    u = state.session.token_usage
    color = _status_color(u.used_fraction)
    used_pct = round(u.used_fraction * 100)
    return templates.TemplateResponse(request, "partials/budget_bar.html", {
        "request": request,
        "used": u.used,
        "limit": u.limit,
        "headroom": u.headroom,
        "headroom_pct": u.headroom_pct,
        "used_pct": used_pct,
        "color": color,
        "cache_hits": u.cache_hits,
        "turns": u.turns,
    })


@router.get("/partials/message_list", response_class=HTMLResponse)
async def message_list_partial(request: Request):
    state = get_state()
    messages = state.session.history if state.session else []
    return templates.TemplateResponse(request, "partials/message_list.html", {"request": request, "messages": messages})


def _strategy_dict(name: str) -> dict | None:
    """Build the template context dict for a single strategy card."""
    state = get_state()
    if not state.session:
        return None
    for s in state.session._strategies:
        if s.name == name:
            return {
                "name": s.name,
                "priority": s.priority,
                "enabled": s.enabled,
                "params": s.params,
            }
    return None


@router.post("/partials/strategy/{name}/toggle", response_class=HTMLResponse)
async def toggle_strategy_partial(request: Request, name: str):
    """HTMX: toggle a strategy and return the updated card HTML."""
    state = get_state()
    if not state.session:
        return HTMLResponse('<div class="strategy-card error">No session</div>')

    current = state.strategy_enabled.get(name, True)
    state.strategy_enabled[name] = not current

    for s in state.session._strategies:
        if s.name == name:
            s.enabled = not current
            break

    strategy = _strategy_dict(name)
    return templates.TemplateResponse(
        request, "partials/strategy_card.html", {"request": request, "strategy": strategy}
    )


@router.post("/partials/strategy/{name}/param/{param_name}", response_class=HTMLResponse)
async def update_strategy_param(
    request: Request, name: str, param_name: str, value: str = Form(...)
):
    """HTMX: update a single strategy parameter and return the refreshed card."""
    state = get_state()
    if not state.session:
        return HTMLResponse('<div class="strategy-card error">No session</div>')

    for s in state.session._strategies:
        if s.name == name:
            try:
                s.set_param(param_name, value)
            except (ValueError, KeyError) as exc:
                return HTMLResponse(f'<div class="strategy-card error">{exc}</div>')
            break

    strategy = _strategy_dict(name)
    return templates.TemplateResponse(
        request, "partials/strategy_card.html", {"request": request, "strategy": strategy}
    )


@router.post("/partials/clear", response_class=HTMLResponse)
async def clear_chat(request: Request):
    """HTMX: clear session history and return an empty message list."""
    state = get_state()
    if state.session:
        state.session.clear()
        state.log_event("clear")
    return HTMLResponse("")


@router.post("/partials/send", response_class=HTMLResponse)
async def send_partial(request: Request, message: str = Form(...)):
    """HTMX endpoint: send message, return new message pair as HTML."""
    state = get_state()
    if not state.session:
        return HTMLResponse('<div class="error">No session loaded</div>')

    try:
        response = await state.session.send(message)
        assistant_text = response.content[0].text if response.content else ""
        state.log_event("message", headroom=state.session.token_usage.headroom)

        history = state.session.history
        # Return the last two messages (user + assistant)
        recent = history[-2:] if len(history) >= 2 else history
        return templates.TemplateResponse(request, "partials/message_pair.html", {"request": request, "messages": recent})
    except Exception as e:
        return HTMLResponse(f'<div class="error">Error: {e}</div>')
