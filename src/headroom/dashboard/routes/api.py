from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse, Response

from headroom.dashboard.state import get_state

router = APIRouter()


@router.get("/status")
async def get_status():
    state = get_state()
    if state.session is None:
        return JSONResponse({"error": "no session"}, status_code=503)
    u = state.session.token_usage
    budget = state.session.budget
    return {
        "used": u.used,
        "limit": u.limit,
        "headroom": u.headroom,
        "headroom_pct": u.headroom_pct,
        "used_fraction": round(u.used_fraction, 4),
        "cache_hits": u.cache_hits,
        "cache_read_tokens": u.cache_read_tokens,
        "cache_write_tokens": u.cache_write_tokens,
        "turns": u.turns,
        "status": budget.status(u.used),
        "session_name": state.session_name,
        "model": state.session.model,
    }


@router.get("/messages")
async def get_messages():
    state = get_state()
    if state.session is None:
        return []
    return [m.to_json() for m in state.session.history]


@router.post("/message")
async def post_message(message: str = Form(...)):
    state = get_state()
    if state.session is None:
        raise HTTPException(status_code=503, detail="No session")
    try:
        response = await state.session.send(message)
        assistant_text = response.content[0].text if response.content else ""
        state.log_event("message_sent", headroom=state.session.token_usage.headroom)
        return {"assistant": assistant_text, "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/export")
async def export_session():
    import json
    import tempfile
    from pathlib import Path

    state = get_state()
    if state.session is None:
        raise HTTPException(status_code=503, detail="No session")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = Path(f.name)

    state.session.export(tmp_path)
    content = tmp_path.read_text()
    tmp_path.unlink(missing_ok=True)

    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{state.session_name}.json"'},
    )


@router.post("/strategy/{name}/toggle")
async def toggle_strategy(name: str):
    state = get_state()
    if state.session is None:
        raise HTTPException(status_code=503, detail="No session")

    current = state.strategy_enabled.get(name, True)
    state.strategy_enabled[name] = not current

    # Enable/disable in session
    for strategy in state.session._strategies:
        if strategy.name.lower() == name.lower():
            strategy._disabled = not current
            break

    return {"name": name, "enabled": not current}


@router.get("/events")
async def get_events():
    return get_state().event_log[-50:]  # last 50 events
