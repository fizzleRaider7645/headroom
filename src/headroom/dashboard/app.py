from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from headroom.dashboard.routes.api import router as api_router
from headroom.dashboard.routes.views import router as views_router

TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app() -> FastAPI:
    app = FastAPI(title="headroom", docs_url=None, redoc_url=None)

    app.include_router(views_router)
    app.include_router(api_router, prefix="/api")

    return app
