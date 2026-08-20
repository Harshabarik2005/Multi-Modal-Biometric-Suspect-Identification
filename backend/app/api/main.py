"""FastAPI application factory (Phase 7).

    cd backend && python -m uvicorn app.api.main:app --reload

Interactive docs at http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.engine import Engine

from app.core.config import Settings, get_settings
from app.core.logging import get_logger, setup_logging
from app.db.repository import create_schema, make_engine, session_factory
from app.api import routes

logger = get_logger(__name__)

DESCRIPTION = """
Multi-modal biometric suspect identification: face, gait and re-ID fused into
one adaptive score.

**No action follows from a match alone.** The matcher only ever creates
*pending* decisions; moving one to confirmed requires a named human operator
via `POST /decisions/{id}/review`. Alerting (phase 9) reads `/alerts`, which
returns confirmed decisions only.

Every decision records the per-modality weights that produced it, so a reviewer
can see whether a match rested on a clear face or mostly on a jacket.

Biometric templates are encrypted at rest and are never returned over this API.
"""


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.logging.level)

    if engine is None:
        database_url = getattr(settings, "database_url", None) or (
            f"sqlite:///{settings.paths.data_dir / 'faceless_frs.db'}"
        )
        settings.paths.data_dir.mkdir(parents=True, exist_ok=True)
        engine = make_engine(database_url)
        logger.info("Database: %s", database_url)

    create_schema(engine)
    make_session = session_factory(engine)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logger.info("%s API starting", settings.project_name)
        yield
        engine.dispose()

    app = FastAPI(
        title=f"{settings.project_name} API",
        description=DESCRIPTION,
        version="0.7.0",
        lifespan=lifespan,
    )

    def get_session():
        session = make_session()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[routes.get_session] = get_session
    app.include_router(routes.router, prefix="/api")
    return app


app = create_app()
