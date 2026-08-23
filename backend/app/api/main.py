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
from app.api import auth, ingest, routes
from app.api.jobs import JobRunner

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

Authentication
--------------
Every route except `/health` requires a signed-in operator. `POST /auth/login`
exchanges a password for a bearer token; send it as `Authorization: Bearer ...`.

The operator recorded on a review comes from that token, never from the request
body. Create the first account with:

    python scripts/manage_operators.py --create <username>

Uploading
---------
`POST /api/enroll` takes photos and/or video of one person; `POST /api/scan`
takes CCTV footage and searches it. Both run in the background and return a job
id -- poll `GET /api/jobs/{id}`. Uploaded files are deleted as soon as the job
finishes, because enrolment footage is personal data and keeping it after the
embeddings are extracted creates a second copy to protect for no benefit.

`POST /api/enroll/preview` runs detection only and reports which of the three
signals an upload can actually support. Worth calling first: a photograph
contains no gait information at all, so a photos-only enrolment silently
produces a profile that can never match on how someone walks.
"""

#: Set by `create_app` so background jobs can open their own database session.
#: Jobs outlive the request that started them, so they cannot borrow its.
session_maker_for_app = None


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

    global session_maker_for_app
    session_maker_for_app = make_session

    runner = JobRunner(
        # One worker on purpose: the models contend for the same GPU memory,
        # so two concurrent scans make both slower and risk running out.
        max_workers=getattr(settings, "job_workers", 1)
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logger.info("%s API starting", settings.project_name)
        # Uploads left by a process that did not stop cleanly. The runner
        # handles cancelled and never-started jobs, but nothing in-process
        # survives a kill -9, and these directories hold footage of real
        # people (SEC-11).
        from app.api.ingest import sweep_stale_uploads

        sweep_stale_uploads()
        yield
        runner.shutdown()
        engine.dispose()

    app = FastAPI(
        title=f"{settings.project_name} API",
        description=DESCRIPTION,
        version="0.9.0",
        lifespan=lifespan,
    )

    def get_session():
        session = make_session()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[routes.get_session] = get_session
    app.dependency_overrides[ingest.get_session] = get_session
    app.dependency_overrides[auth.get_auth_session] = get_session
    app.dependency_overrides[ingest.get_runner] = lambda: runner
    app.dependency_overrides[ingest.get_app_settings] = lambda: settings

    app.include_router(routes.router, prefix="/api")
    app.include_router(ingest.router, prefix="/api")
    app.state.job_runner = runner
    return app


app = create_app()
