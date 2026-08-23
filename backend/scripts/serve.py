"""Run the Faceless FRS API, and migrate file-based enrollments into it.

    python scripts/serve.py                     # start the API
    python scripts/serve.py --import-enrollments  # move data/enrollment into the DB
    python scripts/serve.py --db-url postgresql+psycopg://user:pass@host/frs

Interactive docs at http://127.0.0.1:8000/docs once running.

The database holds biometric templates and the full match audit trail. Set
FRS_TEMPLATE_ENCRYPTION_KEY before enrolling anyone, or templates are stored in
the clear and every write logs a warning saying so.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.db.repository import (  # noqa: E402
    WatchlistRepository,
    create_schema,
    make_engine,
    session_factory,
)
from app.matching.gallery import TEMPLATE_KEY_ENV, GalleryStore  # noqa: E402


def default_db_url(settings) -> str:
    return f"sqlite:///{settings.paths.data_dir / 'faceless_frs.db'}"


def import_enrollments(args) -> int:
    """Copy file-based enrollments (phase 2) into the database (phase 7)."""
    settings = get_settings()
    setup_logging(settings.logging.level)

    store = GalleryStore(settings)
    gallery = store.load_gallery()
    if len(gallery) == 0:
        print(f"Nothing to import: no enrollments under {store.root}")
        return 0

    engine = make_engine(args.db_url or default_db_url(settings))
    create_schema(engine)
    session = session_factory(engine)()
    repo = WatchlistRepository(session)

    imported = skipped = 0
    for person in gallery:
        try:
            repo.enroll(
                person.person_id,
                person.display_name,
                person.embeddings,
                notes=person.notes,
                source=person.source or "imported from file store",
                actor=args.operator,
                replace=args.force,
            )
            imported += 1
            print(f"  imported {person.person_id} "
                  f"({', '.join(m.value for m in person.modalities)})")
        except ValueError as exc:
            skipped += 1
            print(f"  skipped  {person.person_id}: {exc}")

    session.close()
    print(f"\n{imported} imported, {skipped} skipped.")
    if not os.environ.get(TEMPLATE_KEY_ENV):
        print(
            f"\nWARNING: {TEMPLATE_KEY_ENV} is not set, so those templates are "
            "stored unencrypted.\nSection 8 of the build plan requires "
            "encryption at rest."
        )
    return 0


def serve(args) -> int:
    settings = get_settings()
    setup_logging(settings.logging.level)

    if not os.environ.get(TEMPLATE_KEY_ENV):
        print(
            f"WARNING: {TEMPLATE_KEY_ENV} is not set. Any template enrolled "
            "through this\n         instance will be stored unencrypted.\n"
        )

    database_url = args.db_url or default_db_url(settings)
    settings.paths.data_dir.mkdir(parents=True, exist_ok=True)

    import uvicorn

    from app.api.main import create_app

    if args.demo:
        settings.demo_mode = True
        for line in (
            "",
            "  !! SIGN-IN IS OFF (--demo) !!",
            "  Anyone who can reach this port can read the watchlist and",
            "  confirm an identification. Every action is recorded against",
            "  the 'demo' operator, so the trail shows nothing about who",
            "  actually did it. Prototype use only.",
            "",
        ):
            print(line)

    app = create_app(settings, engine=make_engine(database_url))
    print(f"Database : {database_url}")
    print(f"Docs     : http://{args.host}:{args.port}/docs\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Skip sign-in entirely. For showing the prototype. Every action is "
            "recorded against a 'demo' operator and anyone who can reach the "
            "port can take any action, including confirming an identification."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--db-url", default=None,
        help="SQLAlchemy URL. Defaults to SQLite under the data directory.",
    )
    parser.add_argument(
        "--import-enrollments", action="store_true",
        help="Copy data/enrollment into the database, then exit.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="With --import-enrollments, replace entries that already exist.",
    )
    parser.add_argument("--operator", default="import-script")
    args = parser.parse_args(argv)

    return import_enrollments(args) if args.import_enrollments else serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
