"""Run the Faceless FRS API, and migrate file-based enrollments into it.

    python scripts/serve.py                     # start the API
    python scripts/serve.py --import-enrollments  # move data/enrollment into the DB
    python scripts/serve.py --db-url postgresql+psycopg://user:pass@host/frs

Interactive docs at http://127.0.0.1:8000/docs once running.

The database holds biometric templates and the full match audit trail.
FRS_TEMPLATE_ENCRYPTION_KEY MUST be set before enrolling anyone -- without
it, enrolment is refused outright rather than falling back to plaintext
(SEC-06):

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    export FRS_TEMPLATE_ENCRYPTION_KEY=<that key>

For showing the prototype without any of that, use --demo: it skips sign-in
and, as long as no key is set, skips the encryption requirement too. For
throwaway local data without --demo, FRS_ALLOW_PLAINTEXT_TEMPLATES=1 opts
back into writing plaintext deliberately.
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
from app.matching.gallery import (  # noqa: E402
    ALLOW_PLAINTEXT_ENV,
    TEMPLATE_KEY_ENV,
    GalleryStore,
)


def default_db_url(settings) -> str:
    return f"sqlite:///{settings.paths.data_dir / 'faceless_frs.db'}"


#: What the pipeline needs at runtime but imports lazily, mapped to the pip
#: name that provides it.
#:
#: The lazy imports are deliberate -- ultralytics and insightface are slow and
#: heavy, and tests that never touch a frame should not pay for them. The cost
#: is that a server missing them starts perfectly, serves every route, renders
#: the whole console, and then throws a 500 the first time somebody presses
#: Register or Check. That is the worst possible moment to find out: after the
#: form is filled in and the footage is uploaded.
PIPELINE_REQUIREMENTS = {
    "cv2": "opencv-python",
    "torch": "torch",
    "ultralytics": "ultralytics",
    "deep_sort_realtime": "deep-sort-realtime",
    "insightface": "insightface",
}


def missing_pipeline_requirements() -> list[str]:
    """Pip names of anything the pipeline needs and cannot import."""
    import importlib.util

    missing = []
    for module, package in PIPELINE_REQUIREMENTS.items():
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(package)
    return missing


def find_project_venv() -> Path | None:
    """The project's virtualenv interpreter, if there is one and we are not it.

    Returns None when already running inside it, so the "you're using the
    wrong Python" advice is only given when that is actually the problem.
    """
    running = Path(sys.executable).resolve()
    for root in (BACKEND_ROOT.parent, BACKEND_ROOT):
        for relative in ("Scripts/python.exe", "bin/python"):
            candidate = root / ".venv" / relative
            if candidate.is_file() and candidate.resolve() != running:
                return candidate
    return None


def port_is_free(host: str, port: int) -> bool:
    """Whether the server could actually bind here.

    Deliberately no SO_REUSEADDR. On POSIX it permits binding a port still in
    TIME_WAIT, which is precisely the "recently in use" state worth reporting;
    and on Windows it muddies the result -- measured here, binding an occupied
    port with it set fails with errno 13 rather than the WSAEADDRINUSE (10048)
    you would expect and want to see. A plain bind answers the actual question.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def report_port_in_use(host: str, port: int) -> None:
    """Say what is wrong in words, before uvicorn says it in errno."""
    lines = [
        "",
        f"  !! PORT {port} IS ALREADY IN USE !!",
        "",
        f"  Something is already listening on {host}:{port} -- most often",
        "  another copy of this server that was never shut down.",
        "",
        "  Find it:",
        f"    netstat -ano | findstr :{port}          (Windows)",
        f"    lsof -i :{port}                          (macOS / Linux)",
        "",
        "  Then stop that process, or just use a different port:",
        "",
        f"    python scripts/serve.py --demo --port {port + 1}",
        "",
        "  Refusing to start. Carrying on would print a working-looking",
        "  banner and a Docs URL that nothing is serving.",
        "",
    ]
    for line in lines:
        print(line)


def report_missing_requirements(missing: list[str]) -> None:
    """Explain what is missing and, more usefully, why."""
    venv = find_project_venv()

    lines = [
        "",
        "  !! THE PIPELINE CANNOT RUN -- DEPENDENCIES ARE MISSING !!",
        "",
        f"  Missing: {', '.join(missing)}",
        f"  Running: {sys.executable}",
    ]

    if venv is not None:
        # Nearly always the real cause: the server was started with the system
        # interpreter while the dependencies live in the project venv. Saying
        # "pip install X" here would be actively harmful -- it would install
        # into the wrong environment and quietly abandon the venv for good.
        lines += [
            f"  Project venv: {venv}",
            "",
            "  That venv is probably the one you want -- you are not using it.",
            "  Start the server with it instead:",
            "",
            f"    {venv} scripts/serve.py --demo",
        ]
    else:
        lines += [
            "",
            "  Install them into this interpreter:",
            "",
            f"    {sys.executable} -m pip install -r requirements.txt",
        ]

    lines += [
        "",
        "  Refusing to start. Without these the server would come up fine and",
        "  then fail with a 500 the moment you pressed Register or Check --",
        "  after filling in the form and uploading the footage.",
        "",
    ]
    for line in lines:
        print(line)


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
    # Every skip already carries its own reason (see refuse_plaintext in
    # app/matching/gallery.py) -- nothing to add here. This used to claim
    # the imports "are stored unencrypted", true before SEC-06 and backwards
    # after it: a missing key means those people were skipped, not
    # silently written in the clear.
    return 0


def apply_demo_environment(demo: bool) -> None:
    """--demo means "showing the prototype, skip the friction" -- that already
    covered sign-in; it now covers the encryption key too.

    A real key set alongside --demo still wins: _fernet() checks for one
    before ever looking at ALLOW_PLAINTEXT_ENV, so this can only relax an
    unconfigured instance, never downgrade one that was actually set up.

    A function of its own rather than inline in serve() so it is something a
    test can call without also standing up uvicorn.
    """
    if demo and not os.environ.get(TEMPLATE_KEY_ENV):
        os.environ.setdefault(ALLOW_PLAINTEXT_ENV, "1")


def serve(args) -> int:
    settings = get_settings()
    setup_logging(settings.logging.level)

    apply_demo_environment(args.demo)

    # Before anything else: the pipeline's dependencies are imported lazily,
    # so this is the last point at which their absence can be reported as a
    # startup failure rather than as a 500 mid-registration.
    missing = missing_pipeline_requirements()
    if missing:
        report_missing_requirements(missing)
        return 1

    # Checked here, before the banner and the "Docs: ..." line, so a bind
    # failure is not buried under a screen of messages announcing success.
    # uvicorn only discovers this after its own startup has logged
    # "Application startup complete", which reads as though it worked.
    if not port_is_free(args.host, args.port):
        report_port_in_use(args.host, args.port)
        return 1

    if not os.environ.get(TEMPLATE_KEY_ENV) and not os.environ.get(ALLOW_PLAINTEXT_ENV):
        # This used to say enrolment "will be stored unencrypted" -- true
        # before SEC-06, backwards after it: enrolling now REFUSES outright.
        # A warning easy to miss in a scrolling terminal used to describe a
        # degraded mode; today it is hiding a hard failure the operator is
        # about to hit the moment they try to register anyone.
        for line in (
            "",
            f"  !! {TEMPLATE_KEY_ENV} IS NOT SET !!",
            "  Registering anyone will be refused, not stored unencrypted --",
            "  SEC-06 does not allow a silent plaintext fallback. Generate a",
            "  key and set it before you try to enrol anyone:",
            "",
            "    python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"",
            f"    export {TEMPLATE_KEY_ENV}=<that key>",
            "",
            "  Or run with --demo, which turns this off along with sign-in --",
            "  fine for showing the prototype, not for anyone real.",
            "",
        ):
            print(line)

    database_url = args.db_url or default_db_url(settings)
    settings.paths.data_dir.mkdir(parents=True, exist_ok=True)

    import uvicorn

    from app.api.main import create_app

    if args.demo:
        settings.demo_mode = True
        for line in (
            "",
            "  !! PROTOTYPE MODE (--demo) !!",
            "  Sign-in is off: anyone who can reach this port can read the",
            "  watchlist and confirm an identification, recorded against a",
            "  shared 'demo' operator that shows nothing about who really did",
            "  it. Registration photos and biometric templates are stored",
            "  unencrypted unless FRS_TEMPLATE_ENCRYPTION_KEY is set. Fine for",
            "  showing the prototype on your own machine; not for real data.",
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
            "For showing the prototype: skips sign-in, and skips the encryption "
            "key requirement (unless one is set). Anyone who can reach the port "
            "can take any action, and registration photos are stored in the "
            "clear. Not for real data."
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
