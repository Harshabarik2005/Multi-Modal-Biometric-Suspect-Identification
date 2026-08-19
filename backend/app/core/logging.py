"""Console logging setup.

Uses `rich` when it is installed so long pipeline runs stay readable, and
falls back to the stdlib formatter otherwise.
"""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger once. Safe to call repeatedly."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level.upper())
        return

    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(
            rich_tracebacks=True, show_path=False, markup=False
        )
        fmt = "%(message)s"
    except ImportError:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

    logging.basicConfig(level=level.upper(), format=fmt, handlers=[handler])

    # Ultralytics logs a banner per inference call at INFO; quiet it down.
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
