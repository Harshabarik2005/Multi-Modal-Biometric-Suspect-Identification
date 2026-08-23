"""Tests for scripts/match.py, the CLI matching path.

The script is the one the README tells people to run, so its behaviour around
the review queue matters as much as the API's.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from app.core.types import Modality  # noqa: E402


@pytest.fixture
def repo():
    from app.core.types import ModalityEmbedding
    from app.db.repository import (
        WatchlistRepository,
        create_schema,
        make_engine,
        session_factory,
    )
    import numpy as np

    engine = make_engine("sqlite:///:memory:")
    create_schema(engine)
    session = session_factory(engine)()
    repository = WatchlistRepository(session)
    for person_id in ("ravi", "asha"):
        repository.enroll(
            person_id,
            person_id.title(),
            {
                Modality.FACE: ModalityEmbedding(
                    modality=Modality.FACE,
                    vector=np.ones(8, dtype=np.float32),
                    quality=0.9,
                    frames_used=5,
                )
            },
        )
    yield repository
    session.close()


def verdict(track_id: int, person_id: str | None, score: float, matched: int):
    from match import TrackVerdict

    return TrackVerdict(
        track_id=track_id,
        best_person_id=person_id,
        best_person_name=(person_id or "").title(),
        best_similarity=score,
        best_frame=track_id * 10,
        times_matched=matched,
        best_weights={Modality.FACE: 1.0},
        best_calibrated={Modality.FACE: score},
    )


class TestTheReviewQueueIsNotFlooded:
    """LOG-10: the CLI wrote a decision on every above-threshold re-match.

    A person walking across a camera clears the threshold in dozens of
    consecutive frames. One decision each buries the queue in the same sighting
    repeated, which makes the human-confirmation step -- the guardrail the
    whole design rests on -- something nobody can actually work through.
    """

    def test_one_decision_per_track_not_per_frame(self, repo) -> None:
        from match import record_verdicts

        # One track, matched in 40 separate frames.
        verdicts = {1: verdict(1, "ravi", 0.91, matched=40)}

        assert record_verdicts(repo, verdicts, "quality_weighted") == 1
        assert len(repo.pending_decisions()) == 1

    def test_each_track_still_gets_its_own(self, repo) -> None:
        from match import record_verdicts

        verdicts = {
            1: verdict(1, "ravi", 0.91, matched=40),
            2: verdict(2, "asha", 0.87, matched=12),
        }
        assert record_verdicts(repo, verdicts, "quality_weighted") == 2
        assert {d.person.person_id for d in repo.pending_decisions()} == {
            "ravi",
            "asha",
        }

    def test_a_track_that_never_matched_records_nothing(self, repo) -> None:
        from match import record_verdicts

        verdicts = {1: verdict(1, "ravi", 0.40, matched=0)}
        assert record_verdicts(repo, verdicts, "quality_weighted") == 0
        assert repo.pending_decisions() == []

    def test_the_recorded_decision_is_the_best_moment(self, repo) -> None:
        from match import record_verdicts

        entry = verdict(3, "ravi", 0.93, matched=20)
        record_verdicts(repo, {3: entry}, "quality_weighted", camera_id="cam-1")

        decision = repo.pending_decisions()[0]
        assert decision.frame_index == entry.best_frame
        assert decision.score == pytest.approx(0.93)
        assert decision.camera_id == "cam-1"

    def test_decisions_are_written_pending(self, repo) -> None:
        """The confirm step is the guardrail; the CLI must not skip it."""
        from app.db.models import DecisionStatus
        from match import record_verdicts

        record_verdicts(repo, {1: verdict(1, "ravi", 0.99, matched=5)}, "average")
        assert repo.pending_decisions()[0].status is DecisionStatus.PENDING
