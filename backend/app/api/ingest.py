"""Upload endpoints: enrol a person, then scan footage for them.

Two routes do the work the CLI scripts used to:

* `POST /api/enroll` -- photos and/or video of one person, plus their details.
* `POST /api/scan`   -- CCTV footage to search for anyone on the watchlist.

Both return a job id immediately and run in the background, because each takes
minutes. `GET /api/jobs/{id}` reports progress and, when finished, the result.

`POST /api/enroll/preview` runs the cheap half only -- it finds the person and
reports which signals the upload can support -- so someone can be told they
need a walking video *before* they wait through a full enrolment that stores a
face-only profile.

Uploaded files are written to a temporary directory and deleted as soon as the
job finishes. Enrolment footage is personal data; keeping it around after the
embeddings are extracted creates a second copy to protect for no benefit.
"""

from __future__ import annotations

import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.jobs import Job, JobRunner, ProgressReporter
from app.api.media import (
    assess_readiness,
    gait_observations,
    kind_of,
    observations_from_uploads,
)
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.types import Modality
from app.db.repository import WatchlistRepository

logger = get_logger(__name__)
router = APIRouter()

# Guard against a mis-click uploading a 10GB file and filling the disk.
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_FILES = 60

#: Imported rather than redefined, so the API and the file-backed gallery
#: cannot drift apart on what counts as a safe id.
from app.matching.gallery import PERSON_ID_PATTERN  # noqa: E402


def get_session() -> Session:  # pragma: no cover - replaced by the app factory
    raise RuntimeError("Session dependency was not configured.")


def get_runner() -> JobRunner:  # pragma: no cover - replaced by the app factory
    raise RuntimeError("Job runner dependency was not configured.")


def get_app_settings() -> Settings:
    return get_settings()


Runner = Annotated[JobRunner, Depends(get_runner)]
AppSettings = Annotated[Settings, Depends(get_app_settings)]


# -- schemas ---------------------------------------------------------------

class ReadinessOut(BaseModel):
    modality: str
    ready: bool
    reason: str
    requirement: str


class PreviewOut(BaseModel):
    images: int
    videos: int
    observations: int
    tracks_found: int
    rejected: list[dict[str, str]] = Field(default_factory=list)
    readiness: list[ReadinessOut] = Field(default_factory=list)
    #: Set when several people were tracked in the footage.
    warning: str = ""


class JobOut(BaseModel):
    id: str
    kind: str
    status: str
    progress: float
    message: str
    result: dict | None = None
    error: str = ""
    created_at: str = ""
    finished_at: str = ""


# -- upload handling -------------------------------------------------------

async def _stage(files: list[UploadFile]) -> tuple[Path, list[Path]]:
    """Write uploads to a temp directory. Caller must delete it."""
    if not files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No files were uploaded.")
    if len(files) > MAX_FILES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Too many files ({len(files)}); the limit is {MAX_FILES}.",
        )

    directory = Path(tempfile.mkdtemp(prefix="frs-upload-"))
    saved: list[Path] = []
    total = 0

    try:
        for upload in files:
            name = Path(upload.filename or "upload").name
            if kind_of(Path(name)) == "unknown":
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    f"{name}: unsupported file type. Upload images "
                    "(jpg/png) or video (mp4/mov/avi/mkv).",
                )

            destination = directory / name
            with destination.open("wb") as handle:
                while chunk := await upload.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.",
                        )
                    handle.write(chunk)
            saved.append(destination)
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise

    return directory, saved


# -- preview ---------------------------------------------------------------

@router.post("/enroll/preview", response_model=PreviewOut, tags=["enrollment"])
async def preview_enrollment(
    settings: AppSettings,
    files: list[UploadFile] = File(...),
) -> PreviewOut:
    """Check what an upload can support, without enrolling anyone.

    Runs detection only -- no embedding models -- so it comes back in seconds
    and can tell someone they need a walking video while they are still at the
    computer.
    """
    directory, paths = await _stage(files)
    try:
        observations, summary = observations_from_uploads(paths, settings)
        readiness = assess_readiness(summary, settings)

        warning = ""
        if summary.tracks_found > 1:
            warning = (
                f"{summary.tracks_found} people were tracked in this footage. "
                "Enrolment uses the one on screen longest. If that is not your "
                "subject, re-record with only them in frame -- a reference "
                "blended from several people produces confident wrong matches."
            )

        return PreviewOut(
            images=summary.images,
            videos=summary.videos,
            observations=summary.observations,
            tracks_found=summary.tracks_found,
            rejected=[{"file": name, "reason": why} for name, why in summary.rejected],
            readiness=[
                ReadinessOut(
                    modality=check.modality.value,
                    ready=check.ready,
                    reason=check.reason,
                    requirement=check.requirement,
                )
                for check in readiness
            ],
            warning=warning,
        )
    finally:
        shutil.rmtree(directory, ignore_errors=True)


# -- enrollment ------------------------------------------------------------

@router.post("/enroll", response_model=JobOut, tags=["enrollment"])
async def enroll(
    runner: Runner,
    settings: AppSettings,
    # Constrained to a safe filename charset, not just a length. The
    # file-backed gallery uses person_id as a directory name, so "../.." or an
    # absolute path escapes the enrolment root entirely.
    person_id: str = Form(..., pattern=PERSON_ID_PATTERN),
    display_name: str = Form(..., min_length=1, max_length=200),
    notes: str = Form("", max_length=2000),
    operator: str = Form("unknown", max_length=120),
    replace: bool = Form(False),
    files: list[UploadFile] = File(...),
) -> JobOut:
    """Enrol a person from uploaded photos and/or video."""
    directory, paths = await _stage(files)

    from app.api.main import session_maker_for_app  # set by the app factory

    def work(job: Job) -> dict:
        reporter = ProgressReporter(
            job, stages=["locating person", "face", "gait", "appearance", "storing"]
        )
        session = session_maker_for_app()
        try:
            reporter.stage("locating the person in your uploads")
            observations, summary = observations_from_uploads(paths, settings)
            if not observations:
                raise ValueError(
                    "No person could be found in any uploaded file. Check the "
                    "whole body is visible and the images are not too small."
                )

            embeddings = {}
            stored: list[str] = []
            skipped: list[dict[str, str]] = []

            reporter.stage("reading the face")
            from app.embeddings.face import FaceEmbedder

            face = FaceEmbedder(settings).embed_reference(observations)
            if face.has_signal:
                embeddings[Modality.FACE] = face
                stored.append("face")
            else:
                skipped.append({
                    "modality": "face",
                    "reason": "no usable face found -- is it visible and frontal?",
                })

            reporter.stage("reading gait")
            # Only the contiguous run from one video. Passing the whole list
            # would have gait read a cadence across the join between separate
            # recordings, and across photographs that have no cadence at all.
            gait_frames = gait_observations(observations, summary)
            if summary.has_motion_source and gait_frames:
                from app.embeddings.gait import GaitEmbedder

                gait = GaitEmbedder(settings).embed_reference(gait_frames)
                if gait.has_signal:
                    embeddings[Modality.GAIT] = gait
                    stored.append("gait")
                else:
                    skipped.append({
                        "modality": "gait",
                        "reason": (
                            "no gait signal -- the subject has to be WALKING "
                            "through several full step cycles"
                        ),
                    })
            else:
                skipped.append({
                    "modality": "gait",
                    "reason": "photos only -- a still image contains no gait",
                })

            reporter.stage("reading appearance")
            from app.embeddings.reid import ReIDEmbedder

            reid = ReIDEmbedder(settings).embed_reference(observations)
            if reid.has_signal:
                embeddings[Modality.REID] = reid
                stored.append("reid")
            else:
                skipped.append({
                    "modality": "reid",
                    "reason": "no usable body crop -- too small, or odd box shape",
                })

            if not embeddings:
                raise ValueError(
                    "Nothing could be enrolled: no signal from any of the three "
                    "branches. The person may be too small or too blurred."
                )

            reporter.stage("storing the profile")
            repo = WatchlistRepository(session)
            repo.enroll(
                person_id,
                display_name,
                embeddings,
                notes=notes,
                source=f"{summary.videos} video(s), {summary.images} image(s)",
                actor=operator,
                replace=replace,
            )

            return {
                "person_id": person_id,
                "display_name": display_name,
                "stored": stored,
                "skipped": skipped,
                "observations": len(observations),
                "images": summary.images,
                "videos": summary.videos,
            }
        finally:
            session.close()
            # Enrolment footage is personal data. The embeddings are extracted;
            # keeping the source creates a second copy to protect for nothing.
            shutil.rmtree(directory, ignore_errors=True)

    return JobOut(**runner.submit("enroll", work).to_dict())


# -- scanning --------------------------------------------------------------

@router.post("/scan", response_model=JobOut, tags=["scanning"])
async def scan(
    runner: Runner,
    settings: AppSettings,
    camera_id: str = Form("", max_length=64),
    # Bounded. Unbounded, a single request could set this to 0 and make every
    # subsequent scan match everyone.
    threshold: float = Form(-1.0, ge=-1.0, le=1.0),
    record: bool = Form(True),
    files: list[UploadFile] = File(...),
) -> JobOut:
    """Search uploaded footage for anyone on the watchlist.

    Every hit is recorded as a PENDING decision. Nothing is confirmed here --
    that still requires a person in the review console.
    """
    directory, paths = await _stage(files)
    videos = [p for p in paths if kind_of(p) == "video"]
    if not videos:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Scanning needs video. Upload CCTV footage, not stills.",
        )

    from app.api.main import session_maker_for_app

    def work(job: Job) -> dict:
        session = session_maker_for_app()
        try:
            repo = WatchlistRepository(session)
            gallery = repo.load_gallery()
            if len(gallery) == 0:
                raise ValueError(
                    "The watchlist is empty -- enrol someone before scanning."
                )

            # Work on a private copy. `get_settings()` is lru_cached and hands
            # every caller the SAME mutable object, so assigning to
            # `settings.fusion.threshold` here rewrote the match threshold for
            # the entire process and never restored it -- one request with a
            # low threshold made every later scan match everyone.
            job_settings = settings.model_copy(deep=True)
            if threshold >= 0:
                job_settings.fusion.threshold = threshold

            job.message = f"scanning against {len(gallery)} enrolled"
            from app.api.scanning import scan_video

            findings = []
            for index, video in enumerate(videos):
                findings.extend(
                    scan_video(
                        video,
                        gallery,
                        repo if record else None,
                        job_settings,
                        camera_id=camera_id,
                        job=job,
                        video_index=index,
                        video_count=len(videos),
                    )
                )

            job.message = (
                f"found {len(findings)} candidate(s)" if findings else "no matches"
            )
            return {
                "camera_id": camera_id,
                "videos": [v.name for v in videos],
                "threshold": job_settings.fusion.threshold,
                "recorded": record,
                "findings": findings,
                "scanned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        finally:
            session.close()
            shutil.rmtree(directory, ignore_errors=True)

    return JobOut(**runner.submit("scan", work).to_dict())


# -- job status ------------------------------------------------------------

@router.get("/jobs/{job_id}", response_model=JobOut, tags=["jobs"])
def job_status(job_id: str, runner: Runner) -> JobOut:
    job = runner.get(job_id)
    if job is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "No such job. Jobs are held in memory and do not survive a restart.",
        )
    return JobOut(**job.to_dict())


@router.get("/jobs", response_model=list[JobOut], tags=["jobs"])
def recent_jobs(runner: Runner, kind: str | None = None, limit: int = 25) -> list[JobOut]:
    """Recent jobs, WITHOUT their results.

    The result payload of a scan lists everyone it found -- names, cameras,
    timestamps, scores. Returning that from a listing route hands the whole
    recent history to anyone who asks. Fetch a specific job by id to see its
    result; the id is only known to whoever started it.
    """
    listed = []
    for job in runner.recent(limit=limit, kind=kind):
        summary = job.to_dict()
        summary["result"] = None
        # Exception text carries temp paths, model paths and database messages.
        summary["error"] = "failed" if summary["error"] else ""
        listed.append(JobOut(**summary))
    return listed
