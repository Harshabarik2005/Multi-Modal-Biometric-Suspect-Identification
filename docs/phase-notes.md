# Phase notes

Running log of what each phase actually delivered and what the next one needs.
Build-plan phase list lives in
[faceless-frs-build-plan.md](../faceless-frs-build-plan.md) section 6.

---

## Phase 0 — scaffold, env, config ✅

- Repo laid out per section 5 of the build plan.
- `backend/requirements.txt` — Phase 1 deps active, later phases listed but
  commented so the first install stays small.
- Config is layered: `backend/config.yaml` → env vars (`FRS_` prefix, `__`
  nesting) → CLI flags, highest wins. Loader in `backend/app/core/config.py`.
- `.gitignore` excludes `data/` wholesale. Enrollment footage and biometric
  templates are personal data and must not enter git history.
- Phase 2+ modules exist as documented stubs that raise `NotImplementedError`.
  Structure without pretend implementations.

## Phase 1 — detection + tracking ✅

**Delivered:** YOLOv8 person detection + DeepSORT tracking over a video source,
printing per-track frame counts.

Key files:

| File | Role |
|---|---|
| `app/core/types.py` | `Detection`, `Track`, `FrameResult` — the contract between stages |
| `app/core/video.py` | `VideoReader` — files, streams, camera indices; stride + frame cap |
| `app/detection/yolo_detector.py` | Ultralytics wrapper → `Detection` list |
| `app/tracking/deepsort_tracker.py` | `deep-sort-realtime` wrapper → `Track` list |
| `app/pipeline.py` | `DetectionTrackingPipeline` — the spine later phases plug into |
| `scripts/run_pipeline.py` | CLI |
| `scripts/make_test_video.py` | Smoke-test clip generator |

### Decisions worth remembering

**Boxes are `xyxy` in absolute source-frame pixels, everywhere.** DeepSORT
wants `ltwh`, so `Detection.ltwh` converts at the boundary. Every later branch
should take crops via `Track.crop(frame)` and never re-derive coordinates.

**Only confirmed tracks leave the tracker.** A detection must survive
`tracking.n_init` (default 3) consecutive frames before it gets an ID. This
keeps single-frame false positives out of the matching stage entirely — which
matters more here than in generic tracking, because a spurious track becomes a
spurious identification attempt against the watchlist.

**`min_box_height` (default 60px) drops tiny detections.** A 20-pixel-tall
person yields a useless face crop and a meaningless gait silhouette. Filtering
at the detector is cheaper than discovering it three branches later.

**DeepSORT's appearance embedder is for association only.** It is *not* the
re-ID embedding used for matching — that is OSNet, in Phase 4. Two different
jobs that both happen to be called "re-ID"; don't conflate them.

**`VideoReader` reports true source frame indices, not emitted counts.** With
`frame_stride=5` the second emitted frame is index 5, timestamp `5/fps`. Gait
cycle detection in Phase 3 depends on real timing, so this has to stay correct.

### Dependency traps hit while getting this running

**`setuptools<81` is a hard pin, not tidiness.** `deep-sort-realtime` 1.3.2
still imports `pkg_resources`, which setuptools removed in 81. On a fresh
install (pip pulls the newest setuptools) the tracker dies at import with
`ModuleNotFoundError: No module named 'pkg_resources'`. The pin is in
`requirements.txt` with that explanation. Drop it once upstream moves to
`importlib.metadata`.

**Ultralytics 8.4 renamed `half=True` to `quantize=16`.** The old name still
works but warns *once per inference call* — 120 frames, 120 warnings.
`YOLOPersonDetector._resolve_precision_kwargs` asks the installed build which
argument it knows, rather than pinning a version, since `requirements.txt`
allows both 8.3 and 8.4.

**Weights download to the working directory by default.** Ultralytics resolves
a bare name like `yolov8n.pt` by fetching it into the cwd, so it lands wherever
the CLI was run from. `_resolve_weights` pre-fetches into `paths.models_dir`
instead, falling back to Ultralytics' own resolution for custom checkpoint
names it does not recognise.

**On Windows, `torch` from PyPI is CPU-only.** Install from the CUDA index
first, then `requirements.txt`. Verified working: torch 2.5.1+cu121 on an
RTX 3050.

### Known limits at this phase

- Tracks break under long occlusion. Expected; multi-modal re-identification is
  precisely what fixes this, from Phase 4 on.
- `detect_batch` exists and truly batches, but `pipeline.stream` still calls
  `detect` per frame — batching needs a frame buffer, which is worth adding when
  throughput becomes the bottleneck, not before.
- The synthetic pan clip is a smoke test, not a benchmark. It proves wiring, not
  accuracy. It tiles one photo, so the "people" in it are duplicates and mirror
  images of the same few pedestrians -- fine for confirming detection and
  tracking run, useless for judging identification. Do not read anything into
  match numbers produced against it.

### Verified on this machine

120 frames of the synthetic clip, RTX 3050, fp16, `yolov8n.pt`: 686 person
detections, 11 confirmed tracks, the longest running 118 of 120 frames. Full
suite 27 passed (26 fast + 1 end-to-end under `--run-slow`).

---

## Phase 2 — face branch (next)

Goal: enroll one person, match them in a test video by cosine similarity. One
modality working end to end before adding the others.

Rough shape:

1. `pip install insightface onnxruntime-gpu` (uncomment in `requirements.txt`).
2. Implement `app/embeddings/face.py`: track crop → face detect/align → ArcFace
   → L2-normalised 512-d embedding, plus a per-frame quality score. That score
   is not optional garnish — Phase 6's attention head consumes it.
3. Enrollment script: 360° rotation video → pose-guided face crops → averaged
   reference embedding, written to `data/enrollment/<person_id>/`.
4. Matching script: run the Phase-1 pipeline, embed each track's face crops,
   compare against the reference by cosine similarity.
5. Sanity threshold: ArcFace cosine similarity above ~0.4 is a plausible
   starting point for same-person, but calibrate it on your own footage rather
   than trusting the number — Phase 10's TAR@FAR curve is what actually sets it.

Watch out for: no face visible in a crop (back turned) is the *normal* case
here, not an error path. The branch must return "no signal" cleanly and let
fusion lean on gait and re-ID instead. That is the entire premise of the
project.
