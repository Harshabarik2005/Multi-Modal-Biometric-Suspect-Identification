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

## Phase 2 — face branch ✅

**Delivered:** enroll a person from video, match them in other footage by
cosine similarity against ArcFace embeddings, with a pose-aware quality score
and encrypted template storage.

| File | Role |
|---|---|
| `app/core/types.py` | `Modality`, `ModalityEmbedding`, `TrackObservation` — the branch contract |
| `app/embeddings/base.py` | `EmbeddingBranch` / `PerFrameBranch` ABCs |
| `app/embeddings/face.py` | `FaceEmbedder` — ArcFace + quality scoring |
| `app/core/track_buffer.py` | Bounded per-track observation buffers |
| `app/matching/gallery.py` | Watchlist, open-set ranking, encrypted storage |
| `scripts/enroll.py` | Enrollment CLI (`--list`, `--inspect`) |
| `scripts/match.py` | Matching CLI |
| `scripts/make_face_fixtures.py` | Smoke-test fixtures |

### Decisions worth remembering

**Gait is not a per-frame signal, and the interface admits it.** Face and
re-ID embed one crop at a time; gait needs a sequence spanning a step cycle.
So every branch takes a *sequence* and returns one embedding: `PerFrameBranch`
implements that aggregation for face and re-ID, and Phase 3's gait branch will
implement `embed()` directly. Forcing gait through a per-frame interface would
have meant faking it, and a fake gait signal is worse than none — fusion would
weight it as real.

**"No signal" is not an error, and not zero.** `ModalityEmbedding.similarity()`
returns `None` when either side has nothing, never `0.0`. Collapsing those
would let a missing modality read as positive evidence of a mismatch. Face
absent (person turned away) is the *normal* case in this project.

**Quality is not detection confidence.** The detector will confidently locate
a face turned 67° away, from which ArcFace produces a confident and wrong
embedding. Quality multiplies three independent failure modes — detection
confidence × frontality × resolution — so any one of them failing drags the
score down. This is the signal Phase 6's attention head consumes; a branch
returning constant quality would silently disable the adaptive weighting the
whole project rests on.

**Enrollment is pickier than matching.** `embed_reference()` keeps only the
top 40% of usable frames by quality. Enrollment is offline with hundreds of
frames to choose from; averaging in profile and back-of-head frames would blur
the reference toward the population mean and cost accuracy on every subsequent
match. Live matching cannot afford to be that choosy.

**Track-level quality is the best frame, not the mean.** One unambiguous
frontal frame is enough to trust an identification; averaging would punish it
for the poor frames around it. Aggregation is quality-weighted for the same
reason — ten turned-away frames must not outvote one clear look by sheer
numbers.

**Buffers are bounded in three directions.** Per track (64 observations, most
recent kept), per crop (downscaled to 256px), and across tracks (50, LRU
evicted). The build plan's "per-track frame buffer" written naively is an
unbounded dict of crop lists — the fastest way to exhaust 4GB on a busy camera.

**Templates are encrypted at rest when a key is set.** `FRS_TEMPLATE_ENCRYPTION_KEY`
turns on Fernet encryption; without it enrollment still works but warns loudly
on every save. Verified: encrypted files carry a `FRSENC1:` header with no
numpy `PK` magic in the clear, and loading without the key fails closed while
the rest of the watchlist still loads.

### Verified on this machine

Enrolled one subject from a 40-frame clip (38 tracked frames, top 16 kept,
quality 0.499), then matched against a clip containing that subject plus a
second person, mirrored/dimmed/blurred so it is not a pixel-identical compare:

| Track | Cosine vs reference | Outcome |
|---|---|---|
| enrolled subject | **+0.947** | match at threshold 0.40 |
| impostor (quality gate bypassed) | **+0.045** | no match |

Separation of ~0.90. Independently, the quality gate rejected the impostor
outright at yaw −67° before any comparison happened — two separate safeguards,
tested separately, because a test that conflates them proves neither.

Full suite: 64 passed (60 fast + 4 slow).

### Known limits at this phase

- **The fixtures derive from one photograph.** They prove the machinery —
  crop → embed → store → load → cosine → threshold — and nothing about
  real-world accuracy. One pose, one lighting condition, no time gap. Real
  validation needs a person enrolled across angles then found in separate
  footage shot at a different time.
- **`track_buffer.store_height` trades memory against face resolution.** At
  256px, a 514px body crop halves, taking its face from ~137px to ~68px and
  roughly halving the quality score (resolution term 0.62 rather than 1.0).
  Raising it improves face quality at direct memory cost — 64 obs × 50 tracks
  is already ~370MB of crops at the current default. Tune against your camera.
- **Face runs on CPU.** The stock `onnxruntime` wheel has no CUDA provider, and
  `onnxruntime-gpu` needs CUDA/cuDNN matching the torch build. Not the
  bottleneck, and the code falls back with a warning rather than failing.
- **The 0.40 threshold is a placeholder.** It sits comfortably between 0.95 and
  0.05 on a trivially easy fixture. Phase 10's TAR@FAR curve is what should
  actually set it, on real footage.
- InsightFace emits a `FutureWarning` about `rcond` from its own
  `transform.py`. Upstream, harmless.

---

## Phase 3 — gait branch (next)

**Unresolved risk, and it should be settled before writing code:** the plan
assumes pretraining on CASIA-B, but that dataset requires a signed agreement
and its distribution has reportedly been unreliable. Whether OpenGait's
published GaitSet/GaitGL checkpoints are redistributable independently of the
dataset licence is the question to answer first. If pretrained gait weights
turn out not to be practically obtainable, the honest fallbacks are:

- Gait Energy Images plus a small encoder trained on your own footage.
- Skeleton-based gait from YOLOv8-pose keypoints, which may in any case be more
  robust than silhouettes at CCTV resolution.
- Treating gait as a low-weight modality initially and letting the attention
  head learn to discount it.

Whichever path, the branch implements `EmbeddingBranch.embed()` directly over a
sequence and returns clean "no signal" when it has fewer frames than one gait
cycle. Silhouettes can come from YOLOv8-seg (ultralytics is already installed)
or, for a fixed camera, classical background subtraction.
