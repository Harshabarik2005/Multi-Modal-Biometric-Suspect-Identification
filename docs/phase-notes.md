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

## Phase 3 — gait branch ✅

**Delivered:** silhouette extraction, gait cycle detection, Gait Energy Images,
and a pluggable encoder, wired into enrollment and matching as a fallback for
when the face branch has nothing.

| File | Role |
|---|---|
| `app/embeddings/silhouette.py` | YOLOv8-seg → normalised 64×44 silhouettes |
| `app/embeddings/gait.py` | cadence detection, GEI, `GaitEmbedder` |

### The licence finding, and what it forced

The plan called for GaitSet/GaitGL pretrained on CASIA-B. Checked directly:

- CASIA-B **pretrained weights are downloadable** from OpenGait's GitHub
  releases (`pretrained_casiab_gaitbase.zip`, 54MB, HTTP 200). No dataset
  agreement is needed for the weights themselves.
- But **OpenGait ships no LICENSE file at all** — verified against the repo
  root and the GitHub licence API, both empty. With no licence declared, the
  code is all-rights-reserved by default, and the weights are useless without
  vendoring its model definitions.

So the default encoder is a **classical GEI descriptor, not a learned
embedding**: pooled GEI plus row/column projection profiles, L2 normalised.
GEI-based recognition genuinely works and predates deep gait models, but it is
markedly weaker than a trained encoder and much weaker than ArcFace. Stated
plainly rather than buried, because a modality that over-claims its own
reliability corrupts the Phase-6 attention weights that decide how far to
trust it.

Preprocessing deliberately produces exactly the 64×44 input a learned gait
model expects, so swapping one in later is an encoder change and nothing else.

### The bug that mattered: gait reported for people standing still

First run against real footage, 3 of 4 tracks produced confident gait
embeddings — one at quality 0.66. **Nobody in that clip is walking**; it is a
panned photograph. The "gait" was segmentation jitter, and autocorrelation
latched onto it happily.

That is the most dangerous possible failure for this branch. A false gait
signal is not a missed detection — it is fabricated evidence that fusion would
weight as real. Three independent gates now stand between a signal and a gait
embedding, because each is individually foolable:

| Gate | Catches | Foolable alone by |
|---|---|---|
| **Periodicity** ≥ 0.55 | signals that do not repeat | a low-amplitude repeating wobble |
| **Swing ratio** ≥ 0.12 | legs that barely move | one big lurch |
| **Area stability** ≤ 0.10 | segmentation failure | — |

The third is the one that actually settled it, and it rests on a physical
invariant rather than a tuned number: **a human body does not change size while
walking**. Legs redistribute silhouette pixels, they do not add or remove them.
Measured coefficient of variation in silhouette area:

- real walking: **0.033–0.045** across every stride and cadence tested
- the false-positive track: **0.159**

Threshold 0.10 sits with roughly 2× headroom on both sides. Result on the
stationary-people clip: **0 of 9 tracks** report gait, down from 3 of 4, while
every synthetic walker still passes all three gates.

`min_half_period` also went from 5 to 7 frames — at 25fps a half gait cycle is
10–15 frames, and short lags are precisely where jitter manufactures fake
periodicity.

### Other decisions worth remembering

**Silhouettes are centred on centre of mass, not bounding box.** An
outstretched arm or a swinging bag shifts the box but barely moves the mass.
Centring on the box makes the whole body jitter sideways frame to frame —
which is exactly the signal gait recognition is trying to read.

**The GEI is truncated to whole gait cycles.** Averaged over a partial cycle it
is biased toward whichever leg happened to be forward when the clip ended, so
the same person recorded twice would not match themselves. Tested directly: 40
frames and 47 frames of the same walk produce GEIs identical to 1e-6.

**Gait has its own match threshold (0.80 vs face's 0.40).** A gait score of
0.85 is a far weaker claim than a face score of 0.85; one shared number would
quietly equate them. The match report also prints which modality produced each
score, so a gait-driven match can never be read as a face-driven one.

**Matching falls back to gait only when face has no signal.** That is the case
the project exists for — turned away, too distant, masked.

### Verified on this machine

- 0 of 9 stationary tracks report gait (was 3 of 4 before the gates).
- Synthetic walkers pass at strides 0.4–1.8 and cadences 7–14 frames.
- Cadence recovered exactly for known periods of 7, 10 and 14 frames.
- Face matching unchanged: enrolled subject +0.953 `via face`.
- Full suite: **95 passed** (89 fast + 6 slow).

### Known limits at this phase

- **No positive validation on real walking footage.** The negative direction is
  tested against real video; the positive direction rests on synthetic
  silhouettes. Nothing here has seen an actual person walk. That is the single
  biggest gap in the gait branch and only real footage closes it.
- The classical descriptor is weak compared to a learned encoder. Expect gait
  to earn a low attention weight in Phase 6 — which is the correct outcome, not
  a failure.
- Segmentation runs per buffered crop, so gait costs a second model pass. Fine
  at current volumes; batching would help if throughput becomes a problem.
- `area_stability` assumes the tracker keeps a consistent box. Heavy occlusion
  will trip it, correctly reporting no gait rather than a corrupted one.

---

## Phase 4 — re-ID branch ✅

**Delivered:** OSNet whole-body appearance embeddings, wired into enrollment
and matching, with time-aware trust decay.

| File | Role |
|---|---|
| `app/embeddings/reid.py` | `ReIDEmbedder`, batched inference, `trust_at()` |
| `app/embeddings/vendor/osnet.py` | OSNet model definition, vendored (MIT) |

### The threshold that would have caused false matches

I set `reid_threshold: 0.75` by analogy with the face threshold. Then measured
it on real crops:

| Comparison | Cosine |
|---|---|
| Same person (two halves of one track) | **0.980** |
| Different people | **0.755** |

The default sat *below* the different-people score. Re-ID similarities cluster
in a high, narrow band — a 0.23 gap between same and different, against face's
0.90 gap — so face thresholds simply do not transfer. Raised to **0.88**,
leaning conservative because a false identification is worse than a missed one.

Confirmed on the probe clip: the impostor track now falls through to re-ID and
scores 0.771 against the enrolled subject, correctly rejected. At 0.75 it would
have matched a stranger.

The same-person figure comes from one continuous track — identical clothing,
lighting, seconds apart — which is far easier than a real cross-camera match.
Phase 10's TAR@FAR curve on real footage is what should actually set this.

### Why OSNet is vendored rather than depended on

`pip install torchreid` succeeds and it is **MIT licensed**, so there is no
legal obstacle — unlike gait. The problem is purely structural: its package
`__init__` imports the entire training stack to reach a model definition,
pulling in `gdown` and then `tensorboard` as undeclared import-time
dependencies, each failing in turn.

`osnet.py` is 598 lines importing nothing beyond `torch`. Copying it (with the
MIT licence and provenance in `vendor/README.md`) removes the whole fragile
chain. The file is unmodified so it can be diffed against upstream. Weights are
not vendored; they download on first use.

Weight availability was verified before any of this was built: the Google Drive
download still works, 10.9MB, 567 valid state-dict keys.

### Decisions worth remembering

**This is not the DeepSORT embedder.** DeepSORT already runs an appearance
model to associate boxes between adjacent frames. That answers "is this the
same blob as last frame"; OSNet answers "is this the person on the watchlist".
Both get called re-ID, and conflating them is an easy mistake.

**Re-ID goes stale, so trust decays.** Face and gait describe a person; re-ID
largely describes their clothing. A match across two hours is strong evidence,
the same score across two weeks probably means a common jacket. `trust_at()`
applies exponential decay with a 3-day half-life. This is Phase 11's
"time-aware trust" brought forward, because the alternative is a system that
confidently misidentifies people by their coat.

**Zero detection confidence is neutral, not disqualifying.** Tracker-predicted
boxes report 0.0 confidence. Treating that as evidence of a bad crop would
throw away usable frames, so it maps to 0.5.

**Quality ignores pose, unlike face.** A back view is perfectly usable for
re-ID — that is the point of the modality. It scores on resolution, box aspect
ratio (a box far from human proportions means a partial or merged body), and
detection confidence.

### Verified on this machine

- Weights download and load: 567 keys, 2.68M params, 512-d unit vectors.
- Same person 0.980, different people 0.755, impostor correctly rejected at
  the 0.88 threshold.
- `embed_batch` agrees with per-frame results to 1e-3.
- Trust decay: day 0 → 1.00, day 3 → 0.50, day 7 → 0.20, day 30 → 0.001.
- Full suite: **116 passed**.

### Known limits at this phase

- **Cross-camera and cross-day re-ID is untested.** Every measurement here
  comes from a single clip. Re-ID is precisely the modality that degrades
  across cameras and days, so these numbers are its best case, not its typical
  one.
- The matching ladder (face → gait → re-ID) takes the strongest *available*
  signal rather than combining them. That is the interim behaviour Phase 5
  replaces with real fusion.
- `trust_at()` exists and is tested but is not yet applied in matching — it has
  nothing to weight until fusion lands in Phase 5.

---

## Phase 5 — baseline fusion (next)

Simple rule-based and average fusion over the three modality embeddings,
mirroring the "average fusion" row of the reference paper's Table I.
Deliberately dumb: fixed weights, no learning. Its whole job is to be the
number Phase 6's attention fusion has to beat, so it must be implemented
honestly rather than hobbled.

Two things it needs that already exist: `ModalityEmbedding.quality` per branch,
and `trust_at()` for re-ID staleness. The obvious baselines to implement are
equal-weight averaging, quality-weighted averaging, and max-confidence
selection — reporting all three gives Phase 6 a real bar to clear.

The per-modality similarity scales measured so far (face: 0.03 different /
0.95 same; re-ID: 0.755 / 0.980) mean raw similarities **cannot** be averaged
directly — they must be calibrated onto a comparable scale first, or re-ID will
dominate every fused score purely by living in a higher numeric range. That is
the main trap in this phase.
