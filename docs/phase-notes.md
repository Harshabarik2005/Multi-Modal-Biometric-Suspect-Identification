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

## Phase 5 — baseline fusion ✅

**Delivered:** per-modality calibration, three fixed-rule fusion strategies,
and gait population centring, replacing the phase-4 fallback ladder.

| File | Role |
|---|---|
| `app/fusion/calibration.py` | `ModalityCalibration` — puts modalities on one scale |
| `app/fusion/baseline.py` | `SingleBestFusion`, `AverageFusion`, `QualityWeightedFusion` |
| `app/matching/gallery.py` | fusion in `rank()`, gait population centring |

### Raw similarities cannot be averaged

The measurements from phases 2–4 make this concrete:

| modality | different people | same person | gap |
|---|---|---|---|
| face | 0.03 | 0.95 | 0.92 |
| gait (centred) | 0.52 | 1.00 | 0.48 |
| re-ID | 0.755 | 0.980 | 0.225 |

Averaging these raw would let re-ID dominate every fused score purely by living
in a higher numeric range — a face score of 0.60, a strong identification,
would be dragged down by a re-ID score of 0.70, which is nothing at all.

`ModalityCalibration` maps each modality onto a common [0, 1] scale anchored on
its measured impostor and genuine points. Calibrated 0.0 means
"indistinguishable from a stranger", 1.0 means "as good as a genuine match
gets", for every modality alike. It is a linear rescale, **not** a probability —
a calibrated 0.7 does not mean a 70% chance of identity. Real probability
calibration needs labelled pairs, which is Phase 10.

The effect is visible on the probe clip: the impostor whose raw re-ID score was
0.771 — which would have matched at the phase-4 threshold of 0.75 — now
calibrates to **0.069** and is decisively rejected.

### The gait finding that changed the design

Measuring the gait descriptor for calibration anchors turned up something worse
than expected. On synthetic walkers, cosine similarity between *different*
walking styles averaged 0.962, against 1.000 for the same style — a separation
of **0.014**. Two consequences:

1. The phase-3 `gait_threshold: 0.80` was another false-positive bug of the
   same family as the re-ID one. Everything scores above 0.93, so it would have
   matched every person alive.
2. The cause is structural, not a tuning problem: **every GEI looks like a
   blurry human**, so cosine similarity is dominated by that shared shape.

The classical remedy — projecting out the population mean — transforms it:

| | same person | different (max) | separation |
|---|---|---|---|
| raw | 1.000 | 0.986 | +0.014 |
| mean-centred | 1.000 | 0.516 | **+0.484** |

A 34× improvement, by removing the "generic human" component and leaving what
actually distinguishes people.

Estimating that mean needs several references, so `Gallery.gait_population_mean`
returns `None` below `fusion.gait_min_references_for_centring` (default 3), and
gait comparison is then **refused outright** rather than falling back to raw.
That refusal is the important part: an uncentred gait similarity of 0.96 looks
like a confident match and is not one. Returning `None` routes it through the
"could not compare" path built in Phase 2, so fusion excludes it rather than
treating it as evidence.

### Decisions worth remembering

**Missing modalities are excluded, not zeroed.** A face that could not be seen
is not evidence against a match. Scoring it 0 would actively penalise the exact
situation this project exists to handle. Tested directly: a probe with no face
and a strong re-ID scores higher than one with a *stranger's* face and the same
re-ID.

**The baselines are implemented honestly.** A hobbled baseline makes the Phase-6
contribution look good and proves nothing, so each is the strongest version of
its idea. `QualityWeightedFusion` already does something the reference paper's
average-fusion baseline cannot — a clear frontal face outvotes a glancing one.

**`SingleBestFusion` prioritises by reliability, not by score.** Otherwise a
weak modality wins by being generous: re-ID at its ceiling calibrates to 1.0
while a mediocre face calibrates lower, and picking the larger number would
hand the decision to the least trustworthy signal.

### Verified on this machine

All three strategies on the probe clip (enrolled subject + impostor):

| strategy | enrolled subject | impostor |
|---|---|---|
| `single_best` | 1.000 (face) | 0.069 |
| `average` | 0.919 (face 0.50 + reid 0.50) | 0.069 |
| `quality_weighted` | 0.913 (face 0.46 + reid 0.54) | 0.069 |

Full suite: **148 passed**.

### Known limits at this phase

- **The strategies cannot yet be ranked.** One genuine and one impostor is not
  an evaluation. Which fusion rule is actually better needs TAR@FAR over many
  pairs — Phase 10. Do not read the table above as `single_best` winning.
- **`quality_weighted` gave re-ID more weight than face (0.54 vs 0.46)** on this
  clip, because the re-ID crop scored higher *quality* even though face is far
  more *discriminative*. That is the baseline's central blind spot: it weights
  by how good a look you got, not by how much that modality is worth. Learning
  that distinction is precisely Phase 6's job, and this is the concrete failure
  the attention head has to fix.
- The anchors are measurements, but from very few clips, and gait's come from
  synthetic walkers. Re-derive all six from real footage in Phase 10.
- One trust value covers a whole ranking pass, taken from the oldest enrolment.
  Per-person decay needs per-person timestamps threaded through ranking.

---

## Phase 6 — keyless attention fusion ⚠️ built, not deployable

**Delivered:** the attention head, its training harness, a synthetic training
demonstration, and a hard gate stopping an untrained head being used.

**Not delivered:** a head trained on real data, because there is none. This
phase is architecturally complete and honestly unusable, and the code says so
in both places it matters — `is_trained` is False until trained, and
`rank_attention` refuses outright.

| File | Role |
|---|---|
| `app/fusion/attention.py` | `KeylessAttentionFusion`, `triplet_loss` |
| `app/fusion/training.py` | triplet training, early stopping, synthetic data |
| `app/matching/gallery.py` | `rank_attention` — compares in the fused space |
| `scripts/train_fusion.py` | training CLI |

### What it does differently from Phase 5

The fixed rules fuse per-modality *similarities*. The attention head fuses
*embeddings* into one adaptive vector, and matching happens once in that shared
space. That is the paper's formulation, and it is why the head must be trained:
it learns a joint representation, not a weighted average of scores.

"Keyless" means there is no query. Each modality's projected embedding, plus
how good a look it got, is scored directly by a learned function, and those are
softmaxed into α (face), β (gait), γ (re-ID).

Three divergences from the reference paper, all forced by this project's shape:
three modalities rather than two; triplet loss on cosine distance rather than a
closed-set softmax, because the watchlist grows; and per-modality projections,
because face (512-d), re-ID (512-d) and gait (812-d) cannot be summed directly.

### It fixes the failure Phase 5 exposed

Phase 5 measured `QualityWeightedFusion` giving re-ID **more** weight than face
(0.54 vs 0.46), because fixed rules weight by *how good a look you got* rather
than *how much the modality is worth*. The trained head reverses this. On
synthetic data where face is far less noisy but often absent, and re-ID is
noisier but almost always present:

    face   0.591   (available 59%)
    reid   0.541   (available 97%)
    gait   0.248   (available 51%)

### Three bugs worth remembering

**The weight report was misleading, not the weights.** The first run appeared
to show the head preferring re-ID (0.495) over face (0.285) — the opposite of
what it should learn. It was averaging over *all* rows including those where
face was absent, so face's number was being dragged down by its 59%
availability rather than by any learned preference. Conditioned on presence,
face led all along. `TrainingReport` now reports both, and names the
present-only figure as the one to read.

**Initialisation was unseeded.** `train()` seeded torch, but the model was
constructed *before* that, so weight init came from ambient RNG state. The same
configuration measured both 1.78× and 4.37× overfitting purely because the
model was built at a different point in the program. Seeding now starts at
construction, and a test asserts reproducibility.

**The head does not reliably learn the weighting when under-resourced.** A
first test failed with face 0.514 against re-ID 0.581. Rather than hunt for a
passing seed, measured across six seeds:

| configuration | face > re-ID |
|---|---|
| 16-d modalities, 64-d shared, 800 triplets | **2/6** — a coin flip |
| 64-d modalities, 128-d shared, 1500 triplets | **6/6** — reliable |

Under-resourced, the head learns nothing useful about weighting. That is a real
deployment constraint, and the test now encodes the adequate configuration with
a slow multi-seed test guarding against regression.

### Refusing to run untrained

An untrained head is *worse* than the fixed rules it replaces: random
projections destroy Phase 5's careful calibration. So `fuse_one` and
`rank_attention` both raise rather than return noise, `train_fusion.py` refuses
to save a head whose train/validation separation ratio is ≥ 3.0, and
`rank_attention` rejects a dimension mismatch with an explicit message — which
is exactly what a synthetic-trained head hits against real 512-d embeddings.

### Verified on this machine

- Absent modalities take exactly zero weight and do not dilute the softmax.
- A row with no modality at all yields no NaN, which would otherwise poison
  the whole batch's gradients.
- Training reduces loss, early-stops, and restores the best weights.
- Save/load round-trips to identical outputs.
- `rank_attention` ranks correctly (0.788 vs −0.274) and refuses untrained
  heads and dimension mismatches.
- Full suite: **176 passed**.

### Known limits at this phase

- **The head is trained on synthetic data only, so it must not be used.** The
  synthetic identities are random vectors with tuned noise; real embeddings
  have structure this does not reproduce.
- **Overfitting persists** even with weight decay, early stopping and a
  validation split: roughly 2× train/validation separation at best. Synthetic
  identities are fresh random draws, so there is little transferable structure
  beyond the weighting policy itself.
- **It has never been compared against the Phase-5 baselines on real data.**
  Whether attention actually beats `quality_weighted` is the central claim of
  this project and is currently unproven. That comparison is Phase 10's
  ablation table.
- `rank_attention` returns `None` for per-modality similarities, since
  comparison happens only in the fused space. The dashboard's explainability
  view will need the attention weights instead, which are recorded.

---

## Phase 7 — backend API and database ✅

**Delivered:** SQLAlchemy schema, a database-backed watchlist, FastAPI routes,
and the human-confirm workflow made structural rather than conventional.

| File | Role |
|---|---|
| `app/db/models.py` | Person, Template, MatchDecision, DecisionReview, AuditEvent |
| `app/db/repository.py` | `WatchlistRepository` — the only code that writes |
| `app/api/routes.py` | watchlist CRUD, decisions, alerts, audit |
| `app/api/main.py` | app factory |
| `scripts/serve.py` | run the API; import phase-2 file enrollments |

### The guardrails are in the schema, not in a convention

**Nothing can confirm a match except a human.** `record_match` only ever writes
`status = PENDING`. The single path to CONFIRMED is `review()`, which requires
a non-blank operator name and accepts only CONFIRMED or REJECTED — the system
cannot mark its own decision resolved. `/alerts` returns confirmed decisions
only, so Phase 9's alerting physically cannot see an unreviewed match.

**The audit trail cannot be rewritten.** A review appends a `DecisionReview`
row; it never edits the decision. A reviewer changing their mind leaves both
judgements visible, because "the system said X and someone changed it to Y" is
exactly what a review needs to be able to see.

**Retiring is not deleting.** Match decisions reference the person, so removing
the row would make past decisions unreviewable. `retired_at` is set instead,
and retired people drop out of the gallery.

**Templates never cross the wire.** Enrollment takes video; the API describes
templates (modality, dim, quality, encrypted) but never returns a vector. An
API that hands out biometric vectors is a biometric-vector leak with extra
steps. Encryption reuses `GalleryStore`'s Fernet path so there is one
implementation rather than two that can drift.

### Two bugs found by the tests

**In-memory SQLite gave every connection its own empty database.** `create_schema`
built the tables on one connection, and the next session opened a blank one and
reported "no such table". Ten tests failed on it. `StaticPool` pins a single
connection for `:memory:` URLs.

**Re-enrollment hit a unique constraint.** Deleting the old templates and
inserting new ones in the same flush let SQLAlchemy order the INSERTs first,
tripping `(person, modality)`. Needed an explicit `flush()` between.

### The `.gitignore` gap

The database file was **not** ignored. It holds biometric templates and the
whole match audit trail, so a single `git add -A` would have committed the
watchlist. `*.db`, `*.sqlite`, and the SQLite journal/WAL sidecars are now
excluded. Nothing sensitive had been committed — checked — but the gap was
real and the earlier `data/` rules did not cover it.

### Verified on this machine

End-to-end through the API: imported the phase-2 file enrollment, recorded a
match with its fusion weights, confirmed `/alerts` was **empty** while the
decision was pending, reviewed it as a named operator, and watched it become
actionable and appear in `/alerts` — with the audit trail showing both the
enrolment and the review with their actors.

Full suite: **206 passed**.

### Known limits at this phase

- **No authentication.** `operator` is a free-text field, so the audit trail
  records a claimed name rather than a verified identity. That is honest for a
  local build but is not an access-control story; real deployment needs auth
  before the operator field means anything.
- **No enrollment-by-upload endpoint.** Enrolment still runs through
  `scripts/enroll.py` and is imported with `serve.py --import-enrollments`.
  Video upload plus background processing is a larger piece of work.
- **No live matching endpoint.** `scripts/match.py` writes to the console, not
  the database. Wiring the matcher to `record_match` is what makes the
  dashboard live, and it is the natural first task of Phase 8.
- SQLite by default. The schema is Postgres-compatible and `--db-url` takes any
  SQLAlchemy URL, but no migrations exist yet — `create_all` only.

---

## Phase 8 — frontend dashboard ✅

**Delivered:** a React review console — review queue with the explainability
view, watchlist, confirmed identifications, and the audit trail.

| File | Role |
|---|---|
| `frontend/src/App.jsx` | the console: queue, watchlist, alerts, audit |
| `frontend/src/api.js` | API client, modality metadata, appearance share |
| `frontend/src/styles.css` | dark console theme |
| `frontend/vite.config.js` | dev server, proxies `/api` to the backend |

Also: `scripts/match.py --record` now writes candidates to the database, which
is what gives the dashboard live data.

### The screen that matters is the review queue

Section 8 requires a human to confirm before anything follows from a match.
That is only meaningful if the human can tell a *good* match from a *plausible*
one, so each candidate shows:

- the fused score,
- a proportional bar of which modalities drove it,
- a table of per-modality weight, calibrated score, and what that modality
  actually measures ("Build and clothing", not "reid"),
- the fusion strategy used.

**And a caution when a match rests mostly on appearance.** Re-ID encodes
clothing more than the person, so when it carries over half the weight the card
says so explicitly rather than leaving the reviewer to infer it from a number.
On the live test data this fired immediately: the top candidate scored 0.909
with re-ID at 54% against face at 46% — exactly the Phase-5 weighting flaw,
now visible to the person being asked to confirm the identification.

The card also says "Nothing has happened yet — confirming is what makes this
actionable", because a queue of alarming-looking cards invites the assumption
that something already has.

### Decisions worth remembering

**The operator name is required and recorded, not implied.** It persists in
`localStorage` so a reviewer does not retype it, but it is still sent with every
verdict — the API rejects a blank one. An anonymous confirmation is not an
audit trail.

**No biometric data crosses into the browser.** The API describes templates but
never returns vectors, so there is nothing in the frontend that could leak one.
The watchlist view flags any template stored *unencrypted* with a warning
marker, so a misconfigured deployment is visible rather than silent.

**Vite proxies `/api` to the backend** rather than enabling CORS, which keeps
the browser on one origin and means there is no CORS policy to get wrong.

### Verified on this machine

Ran the API and dev server together against the real database, and reviewed a
candidate through the UI:

- the queue showed 3 pending candidates with their weight breakdowns,
- the appearance caution fired on the 54%-re-ID candidate,
- confirming as "Harsha" took pending from 3 to 2,
- the decision appeared in `/alerts` as actionable,
- the audit trail recorded `review by Harsha`.

Production build: 153 kB JS (49 kB gzipped). Full suite: **192 passed**.

### Known limits at this phase

- **No live video in the dashboard.** "Live monitoring" in the build plan means
  camera feeds; this shows decisions the matcher has already written. Streaming
  frames to the browser is a substantially larger piece of work.
- **No enrollment flow in the UI.** Enrolment still runs through
  `scripts/enroll.py` — the API has no upload endpoint yet, so there is nothing
  for a form to post to.
- **No authentication**, so the operator field records a claimed name rather
  than a verified identity. Same limitation as Phase 7, and it matters more
  here because this is the screen where the name gets typed.
- The queue polls every 10 seconds. Fine at this scale; a websocket would be
  better for a real feed.
- No frontend tests. The backend is well covered, the UI is not.

---

## Phase 9 — alerting ✅

**Delivered:** SMTP and Twilio transports, gated on human-confirmed decisions.

| File | Role |
|---|---|
| `app/alerts/notifier.py` | transports, rendering, `AlertDispatcher` |
| `scripts/send_alerts.py` | dispatch CLI, dry-run by default |

### The gate is checked twice

`AlertDispatcher` selects only confirmed decisions *and* re-checks each one
immediately before sending. Belt and braces, because this is the single place
in the system where a mistake leaves the building: everything else can be
corrected in the review console, but an alert cannot be unsent.

Tested directly: pending never alerts, rejected never alerts, and a
confirmation later reversed by a second reviewer stops alerting.

### Other decisions

**Alerts carry the reasoning.** A notification saying "match found, 0.91"
invites exactly the unexamined trust the review step exists to prevent, so
every alert lists which modalities drove the score, names the operator who
confirmed it, and warns when it rested mostly on clothing.

**Nothing is sent twice.** Delivery writes an audit event; a re-run skips
anything already alerted. A duplicate alert reads as a second sighting.

**A failed delivery is not marked sent**, so a transient outage does not
silently swallow an alert — the next run retries it.

**Dry run is the default**, and `--send` is required on top of configuration.
Accidentally messaging a real contact list during testing is not recoverable.

---

## Phase 10 — evaluation harness ✅

**Delivered:** TAR@FAR, ROC-AUC, EER, CMC/Rank-N, the ablation table, and a
demographic fairness breakdown.

| File | Role |
|---|---|
| `eval/metrics.py` | the open-set metrics |
| `eval/ablation.py` | Table I equivalent, with coverage handling |
| `eval/run_evaluation.py` | CLI |

### Why not accuracy

In an open-set watchlist almost everyone passing a camera is on nobody's list,
so a system that matched *nobody, ever* would score superbly on accuracy while
being useless. The question is "at a false-alarm rate we can live with, what
fraction of the people we are looking for do we find" — TAR@FAR.

### The methodological bug the harness caught in itself

The first ablation run looked damning: face-only scored **0.978 AUC** and
fused scored **0.697**. Read naively, fusion makes things worse.

It does not. Face-only was scored on **6,903** pairs and fusion on **18,949**.
A single-modality row can only score pairs where that modality is present on
*both* sides, so "face only" was being graded on precisely the easy cases —
the ones where a face was visible. The two numbers describe different
populations and are not comparable.

The harness now reports two tables:

* **Native coverage** — each row on the pairs it can score, with a `cover`
  column, plus an automatic warning when coverage differs enough that the AUCs
  are not comparable.
* **Common subset** — every row restricted to pairs where all three modalities
  are present. The only apples-to-apples AUC comparison, and an
  unrepresentatively easy population, since a hidden face is the norm.

### What that reveals about fusion

On the common subset, face alone still beats fusion (0.949 vs 0.811). Fusion
does not make a clear face better. What it buys is **coverage**: it produces a
usable score on **97%** of pairs against face's **33%**.

In deployment the alternative to a fused score on a turned-away person is not a
better score — it is *no score at all*. That is the entire premise of the
project, and it is now a measured statement rather than an assertion. The CLI
prints this comparison automatically.

### Other decisions

**Missing modalities are skipped, not zeroed**, in the single-modality rows.
Scoring them zero would measure availability rather than discriminative power
and make face — the strongest signal — look like the worst.

**Calibration anchors are derived from the data being evaluated**, not taken
from config. The configured anchors were measured on real footage; applying
them to synthetic scores would measure the mismatch rather than the fusion.

**Small samples are flagged.** With 18 genuine pairs, TAR resolves in steps of
5.6% and differences between rows are noise. The CLI says so rather than
printing four decimal places of nothing.

**Fairness groups are supplied, never inferred.** This code does not attempt to
derive demographic attributes from biometric data. Per-group TAR@FAR at a
shared threshold is what makes a gap visible; an aggregate number hides it.

---

## Phase 11 — extensions ✅

**Delivered:** disguise augmentation, the per-stage benchmark, and the
observation cap that came out of it. Time-aware re-ID trust decay landed in
Phase 4 (`trust_at`).

| File | Role |
|---|---|
| `app/embeddings/disguise.py` | synthetic masks, sunglasses, hoods, blur |
| `scripts/test_disguise.py` | measures the face branch under occlusion |
| `scripts/benchmark.py` | per-stage timing |

### The face branch is occlusion-aware, and now it is measured

The build plan claimed an "occlusion-aware face branch" that had never been
tested. Measured on the enrollment fixture (baseline quality 0.480):

| disguise | face covered | similarity | quality |
|---|---|---|---|
| mask | 36% | 0.741 | 0.340 |
| sunglasses | 16% | 0.797 | 0.430 |
| hood | 47% | 0.831 | 0.450 |
| **mask + sunglasses** | 52% | **0.254** | **0.233** |
| blur | — | 0.764 | 0.425 |

Similarity and quality fall **together**. No disguise produced the dangerous
case — an embedding far from the reference while still reporting good quality —
which is what fusion needs, since it weights by quality and would trust such an
embedding.

Note the practical reading: a masked face still matches at 0.741, comfortably
over threshold. A face that is both masked and wearing sunglasses drops to
0.254 and correctly does *not* match — which is exactly the situation where
gait and re-ID have to carry the identification.

**The first run of this test was broken and looked like a triumph.** It
reported similarity 1.000 for a masked face — apparently perfect invariance. In
fact the disguises are proportioned to a *face* and were being applied to a
*body* crop, so the "mask" landed around the knees. `apply_to_region` now
places them inside the detected face box.

### The benchmark found the real bottleneck

Per-stage timing over 40 frames on the RTX 3050:

- **face embedding: 70% of total runtime, 5.3 s per track**
- detection + tracking together: 15.3 fps, not the problem

It was embedding all 64 buffered crops on CPU. `TrackBuffer.best()` had existed
since Phase 2 and was never wired up. Capping the live path at the largest 16
crops took total runtime from **60.4 s to 29.9 s**.

**That cap immediately introduced a second bug**, which the benchmark also
caught: gait dropped to 0.00 s — producing nothing at all. Two reasons, both
instructive. `best()` sorts by box height, destroying the temporal order
cadence detection depends on; and 16 is below `gait.min_frames` of 20, so gait
could never fire. The cap now applies only to the per-frame branches (face,
re-ID) while gait receives the full ordered sequence. That distinction is the
Phase-2 per-frame-versus-sequence split showing up again, this time as a
performance bug.

### Edge deployment

Documented rather than implemented: TensorRT export needs the target device to
build an engine, so it cannot be done meaningfully from a desktop. The
benchmark's module docstring lists the levers in the order worth trying —
raise `frame_stride`, raise `rematch_every`, drop to smaller backbones, and cut
gait first since it costs a second segmentation pass and is the weakest signal.

---

## Where the project stands

Every phase in the build plan is implemented. **273 tests pass.**

The one thing that is not done, and cannot be done here, is validation on real
footage. It blocks the same three things throughout:

1. **Gait has no positive validation.** The negative direction is tested on
   real video; the positive rests entirely on synthetic silhouettes.
2. **The attention head cannot be honestly trained**, so Phase 6 refuses to
   run. It learns the right weighting on synthetic data and that is all that
   can be said.
3. **The central claim is unproven.** Whether attention fusion beats
   `quality_weighted` has never been tested on real data. The ablation harness
   that would settle it is built, tested, and waiting for input.

Every threshold in the system is a placeholder measured on one or two clips.
`eval/run_evaluation.py` is what should set them, from a TAR@FAR curve on
footage of real people.

What real footage means concretely: several people, each recorded more than
once, ideally at different times and from different cameras, with consent and a
lawful basis. Run them through the pipeline, collect per-modality embeddings
per sighting, and build `eval.ablation.Observation` records. Everything
downstream of that point already works.
