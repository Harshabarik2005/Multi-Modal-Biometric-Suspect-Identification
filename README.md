# Faceless FRS

Multi-modal biometric suspect identification — identifies a person from CCTV
footage even when their face is unclear, hidden, or turned away, by fusing
three signals instead of relying on face alone:

- **Face** — ArcFace embeddings
- **Gait** — how the person walks
- **Re-ID** — general appearance (build, clothing)

Built as an extension of Prakash et al., *"Multimodal Adaptive Fusion of Face
and Gait Features using Keyless Attention based Deep Neural Networks for Human
Identification"* (2023, [arXiv:2303.13814](https://arxiv.org/abs/2303.13814)).
The full spec, including where this project deliberately diverges from the
paper, is in [faceless-frs-build-plan.md](faceless-frs-build-plan.md).

> **Status: all 11 build phases implemented · 499 tests.**
>
> Face and appearance produce identifications on real test footage, through the
> full browser workflow from enrolment to human review. **Gait is computed and
> shown on every result but does not yet count toward scores** — its scoring
> scale was only ever calibrated on synthetic walkers (see
> [Getting gait to work](#getting-gait-to-work)).
>
> What is *not* done is validation at scale: whether attention fusion beats the
> fixed-rule baselines has never been tested on real data, and every threshold
> is measured on a handful of clips. The evaluation harness that would settle
> it is built and waiting for input.

## Contents

- [How it works](#how-it-works) · [Tech stack](#tech-stack) · [Responsible use](#responsible-use)
- [Setup](#setup) · [Using the console](#using-the-console) · [Getting gait to work](#getting-gait-to-work)
- [Command-line tools](#command-line-tools) · [The API](#the-api) · [Alerting](#alerting)
- [Evaluation](#evaluation) · [Robustness and performance](#robustness-and-performance) · [The attention head](#the-attention-head)
- [Tests](#tests) · [Configuration](#configuration) · [Layout](#layout) · [Documentation](#documentation) · [The honest limits](#the-honest-limits)

## How it works

1. **Enrol.** Upload video or photos of a watchlist person. Each is run through
   detection and tracking, and a reference template is built for every signal
   the footage supports — face, gait, appearance — then encrypted and stored.
2. **Detect and track.** For CCTV being searched, YOLOv8 finds people in every
   frame and DeepSORT links them into tracks. Each track keeps a bounded buffer
   of its recent crops.
3. **Three signals per track.**
   - *Face* — InsightFace (ArcFace) embeddings, with a quality score that
     accounts for head angle, size, and how much of the face is actually
     visible.
   - *Gait* — body silhouettes from YOLOv8-seg, averaged over whole stride
     cycles into a Gait Energy Image.
   - *Appearance* — OSNet re-identification embeddings of the whole body.
4. **Compare and fuse.** Each signal is compared with every enrolled person,
   mapped onto a common 0–1 scale, and combined with weights reflecting how
   good a look each signal got. Anything above the threshold (0.55) becomes a
   candidate.
5. **A human decides.** Every candidate is queued as *pending*. The reviewer
   sees the crop it was made on beside the enrolment photo, how much each
   signal contributed, and which signals were not counted and why — then
   confirms or rejects, recorded against their signed-in account.
6. **Alert.** Only confirmed identifications can notify anyone, by console,
   email or SMS.

## Tech stack

| Layer | Technology |
|---|---|
| Person detection | YOLOv8n (Ultralytics) |
| Tracking | DeepSORT (`deep-sort-realtime`, MobileNet appearance embedder) |
| Face | InsightFace `buffalo_l` — ArcFace, 512-d — on ONNX Runtime |
| Gait | YOLOv8n-seg silhouettes → Gait Energy Image (classical descriptor) |
| Appearance re-ID | OSNet x1_0 with MSMT17 re-ID weights (model vendored, MIT) |
| Fusion | Calibrated quality-weighted rule; keyless-attention head in PyTorch |
| Deep learning | PyTorch 2 (CUDA 12.1 wheels on Windows), NumPy, OpenCV, SciPy |
| Backend | Python 3.10, FastAPI, Uvicorn, Pydantic 2 |
| Database | SQLAlchemy 2 on SQLite by default (`--db-url` takes any SQLAlchemy URL) |
| Security | Fernet-encrypted templates and evidence (`cryptography`), operator accounts |
| Console | React 18, Vite 5 |
| Alerts | Console, SMTP, Twilio |
| Evaluation | TAR@FAR, ROC-AUC, EER, CMC, ablation and fairness breakdown |
| Tests | pytest |

## Responsible use

This system identifies people from surveillance footage. Four guardrails are
part of the design, not add-ons (section 8 of the build plan):

1. **No automated action on a match.** A human confirms before anything happens.
2. **Every match decision is logged**, including the fusion weights that
   produced it and every signal that did not count, for audit.
3. **Fairness/bias evaluation across demographics** is part of the evaluation
   harness, not an afterthought.
4. **Biometric templates are encrypted at rest.**

Enrollment footage and biometric templates are personal data. `.gitignore`
excludes `data/` for that reason — keep it that way.

## Setup

Requires Python 3.10+, and Node.js 18+ for the console. Verified on Python
3.10.11 / Windows 11 with an RTX 3050.

```bash
python -m venv .venv
```

Activate it — `.venv\Scripts\activate` on Windows (PowerShell:
`.venv\Scripts\Activate.ps1`), `source .venv/bin/activate` elsewhere.

The torch wheels on PyPI are CPU-only on Windows, so install torch from the
PyTorch index **first** if you have a CUDA GPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then the rest:

```bash
pip install -r backend/requirements.txt
```

Confirm CUDA was picked up:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If that prints `False`, everything still works — the pipeline falls back to CPU,
just slower.

**Keep the venv activated for everything below.** The pipeline's heavy
dependencies (`deep-sort-realtime`, `insightface`, `ultralytics`) are imported
lazily, so a server started with a different interpreter comes up perfectly,
serves the whole console, and only fails — as a bare `500` — the moment you
press Register or Check. `scripts/serve.py` checks for them at startup and
refuses to run rather than let that happen, naming the interpreter it is using
and the venv it thinks you meant.

Model weights download on first use: YOLOv8n (~6 MB), InsightFace `buffalo_l`
(~280 MB, to `~/.insightface`) and OSNet (~11 MB).

## Using the console

Run the API and the console together:

```bash
cd backend && python scripts/serve.py
```
```bash
cd frontend && npm install && npm run dev
```

Open `http://localhost:4173`.

**Showing the prototype?** Start the API with `--demo` and there is no sign-in
at all:

```bash
cd backend && python scripts/serve.py --demo
```

`--demo` also waives the encryption-key requirement, as long as no key is set —
registering someone needs nothing exported by hand first. A real key set
alongside `--demo` still wins and templates get encrypted as normal; the flag
only relaxes an unconfigured instance, never downgrades one that was actually
set up.

Every action is then recorded against a shared `demo` operator, the server
prints a warning on start, and the console says so on every page. It is off
unless you ask for it, because an identification system that ships open
because the safe setting was the one you had to remember is not a system
anyone should trust. Everything below applies when it is off.

**You need an account to get in.** There is no self-registration, deliberately:
shell access to the host is the right bar for a system that can confirm an
identification of a real person. Create the first one on the server:

```bash
cd backend && python scripts/manage_operators.py --create <username> --admin
```

The console has five screens.

### Register

Enrol someone from **one video** that sweeps through what each signal needs,
from **separate photos**, or straight from the **camera**. Each section's
**Check** runs detection only and takes seconds; it tells you which signals
your files can actually support before you commit.

That check matters more than it sounds. Photos enrol a face and an appearance
profile but **no gait profile at all** — a still image contains no gait
information — and nothing in the resulting numbers would tell you.

| Signal | Accepts | Needs |
|---|---|---|
| Face | photos or video | face visible, roughly front-on, enough pixels |
| Gait | **video only, of them walking** | 3s+ of steady walking, whole body including feet in frame, side-on, one direction per clip |
| Appearance | photos or video | whole body in frame |

### Records

The watchlist. For each person you can **edit details**, **replace their
footage** (re-enrol with newer material), and take them off the list in one of
three ways, each with its own confirmation:

| Action | What it does | Reversible |
|---|---|---|
| **Retire** | Off the active watchlist; nothing is destroyed | Yes — **Restore** |
| **Erase biometrics** | Destroys templates and the enrolment photo; the record and every past decision stay reviewable | No |
| **Delete permanently** | Removes the record entirely — refused once any decision has been filed against the person; erase their biometrics instead | No |

### Identify

Upload CCTV and it detects, tracks and compares everyone against the
watchlist, then reports who it found, when in the clip, and what drove each
result. It runs in the background with a progress bar, because a scan runs
three recognition models over every frame — roughly a minute per thousand
frames. Everything found is queued for review; nothing is confirmed
automatically.

### Review

The screen that matters. Each candidate shows:

- **the crop the match was made on, beside the enrolment photo**, so the
  decision is a comparison of two pictures rather than a judgement about a
  number. Where no image was captured the card says so and tells you to reject
  — an absent safeguard has to be visible;
- **the fused score and how much each signal contributed**, with what each one
  actually measures;
- **every signal that was not counted, and why** — "too few usable observations",
  "no complete gait cycle detected", "gait scoring has only been calibrated on
  synthetic walkers" — so a card showing face and appearance is
  never silently quiet about gait;
- **a caution when more than half the decision rests on appearance**, because
  clothing is the weakest of the three signals and goes stale.

Confirm or reject, with an optional note. The operator recorded is your
signed-in account, taken from the session rather than the request — a review
cannot claim to be someone else's.

Evidence crops are personal data. They are encrypted like the templates,
served only to a signed-in operator, and
`WatchlistRepository.purge_evidence(days)` drops them from old decisions
without touching the decisions, their scores or their reviews. How long to keep
them is a policy call for whoever runs the deployment, so nothing calls it
automatically.

### Activity

Confirmed identifications, and the append-only record of everything the system
has been asked to do. A change of mind adds an entry rather than replacing one.

## Getting gait to work

**Right now gait does not count toward any score.** Every result still shows
gait's outcome, but its weight is withheld (`fusion.gait_anchors_validated:
false`), for a measured reason: gait's scoring scale — the similarity a stranger
scores and the similarity the same person scores — came from synthetic
walkers. Every gait similarity measured on real footage falls below the
"stranger" anchor and calibrates to exactly 0, and allowed to vote, that zero
would have taken 21–65% of the weight and pushed seven of the eight test
identifications under the threshold. The full measurements are in
[docs/gait-investigation.md](docs/gait-investigation.md).

What gait needs, in order:

1. **Footage where the legs are visible.** Side-on, or far enough back that the
   whole body stays in frame. Someone walking straight at a camera at head
   height gives nothing: measured on the test clips, the "leg" band of the
   silhouette is chest and shoulders for much of the second half, because the
   legs have left the frame. Gait correctly refuses that.
2. **Clean enrolment walks.** Several seconds of steady walking in one
   direction per clip. A single take that walks left, right, towards and away
   leaves most of it unusable — on the enrolled walks here, only the second
   half of each clip produced gait at all.
3. **At least three people enrolled with gait.** Gait descriptors are dominated
   by the shared shape of a walking human, so they are compared with the
   population mean removed, and that mean cannot be estimated from fewer
   (`fusion.gait_min_references_for_centring`).
4. **A real calibration set** — several people, each filmed walking on two
   separate occasions — to re-measure `fusion.gait_impostor` and
   `fusion.gait_genuine`. Then set `fusion.gait_anchors_validated: true`.

Live scans keep their own few seconds of each track for gait, thinned by whole
frames so the spacing stays even (`track_buffer.gait_sample_hz`), so a stride
fits whether the camera runs at 25 or 60 fps. Face and appearance are
unaffected by that.

**Gait also limits how far you can raise `video.frame_stride`.** Cadence is a
rate, so it can only be recovered if frames arrive often enough to sample it.
At 25 fps a stride of 1 or 2 is fine; at 5 a half step-cycle arrives as roughly
two samples, and autocorrelation on two samples does not report uncertainty —
it locks onto the full cycle and returns a confident number twice the truth.
Gait refuses below that and says so in the log. Face and appearance are
unaffected, so a high stride is still the right lever when you do not need gait.

## Command-line tools

Everything the console does is also available from the command line, which is
what you want for batch runs and calibration.

### Detection and tracking

Generate a smoke-test clip (pans across Ultralytics' sample image, which
contains pedestrians) if you have no footage yet:

```bash
cd backend && python scripts/make_test_video.py
```

Then run detection + tracking over it:

```bash
cd backend && python scripts/run_pipeline.py --source ../data/test_videos/synthetic_pan.mp4
```

Useful flags:

| Flag | Effect |
|---|---|
| `--source 0` | Use webcam instead of a file |
| `--save-annotated` | Write a video with boxes + track IDs to `data/output/` |
| `--stride N` | Process every Nth frame |
| `--max-frames N` | Stop after N frames |
| `--conf 0.5` | Detection confidence threshold |
| `--model yolov8s.pt` | Larger detector (better on small/distant people) |
| `--device cpu` | Force CPU |

The run ends with a report: frames processed, total person detections, and one
row per track ID showing how many frames it survived.

### Enrol and match

Generate smoke-test fixtures if you have no footage yet:

```bash
cd backend && python scripts/make_face_fixtures.py
```

Enroll someone. The footage should show **one** person:

```bash
cd backend && python scripts/enroll.py --source ../data/test_videos/enroll_subject_a.mp4 --person-id subject_a --name "Test Subject A"
```

`--list` shows who is enrolled, `--inspect <id>` shows one record in detail,
including which signals were actually stored.

Then match against other footage:

```bash
cd backend && python scripts/match.py --source ../data/test_videos/probe_two_subjects.mp4 --show-all
```

The report lists each candidate with its fused score, the weight each signal
carried, and every signal that was not counted and why. `--show-all` also
prints below-threshold tracks, which is what you need when calibrating the
threshold. Nothing is acted on automatically.

To feed the console's review queue from the command line instead, match
against the database and record the results:

```bash
cd backend && python scripts/match.py --source <clip> --from-db --record --camera-id cam-1
```

### Fusion and calibration

**Scores are fused and calibrated.** The three signals produce similarities on
completely different scales — two different people score 0.03 by face but
0.755 by re-ID (measured against ImageNet weights; see below) — so each is
mapped onto a common 0–1 scale before being combined, where 0 means
"indistinguishable from a stranger" and 1 means "as good as a genuine match
gets". One threshold (`fusion.threshold`, default 0.55) then applies to every
signal alike.

Pick the fusion rule with `--strategy`:

| Strategy | Behaviour |
|---|---|
| `single_best` | Trust only the most reliable signal present |
| `average` | Equal weight to every signal with a score (the paper's baseline) |
| `quality_weighted` | Weight by how good a look each signal got and how much it can tell people apart (default) |

**Re-ID references go stale.** Re-ID encodes clothing as much as the person, so
its stored reference decays with a 3-day half-life (`reid.trust_half_life_days`).
Re-enroll if you need it current.

**The re-ID anchors are provisional, and the console says so.** The appearance
branch originally loaded OSNet's *ImageNet* checkpoint — generic classification
features presented as re-identification. Two strangers scoring 0.755 is exactly
what that produces; a model trained to separate identities pushes impostors far
lower. The default is now a checkpoint trained for re-ID on MSMT17
(`reid.weights`).

Re-measured under the new checkpoint on the same fixtures:

| checkpoint | impostor | genuine | separation |
|---|---|---|---|
| `imagenet` | 0.765 | 0.942 | 0.177 |
| `msmt17` | 0.726 | 0.881 | 0.155 |

The impostor drops, which is what a model trained to tell people apart should
do. The genuine score drops further — the fixtures are two crops of one
photograph, so ImageNet's generic features were partly matching the *image*
rather than the person. The old `reid_threshold: 0.88` then sat directly on top
of the genuine score of 0.881 and would have rejected a correct match on a
rounding error; it is now 0.80, the midpoint of the measured pair.

This is still one photograph of two people. Re-derive from your own footage with
the TAR@FAR curve before operational use, and set
`fusion.reid_anchors_measured_on` to whatever you measured against — when it
disagrees with `reid.weights`, `/api/stats` reports it and the console shows it
above the review queue, so nobody confirms an identification without knowing the
score is uncalibrated.

**Changing `reid.weights` invalidates everyone already enrolled.** A stored
template is only comparable to a probe from the same model: the vectors still
load, still have the right length, and still produce a cosine similarity — one
that means nothing. Every template records which model produced it, and a
cross-model pair is reported as "could not compare" rather than scored. Anything
enrolled before that record existed is allowed but flagged on `/api/stats`,
because stranding an existing watchlist would be worse than saying it is
unverified. **If you change the checkpoint, re-enrol.**

### Encrypting stored templates

Biometric templates are personal data and the build plan requires them
encrypted at rest. Generate a key and put it in your environment:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Set it as `FRS_TEMPLATE_ENCRYPTION_KEY` (see `.env.example`).

**Enrollment refuses to write without a key.** It used to warn and write
plaintext, which made the insecure state the default and — because encryption
is recorded per template — let a database end up half-encrypted while still
looking right in a spot check. For local work with throwaway data, set
`FRS_ALLOW_PLAINTEXT_TEMPLATES=1` to make that choice deliberate — or start
the server with `--demo`, which sets it for you.

Keep the key somewhere other than the database it protects: records encrypted
with a key cannot be read back without it.

## The API

```bash
cd backend && python scripts/serve.py
```

Interactive docs at `http://127.0.0.1:8000/docs`. To move enrollments created
by `scripts/enroll.py` into the database:

```bash
cd backend && python scripts/serve.py --import-enrollments
```

The workflow enforces the build plan's guardrails structurally rather than by
convention:

| Guardrail | How it is enforced |
|---|---|
| No action on a match alone | Matches are always created `pending`; the only route to `confirmed` is `POST /decisions/{id}/review`, and the operator recorded is the authenticated session, not a name in the request body |
| Alerting sees confirmed only | `/alerts` returns confirmed decisions; a pending one is invisible to it |
| Audit trail is append-only | A review adds a row; it never edits the decision, so a change of mind leaves both judgements visible |
| Every decision explains itself | Each decision stores its fusion weights, calibrated scores, and every signal that did not count and why |
| Templates stay private | The API describes templates but never returns a vector |
| Removal preserves history | `DELETE /watchlist/{id}` retires rather than deletes; erasing biometrics keeps every decision reviewable; permanent deletion is refused (`409`) once any decision references the person |

**Set `FRS_TEMPLATE_ENCRYPTION_KEY` before enrolling anyone through the API.**
Without it, enrollment is refused rather than falling back to plaintext. The
database file is git-ignored — it holds biometric templates and the whole audit
trail.

## Alerting

```bash
cd backend && python scripts/send_alerts.py
```

Dry run by default — it shows exactly what would be delivered. `--send` is
required to actually notify, and `--transport smtp|twilio` selects how.

Alerts fire **only** on decisions a human has confirmed, and the gate is
checked twice: once when selecting decisions, once immediately before sending.
Every alert names the confirming operator and lists which signals drove the
score, because a notification saying only "match found, 0.91" invites exactly
the unexamined trust the review step exists to prevent.

## Evaluation

```bash
cd backend && python ../eval/run_evaluation.py --synthetic --people 80 --sightings 8 --fairness
```

Produces the ablation table — each signal alone, then each fusion rule — with
TAR@FAR, ROC-AUC, EER and CMC, plus a per-group fairness breakdown.

It prints **two** tables, and the reason matters. A single-signal row can only
score pairs where that signal is visible on both sides, so "face only" gets
graded on the easy cases. Comparing its AUC against fusion's directly is
meaningless. The first table shows each row on its own coverage; the second
restricts everything to pairs where all three signals are present, which is
the only like-for-like comparison.

What that shows: face alone beats fusion when a face is available (0.949 vs
0.811 AUC) — but it is only available on **33%** of pairs, against fusion's
**97%**. Fusion buys coverage, not peak accuracy. In deployment the alternative
to a fused score on a turned-away person is no score at all.

## Robustness and performance

```bash
cd backend && python scripts/test_disguise.py
```
```bash
cd backend && python scripts/benchmark.py
```

The first measures the face branch under synthetic masks, sunglasses and hoods.
Similarity and quality fall together, which is what fusion needs — a masked
face still matches at 0.741, while masked *and* wearing sunglasses drops to
0.254 and correctly does not.

**The face branch scores how much of the face is visible**
(`face.occlusion_aware`, `app/embeddings/occlusion.py`). Without it a covered
face scored *higher* quality than a clear one — an opaque shape makes the
detector more confident while head angle and pixel count do not move — and
quality is what fusion weights by, so the branch that had stopped working was
handed the vote. Measured on a real enrolment crop, a mask *raised* quality
from 0.596 to 0.642; with occlusion awareness on it falls to 0.263, and mask
plus sunglasses is refused.

The benchmark times each stage. It found face embedding taking 70% of runtime
at 5.3s per track; capping the live path at the best 16 crops halved total
runtime.

## The attention head

The project's core contribution is implemented and trains:

```bash
cd backend && python scripts/train_fusion.py --synthetic
```

On synthetic data it learns exactly what it should — weighting face (0.591)
above re-ID (0.541) when both are present, reversing the mistake the fixed
rules make.

**It cannot be used for real matching.** It is trained on synthetic identities
that do not reproduce the structure of real embeddings, so it refuses to run
against real footage rather than silently producing worse results than the
fixed rules. Training it properly needs labelled same/different pairs — the
same person recorded twice plus other people — which is where real footage
stops being optional.

## Tests

```bash
cd backend && python -m pytest
```

499 tests. Twenty of them need model weights or real video, and are skipped
unless you ask for them:

```bash
cd backend && python -m pytest --run-slow
```

## Configuration

[`backend/config.yaml`](backend/config.yaml) holds the defaults, with the
measurement behind most of them in the comments. Any value can be overridden
by an environment variable using the `FRS_` prefix and `__` for nesting:

```bash
FRS_DETECTION__CONF_THRESHOLD=0.5
FRS_DEVICE=cpu
```

Precedence: CLI flags > environment variables > `config.yaml`.

## Layout

```
backend/
  app/
    core/          config, logging, shared types, video reader, track buffer, evidence crops
    detection/     YOLOv8 person detector
    tracking/      DeepSORT tracker
    embeddings/    face.py, occlusion.py, disguise.py, gait.py, silhouette.py, reid.py
      vendor/      OSNet model definition, vendored (MIT)
    matching/      watchlist gallery, open-set ranking, gait population centring
    fusion/        calibration.py, baseline.py        (live)
                   attention.py, training.py          (implemented, not used on real data)
    api/           FastAPI app, routes, auth, enrolment ingest, scanning, background jobs
    db/            models + repository: people, templates, decisions, reviews, audit trail
    alerts/        console / SMTP / Twilio, confirmed-only gate
    pipeline.py    detection + tracking spine
  scripts/         CLI tools: serve, enroll, match, operators, alerts, calibration, benchmarks
  tests/
frontend/src/      React console: Register, Records, Identify, Review, Activity
eval/              TAR@FAR, ROC-AUC, EER, CMC, ablation, fairness breakdown
data/              enrolment footage, test clips, model weights, database  (git-ignored)
docs/              investigation notes, audit, project report
```

## Documentation

| Document | What it covers |
|---|---|
| [faceless-frs-build-plan.md](faceless-frs-build-plan.md) | The original spec, the 11 build phases, and where the project diverges from the paper |
| [docs/phase-notes.md](docs/phase-notes.md) | Design notes and measurements, phase by phase |
| [docs/AUDIT.md](docs/AUDIT.md) | Security and correctness review findings |
| [docs/gait-investigation.md](docs/gait-investigation.md) | Why gait never fired on the test footage, the four defects underneath, and the measurements |
| [docs/report/Faceless-FRS-Report.docx](docs/report/Faceless-FRS-Report.docx) | Project report with results (its gait sections predate the gait investigation) |

## The honest limits

Everything is implemented; very little is validated on real people.

1. **Gait does not count toward scores yet.** Its calibration is synthetic, and
   on real footage its accuracy is unmeasured rather than good or bad: one
   genuine same-person pair exists, and the result flips depending on which
   population mean it is centred against. The negative direction — refusing
   people who are not walking — is tested on real video.
2. **Appearance is mostly clothing.** On the test footage half the
   identifications rested on appearance alone, and it does not survive a
   change of clothes — which is why its trust halves every three days.
3. **The attention head refuses to run**, because it can only be trained on
   synthetic data and an untrained head is worse than the fixed rules.
4. **The central claim is unproven** — whether attention fusion beats
   `quality_weighted` has never been tested on real data.
5. **Every threshold is measured on a handful of clips.**

Real footage means several people, each recorded more than once, ideally at
different times and cameras, with consent and a lawful basis. See
[docs/phase-notes.md](docs/phase-notes.md).

Two standing caveats on signal strength. Gait uses a **classical GEI
descriptor, not a learned embedding**, because OpenGait ships no licence file
and its weights are unusable without vendoring its model code — and its
descriptors must be population-centred to discriminate at all. Re-ID is
strongest within a single camera and day and degrades across both.
