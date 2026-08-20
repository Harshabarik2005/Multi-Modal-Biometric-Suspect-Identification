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

> **Status: all 11 phases implemented. 273 tests pass.**
>
> What is *not* done is validation on real footage, and that blocks the
> project's central claim: whether attention fusion beats the fixed-rule
> baselines has never been tested on real data. The evaluation harness that
> would settle it is built and waiting for input. Every threshold in the
> system is a placeholder measured on one or two clips.

## Responsible use

This system identifies people from surveillance footage. Four guardrails are
part of the design, not add-ons (section 8 of the build plan):

1. **No automated action on a match.** A human confirms before anything happens.
2. **Every match decision is logged**, including the attention weights that
   produced it, for audit.
3. **Fairness/bias evaluation across demographics** is part of the evaluation
   harness, not an afterthought.
4. **Biometric templates are encrypted at rest.**

Enrollment footage and biometric templates are personal data. `.gitignore`
excludes `data/` for that reason — keep it that way.

## Setup

Requires Python 3.10+. Verified on Python 3.10.11 / Windows 11 with an
RTX 3050.

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

## Run the Phase-1 pipeline

Generate a smoke-test clip (pans across Ultralytics' sample image, which
contains pedestrians) if you have no footage yet:

```bash
cd backend && python scripts/make_test_video.py
```

Then run detection + tracking over it:

```bash
cd backend && python scripts/run_pipeline.py --source ../data/test_videos/synthetic_pan.mp4
```

YOLO weights download automatically on first run (~6 MB for `yolov8n.pt`).

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

## Enroll and match (Phases 2-5)

Generate smoke-test fixtures if you have no footage yet:

```bash
cd backend && python scripts/make_face_fixtures.py
```

Enroll someone. The footage should show **one** person, ideally rotating
through 360° so the reference covers several angles:

```bash
cd backend && python scripts/enroll.py --source ../data/test_videos/enroll_subject_a.mp4 --person-id subject_a --name "Test Subject A"
```

InsightFace downloads its model pack (~280MB) to `~/.insightface` on first run.
`--list` shows who is enrolled, `--inspect <id>` shows one record in detail.

Then match against other footage:

```bash
cd backend && python scripts/match.py --source ../data/test_videos/probe_two_subjects.mp4 --show-all
```

`--show-all` also prints below-threshold tracks, which is what you need when
calibrating the threshold. The report names the modality behind each score
(`via face` / `via gait`), and the two have separate thresholds — a gait score
of 0.85 is a much weaker claim than a face score of 0.85. Nothing is acted on
automatically; matches are printed as candidates for a human to confirm.

**Enrolling gait needs walking footage.** A 360° rotation on the spot gives an
excellent face reference and no gait reference at all — gait needs several full
step cycles. Enroll from footage of the person walking if you want both, and
`enroll.py --inspect <id>` will show which modalities were actually stored.

**Scores are fused and calibrated.** The three modalities produce
similarities on completely different scales — two different people score 0.03
by face but 0.755 by re-ID — so each is mapped onto a common 0–1 scale before
being combined, where 0 means "indistinguishable from a stranger" and 1 means
"as good as a genuine match gets". One threshold (`fusion.threshold`, default
0.55) then applies to every modality alike.

Pick the fusion rule with `--strategy`:

| Strategy | Behaviour |
|---|---|
| `single_best` | Trust only the most reliable modality present |
| `average` | Equal weight to every modality with a signal (the paper's baseline) |
| `quality_weighted` | Weight by how good a look each modality got (default) |

The report prints the per-modality breakdown under each match, so you can see
whether a hit was driven by a clear face or mostly by a jacket.

**Re-ID references go stale.** Re-ID encodes clothing as much as the person, so
its stored reference decays with a 3-day half-life (`reid.trust_half_life_days`).
Re-enroll if you need it current.

### Encrypting stored templates

Biometric templates are personal data and the build plan requires them
encrypted at rest. Generate a key and put it in your environment:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Set it as `FRS_TEMPLATE_ENCRYPTION_KEY` (see `.env.example`). Without it,
enrollment still works but warns on every save, and templates sit on disk in
the clear. Records encrypted with a key cannot be read back without it.

## Tests

```bash
cd backend && python -m pytest
```

That runs the fast tests only. The end-to-end test needs model weights and the
generated clip:

```bash
cd backend && python -m pytest --run-slow
```

## Configuration

[`backend/config.yaml`](backend/config.yaml) holds the defaults. Any value can
be overridden by an environment variable using the `FRS_` prefix and `__` for
nesting:

```bash
FRS_DETECTION__CONF_THRESHOLD=0.5
FRS_DEVICE=cpu
```

Precedence: CLI flags > environment variables > `config.yaml`.

## Layout

```
backend/
  app/
    core/          config, logging, shared types, video reader, track buffer
    detection/     YOLOv8 wrapper                        [Phase 1 ✓]
    tracking/      DeepSORT wrapper                      [Phase 1 ✓]
    embeddings/    face.py, gait.py, silhouette.py, reid.py  [Phases 2-4 ✓]
      vendor/      OSNet model definition, vendored (MIT)
    matching/      watchlist gallery + open-set ranking  [Phase 2 ✓]
    fusion/        calibration.py, baseline.py [Phase 5 ✓]
                   attention.py, training.py   [Phase 6, untrained]
    api/           FastAPI routes + app factory          [Phase 7 ✓]
    db/            models + repository                    [Phase 7 ✓]
    alerts/        Twilio / SMTP, confirmed-only gate    [Phase 9 ✓]
    pipeline.py    detection + tracking spine            [Phase 1 ✓]
  scripts/         CLI entry points
  tests/
frontend/          React review console                  [Phase 8 ✓]
eval/              TAR@FAR, ROC-AUC, CMC, ablation table [Phase 10 ✓]
data/
  enrollment/      per-person reference footage          (git-ignored)
  test_videos/     CCTV test clips                       (git-ignored)
  models/          downloaded weights                    (git-ignored)
docs/
```

Modules for phases 8+ exist as documented placeholders that raise
`NotImplementedError` — the layout is in place, the code is not.

## The attention head (Phase 6)

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
Phase-5 rules. Training it properly needs labelled same/different pairs — the
same person recorded twice plus other people — which is where real footage
stops being optional.

## The API (Phase 7)

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
| No action on a match alone | Matches are always created `pending`; the only route to `confirmed` is `POST /decisions/{id}/review` with a named operator |
| Alerting sees confirmed only | `/alerts` returns confirmed decisions; a pending one is invisible to it |
| Audit trail is append-only | A review adds a row; it never edits the decision, so a change of mind leaves both judgements visible |
| Templates stay private | The API describes templates but never returns a vector |
| Removal preserves history | `DELETE /watchlist/{id}` retires rather than deletes, so past decisions stay reviewable |

**Set `FRS_TEMPLATE_ENCRYPTION_KEY` before enrolling anyone through the API**,
or templates are written unencrypted and every write logs a warning saying so.
The database file is git-ignored — it holds biometric templates and the whole
audit trail.

## The review console (Phase 8)

Run the API and the dashboard together:

```bash
cd backend && python scripts/serve.py
```
```bash
cd frontend && npm install && npm run dev
```

Then open `http://localhost:5173`. Feed it candidates by running the matcher
with `--record`:

```bash
cd backend && python scripts/match.py --source <clip> --from-db --record --camera-id cam-1
```

The review queue is the screen that matters. Each candidate shows the fused
score, which modalities drove it, and what each of those actually measures —
so a reviewer can judge the match rather than trust it. When more than half the
decision rests on appearance, the card says so, because clothing is the weakest
of the three signals and goes stale.

Confirming requires a name, and that name is recorded on the decision. Nothing
becomes actionable until someone does it.

## Alerting (Phase 9)

```bash
cd backend && python scripts/send_alerts.py
```

Dry run by default — it shows exactly what would be delivered. `--send` is
required to actually notify, and `--transport smtp|twilio` selects how.

Alerts fire **only** on decisions a human has confirmed, and the gate is
checked twice: once when selecting decisions, once immediately before sending.
Every alert names the confirming operator and lists which modalities drove the
score, because a notification saying only "match found, 0.91" invites exactly
the unexamined trust the review step exists to prevent.

## Evaluation (Phase 10)

```bash
cd backend && python ../eval/run_evaluation.py --synthetic --people 80 --sightings 8 --fairness
```

Produces the ablation table — each modality alone, then each fusion rule — with
TAR@FAR, ROC-AUC, EER and CMC, plus a per-group fairness breakdown.

It prints **two** tables, and the reason matters. A single-modality row can only
score pairs where that modality is visible on both sides, so "face only" gets
graded on the easy cases. Comparing its AUC against fusion's directly is
meaningless. The first table shows each row on its own coverage; the second
restricts everything to pairs where all three modalities are present, which is
the only like-for-like comparison.

What that shows: face alone beats fusion when a face is available (0.949 vs
0.811 AUC) — but it is only available on **33%** of pairs, against fusion's
**97%**. Fusion buys coverage, not peak accuracy. In deployment the alternative
to a fused score on a turned-away person is no score at all.

## Robustness and performance (Phase 11)

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

The second times each stage. It found face embedding taking 70% of runtime at
5.3s per track; capping the live path at the best 16 crops halved total runtime.

## The honest limits

Everything is implemented; almost nothing is validated on real people.

1. **Gait has no positive validation** — the negative direction is tested on
   real video, the positive rests on synthetic silhouettes.
2. **The attention head refuses to run**, because it can only be trained on
   synthetic data and an untrained head is worse than the fixed rules.
3. **The central claim is unproven** — whether attention fusion beats
   `quality_weighted` has never been tested on real data.

Real footage means several people, each recorded more than once, ideally at
different times and cameras, with consent and a lawful basis. See
[docs/phase-notes.md](docs/phase-notes.md).

Two standing caveats on modality strength. Gait uses a **classical GEI
descriptor, not a learned embedding**, because OpenGait ships no licence file
and its weights are unusable without vendoring its model code — and its
descriptors must be population-centred to discriminate at all. Re-ID is
strongest within a single camera and day and degrades across both.
