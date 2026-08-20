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

> **Status: Phase 2 complete.** Detection, tracking, and the face branch run
> end to end: you can enroll a person from video and match them in other
> footage. Gait and re-ID (phases 3–4) are not wired in yet.

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

## Enroll and match (Phase 2)

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
calibrating the threshold. Nothing is acted on automatically — matches are
printed as candidates for a human to confirm.

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
    embeddings/    face.py [Phase 2 ✓] / gait.py, reid.py [Phases 3-4]
    matching/      watchlist gallery + open-set ranking  [Phase 2 ✓]
    fusion/        baseline.py / attention.py            [Phases 5-6]
    api/           FastAPI routes                        [Phase 7]
    db/            models + migrations                   [Phase 7]
    alerts/        Twilio / SMTP                         [Phase 9]
    pipeline.py    detection + tracking spine            [Phase 1 ✓]
  scripts/         CLI entry points
  tests/
frontend/          enrollment UI, live dashboard         [Phase 8]
eval/              TAR@FAR, ROC-AUC, CMC, ablation table [Phase 10]
data/
  enrollment/      per-person reference footage          (git-ignored)
  test_videos/     CCTV test clips                       (git-ignored)
  models/          downloaded weights                    (git-ignored)
docs/
```

Modules for phases 3+ exist as documented placeholders that raise
`NotImplementedError` — the layout is in place, the code is not.

## Next: Phase 3

Gait branch. One risk to settle before writing code: the plan assumes
pretraining on CASIA-B, which is licence-gated and has been unreliable to
obtain. Whether OpenGait's published checkpoints are usable independently of
that dataset licence decides the approach. See
[docs/phase-notes.md](docs/phase-notes.md) for the fallbacks.
