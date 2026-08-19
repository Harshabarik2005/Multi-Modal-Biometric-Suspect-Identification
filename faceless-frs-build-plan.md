# Faceless FRS — Multi-Modal Biometric Suspect Identification

Build plan and architecture reference. Built as an extension of Prakash et al.,
"Multimodal Adaptive Fusion of Face and Gait Features using Keyless Attention
based Deep Neural Networks for Human Identification" (2023, arXiv:2303.13814).

## 1. What it does

Identifies a person from CCTV footage even when their face is unclear, hidden,
or turned away, by combining three signals instead of relying on face alone:

- **Face** — via ArcFace embeddings
- **Gait** — how the person walks
- **Re-ID** — general appearance (build, clothing)

An enrollment step builds a reference profile for each watchlist person (all
three signals). A live-feed matcher then compares tracked people against that
watchlist and raises an alert on a match, with a human confirming before any
action is taken.

## 2. What the reference paper contributes, and where we diverge

The paper's core idea: a **keyless attention** mechanism learns, per frame,
how much to trust each modality — instead of a fixed rule. It normalizes those
into global weights and does a weighted fusion of embeddings.

Where this project diverges from the paper:

| Paper | This project |
|---|---|
| Face + gait only | Face + gait + re-ID (3-way fusion) |
| Closed-set softmax over 20 known people | Open-set metric learning — cosine similarity against a growable watchlist gallery |
| ConvLSTM trained from scratch on ~19K images | Pretrained backbones (ArcFace, GaitSet/GaitGL, OSNet) + a small trainable attention head |
| No occlusion/disguise handling | Occlusion-aware face branch (masks, sunglasses) |
| No handling of appearance changing over time | Time-aware trust: re-ID weight decays the further apart two sightings are |
| Accuracy / log-loss on a lab dataset | TAR@FAR, ROC-AUC, CMC/Rank-N — proper open-set watchlist metrics |
| Black-box classification | Attention weights (α/β/γ) surfaced in the dashboard for investigator review |

## 3. Tech stack

- **Detection & tracking**: YOLOv8, DeepSORT
- **Face embeddings**: ArcFace / InsightFace (or DeepFace)
- **Gait embeddings**: GaitSet or GaitGL, on Gait Energy Images (GEI)
- **Re-ID embeddings**: OSNet (torchreid)
- **Fusion**: custom keyless-attention head (PyTorch)
- **Backend**: FastAPI
- **Frontend**: React
- **Database**: PostgreSQL or MongoDB
- **Alerts**: Twilio / SMTP
- **Vector search (future scope)**: FAISS, once the watchlist grows

## 4. Pipeline

**Enrollment (offline, once per watchlist person)**

360° rotation video → face crops (pose-guided), gait silhouette sequence, body
crops → embeddings for all three → stored as that person's reference vectors.

**Live matching (per tracked person, per camera)**

Detect + track (YOLOv8 + DeepSORT) → per-track frame buffer → three embedding
branches run in parallel → keyless attention fusion produces one adaptive
embedding → cosine similarity against the watchlist gallery → match above
threshold → alert with attention-weight breakdown for a human to confirm.

## 5. Repo structure

```
faceless-frs/
  backend/
    app/
      detection/        # YOLOv8 wrapper
      tracking/          # DeepSORT wrapper
      embeddings/
        face.py           # ArcFace / InsightFace
        gait.py           # GEI extraction + GaitSet/GaitGL
        reid.py           # OSNet
      fusion/
        baseline.py        # rule-based fallback (phase 1 baseline)
        attention.py        # keyless attention fusion (the real contribution)
      api/                # FastAPI routes (enroll, match, watchlist, alerts)
      db/                 # models + migrations
      alerts/             # Twilio / SMTP
    tests/
    requirements.txt
  frontend/
    src/                  # enrollment UI, live dashboard, alerts, explainability view
  eval/
    metrics.py             # TAR@FAR, ROC-AUC, CMC/Rank-N
    ablation.py              # mirrors the paper's Table I (baseline vs naive fusion vs attention fusion)
  data/
    enrollment/
    test_videos/
  docs/
```

Two additions made during Phase 0, both additive to the above:
`backend/app/core/` (config, logging, shared types, video reader) and
`backend/scripts/` (CLI entry points).

## 6. Build phases

0. Repo scaffold, env setup, config
1. Detection + tracking working end-to-end on a test video (YOLOv8 + DeepSORT)
2. Face branch only: enroll one person, match against a test video, cosine similarity — get one modality fully working before adding more
3. Gait branch: silhouette extraction, gait cycle detection, GaitSet/GaitGL embeddings
4. Re-ID branch: OSNet embeddings
5. Fusion v1 (baseline): simple rule-based / average fusion — mirrors the paper's "average fusion" baseline, gives you a number to beat
6. Fusion v2: keyless attention fusion, trained with a metric-learning loss (ArcFace-margin or triplet) for open-set matching
7. Backend API + database (enrollment storage, matching endpoint, watchlist CRUD)
8. Frontend dashboard (enrollment flow, live monitoring, alerts, attention-weight explainability view)
9. Alerting (Twilio/SMTP)
10. Evaluation harness (TAR@FAR, ROC, CMC, ablation table)
11. Extensions: disguise-augmented training, time-aware re-ID trust decay, edge-deployment optimization (Jetson Nano)

## 7. Datasets to pull in

- **CASIA-B** — better gait pretraining than the paper's CASIA-A (124 subjects, 11 angles, bag/coat conditions)
- **OU-MVLP** — large-scale gait, if more data is needed
- **Disguised Faces in the Wild (DFW)** — for testing/training disguise robustness
- Own recorded enrollment + CCTV test footage (per the testing plan: enroll one person across angles, validate against group footage)

## 8. Responsible-use guardrails to build in from the start

- No automated action on a match alone — always a human-confirm step
- Log every match decision (including attention weights) for audit
- Run a fairness/bias check across demographics as part of evaluation
- Encrypt stored biometric templates at rest
