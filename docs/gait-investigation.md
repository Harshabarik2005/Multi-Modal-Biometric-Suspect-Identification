# Why gait never fired — and the four defects underneath

16 September 2026 · three people enrolled (`case-0003` Ashish v3, `case-0004` Anurag N,
`case-0005` Hemanth), all three carrying face, gait and re-ID templates · test footage
`test/new_2` (six clips) and `test/old` (two clips)

Gait contributed nothing to any of the eight identifications raised from `new_2`. On that
footage that was the correct outcome. Underneath it were four real defects, three of which
would have mattered on any footage at all.

---

## 1. That footage carries no gait, and refusing was right

All six clips are 58–60fps, 848x478, 7.4–10.8s, and in each the subject walks straight into
a camera at roughly head height. The body leaves the bottom of the frame part-way through.

Leg-band width of the normalised silhouette — the signal cadence is read from — across
`test6`'s longest track (466 silhouettes), by decile:

| decile | 0–10 | 10–20 | 20–30 | 30–40 | 40–50 | 50–60 | 60–70 | 70–80 | 80–90 | 90–100 |
|---|---|---|---|---|---|---|---|---|---|---|
| leg-band width (px) | 9.5 | 9.5 | 9.7 | 9.3 | 11.7 | 19.8 | 31.0 | 42.3 | 43.9 | 41.7 |
| silhouette area | 849 | 853 | 860 | 843 | 976 | 1308 | 1643 | 2022 | 2122 | 2534 |

The canvas is 44px wide. From 60% of the clip onward the "leg" band fills it: that is chest
and shoulders, because the legs are out of shot. 309 of 491 boxes on that track end within
4px of the frame's bottom edge.

Bypassing the gates deliberately, to see what gait *would* have said on the three clips that
contain Ashish (centred similarity against each enrolled template):

| clip | case-0005 Hemanth | case-0003 Ashish | case-0004 Anurag | winner | margin |
|---|---|---|---|---|---|
| test6 | **+0.393** | −0.186 | −0.290 | Hemanth (wrong) | +0.580 |
| test5 | **+0.385** | −0.150 | −0.332 | Hemanth (wrong) | +0.535 |
| test2 | **+0.405** | −0.210 | −0.273 | Hemanth (wrong) | +0.615 |

Gait quality on those tracks was 0.95–0.98, so it would have been weighted as strong
evidence. The gates were the only thing preventing a confident wrong identification.

## 2. Gait could not have fired in a live scan on 60fps footage at all

The per-track buffer held 64 **frames**. That is 2.6s at 25fps but 1.07s at 60fps, and gait
refuses until it has seen one whole stride — a little over a second. Measured on Ashish's own
clean side-on enrolment walk, running the real gate chain over sliding windows:

| sample rate | window | span | windows passing every gait gate |
|---|---|---|---|
| 60Hz | 64 | 1.07s | **0 of 76** |
| 60Hz | 128 | 2.14s | 6 of 36 |
| 30Hz | 64 | 2.13s | 8 of 36 |
| 30Hz | 96 | 3.20s | 20 of 23 |
| 20Hz | 64 | 3.20s | 19 of 23 |

**Fix.** Gait now keeps its own ring of the same observation objects, thinned towards
`track_buffer.gait_sample_hz` (20/s) by skipping a whole number of frames. Face and
appearance keep the untouched every-frame ring.

Two things were got wrong on the way here, both caught by measurement:

- **Thinning the shared buffer hurt face.** Face and re-ID choose crops from the buffer by
  box height, and for someone walking into a camera the tallest boxes are mid-walk, when the
  face is still small. A 3.2s window traded the close-ups for them: on `test2` an
  identification fell from 0.713 to 0.569 as its face score dropped from 0.83 to 0.59. Hence
  the separate ring.
- **A wall-clock slot spaced frames unevenly.** "One every 0.05s" out of 25fps keeps four
  frames in five, with gaps 0.04/0.04/0.04/0.08, and `resample_cadence` lays its grid on the
  *median* gap. Cadence coverage falls from 1.00 to 0.80 at 25fps and 0.67 at 29.97fps
  against a floor of 0.60 — before a single mask has failed. A whole-frame stride keeps the
  spacing even at every rate: 1 at 25 and 29.97fps (exactly what gait saw before), 3 at 60fps.

## 3. The "is the body cut off?" check missed almost every case

`at_frame_edge` compared the box against the frame edge exactly, and a detector box on a body
running out of shot stops a pixel or two short of it.

| clip | boxes within 4px of the bottom edge | flagged |
|---|---|---|
| test6 | 309 of 491 | 10 |
| test2 | 181 of 474 | 17 |

Those fragments reached gait presented as whole bodies, and the flag multiplies into gait
quality. Fixed with `track_buffer.edge_margin_fraction` (1% of the frame dimension).

Impact on enrolment, measured on the three enrolled walks: 0.0%, 0.0% and 4.7% of frames are
newly flagged. The flag feeds quality only, never the stored vector, so no template changes.

## 4. Gait's scoring scale is unmeasured, and would have wiped out working results

`fusion.gait_impostor = 0.52` and `gait_genuine = 0.95` were measured on **synthetic**
walkers. Every centred gait similarity measured on real footage — genuine and impostor alike,
−0.76 to +0.41 — sits *below* the impostor anchor, so all of them calibrate to exactly 0.0.
Gait's separation is 0.43 (nearly double re-ID's 0.225) and its trust is fixed at 1.0, so it
would have taken 21–65% of the weight while contributing nothing:

| result as raised | carried by | gait's share if it fired | becomes | verdict |
|---|---|---|---|---|
| 1.000 | appearance | 65% | 0.351 | gone |
| 0.984 | appearance | 65% | 0.345 | gone |
| 0.907 | appearance | 65% | 0.318 | gone |
| 0.889 | appearance | 65% | 0.312 | gone |
| 0.860 | appearance | 65% | 0.302 | gone |
| 0.774 | face 66% + appearance | 39% | 0.477 | gone |
| 0.713 | face 86% + appearance | 21% | 0.567 | survives |
| 0.599 | face 85% + appearance | 22% | 0.473 | gone |

Threshold is 0.55, and these are upper bounds (they assume the most generous possible re-ID
weight). This is the same failure `fusion/baseline.py` already documents for re-ID — every
score under its own impostor anchor, clipping to 0.0, taking 58% of the weight and dragging a
correct 0.74 down to 0.30 — in a modality with twice the pull.

**Fix.** `ModalityCalibration.withheld_reason`: gait is still computed, compared and reported,
and the reviewer is told it was not counted and why, but it carries no weight until
`fusion.gait_anchors_validated` is set.

## 5. The first genuine-versus-impostor gait measurement on real footage

Every gait number in this project had come from synthetic walkers. `ashish_2/walk.mp4` and
`ashish_3/walk.mp4` are two separate recordings of the same person, which makes one genuine
pair; the other enrolled walks give impostor pairs.

| comparison | genuine | impostor | separation |
|---|---|---|---|
| raw cosine | +0.9373 | +0.9102 … +0.9835 | **−0.046 (overlapping)** |
| centred on the 3 enrolled templates | +0.3055 | −0.5389 … +0.0503 | +0.2552 |
| centred on the 4 measured vectors | −0.4700 | −0.7198 … +0.3037 | **−0.774 (overlapping)** |

The same four vectors give opposite answers depending on which population mean they are
centred against. With one genuine pair and a result that flips with an arbitrary choice,
gait's accuracy on this footage is **unmeasured** — not good, not bad.

For context, raw cosine between *different* people's stored gait templates is 0.9722–0.9849:
every Gait Energy Image is a blurry human, and so is everyone else's. (Three mean-centred
vectors spanning a rank-2 space is arithmetic, not a finding.)

## 6. Nothing said why a signal went unused

The review card showed face and appearance and was silent about gait, so "saw nothing",
"could not be compared" and "not allowed to vote" all looked identical — absent.

Each decision now records `unused_json`, served as `DecisionOut.unused` and rendered as a
muted "Not counted: …" row in the console and in the CLI's report. The DES-01 caution
("present but could not be compared") is now keyed on `ModalityScore.model_mismatch`, so
routine refusals no longer trigger the warning meant for a reference enrolled under a
different model.

---

## Verification

- **479 unit tests pass; 498 with `--run-slow`**, which includes the real-footage paths
  (gait refusing stationary people, enrol-then-scan, end-to-end tracking).
- **All eight clips re-scanned through the real scan path.** Five results are bit-identical:
  0.889, 0.860, 0.984, 1.000, 0.907. Three rose — Hemanth 0.713 → 0.823 and 0.599 → 0.699,
  Ashish 0.774 → 0.789 — because re-ID's weight collapsed from ~15% to ~1%: appearance trust
  halves every 3 days (`reid.trust_half_life_days`) and enrolment is now 13 days old. Backing
  face out of the arithmetic gives 0.83 and 0.71, unchanged. Face and appearance see exactly
  the frames they saw before.
- Gait reported an honest reason on every clip: "no complete gait cycle detected", "signal not
  periodic enough to be a walk", "fewer frames than one gait cycle", or "silhouette shape keeps
  changing — the body is leaving the frame, turning, or not segmenting cleanly".
- The console rows were checked in the running app against a scratch database. The live
  watchlist database and its audit trail were never written to.

## What gait still needs before it can be trusted

1. **Footage where the legs are visible** — side-on, or far enough back that the whole body
   stays in frame. Walking into the camera is the one angle gait cannot use.
2. **Better enrolment takes.** Only the *second half* of each of the three `walk.mp4` files
   produces gait at all; the first half fails as "signal not periodic enough". Filming several
   walk directions in a single take leaves most of it unusable.
3. **A real calibration set** — several people, each filmed walking on two separate occasions
   — to re-measure `gait_impostor` / `gait_genuine`, after which
   `fusion.gait_anchors_validated` can be turned on. One genuine pair exists today.

## Relationship to the Word report

`docs/report/Faceless-FRS-Report.docx` predates this investigation. Its limitation that
"whether this is the buffer length or the specific footage has not been determined" is
answered above — both, for different reasons — and its gait sections are superseded by this
document.
