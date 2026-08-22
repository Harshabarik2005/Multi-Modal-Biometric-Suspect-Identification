# Faceless FRS — security & correctness review

Reviewed 22 August 2026 · commit `c80518f` · ~11,200 lines (backend, eval harness, React console)

**29 findings: 2 critical · 8 high · 11 medium · 8 low**

Full report with reproductions: https://claude.ai/code/artifact/68ae5198-38b1-415a-a394-0935114390ba

## What was and was not run

PyPI was blocked in the review environment, so `torch`, `ultralytics`, `fastapi` and
`insightface` could not be installed — **the test suite and the API were never executed**.
Treat "273 tests pass" as unverified.

What was done instead: byte-compiled every Python file (clean, no syntax errors anywhere),
confirmed the project `.venv` resolves every pin correctly (torch 2.5.1+cu121,
ultralytics 8.4.123, numpy 1.26.4, setuptools 80.10.2 under the `<81` pin), and ran the
pure-numpy / pydantic modules directly to reproduce specific defects.

Findings tagged **[reproduced]** were executed. Findings tagged **[inspection]** were not.

---

## Security

### SEC-01 · CRITICAL · [reproduced]
**No authentication on any endpoint; the "named human operator" is a browser-supplied string.**
`app/api/routes.py` (all 10 routes), `app/api/ingest.py` (all 5), `app/api/main.py:87-108`,
`frontend/src/App.jsx:426-434`

15 endpoints, zero auth, no CORS middleware, no CSRF token, no rate limit. Anyone who can reach
port 8000 can read the watchlist and audit trail, enrol people, run scans, and **confirm a
match** — the one act that makes an identification actionable and can trigger an SMS/email.
The operator name is typed into a text box, cached in `localStorage`, and sent as ordinary
request data. `db/models.py:181` says "Free text until Phase 7 grows real authentication";
Phase 7 is marked complete. Guardrails 1 and 2 are structurally shaped but unenforceable —
the audit trail records unverified self-asserted names and is fully repudiable.

`POST /api/enroll` and `POST /api/scan` are `multipart/form-data` — CORS-simple, no preflight —
so any page an operator visits can silently POST to their local instance.

*Fix:* auth dependency on the router; operator identity from the authenticated principal, never
from the request body.

### SEC-02 · CRITICAL · [reproduced]
**A scan request permanently rewrites the match threshold for the whole process.**
`app/api/ingest.py:349-350`, `app/core/config.py:373-386`

`get_settings()` is `@lru_cache`d — one shared mutable object. `settings.fusion.threshold = threshold`
writes into that singleton and is never undone. One unauthenticated request with `threshold=0`
makes every later scan match everyone. No `ge=`/`le=` bounds; races concurrent jobs.

```
>>> s = get_settings(); s.fusion.threshold      -> 0.55
>>> get_settings() is s                         -> True
>>> s.fusion.threshold = 0.01                   # what /api/scan does
>>> get_settings().fusion.threshold             -> 0.01
```

*Fix:* `settings.model_copy(deep=True)` per job; bound the form field to [0,1].

### SEC-03 · HIGH · [inspection]
**Pickle deserialization of weights downloaded from Google Drive at runtime.**
`app/embeddings/reid.py:316` (`torch.load(..., weights_only=False)`), same at `app/fusion/attention.py:266`

`_download_weights` fetches a `.pth` over gdown with no checksum and no signature, then loads it
with pickle enabled. Arbitrary code execution as the server user if the Drive object is swapped,
the account compromised, or anything can write to `data/models/`. These are plain state dicts —
`weights_only=True` loads them unchanged. A Drive quota-error HTML page also passes the
`path.is_file()` check and only fails later inside the unpickler.

*Fix:* `weights_only=True` both places, plus a pinned SHA-256 verified before load.

### SEC-04 · HIGH · [reproduced]
**SMTP alerts negotiate TLS without verifying the server certificate.**
`app/alerts/notifier.py:183-187`

`server.starttls()` with no `context` makes CPython use `ssl._create_stdlib_context()` —
`check_hostname=False`, `verify_mode=CERT_NONE`. The credentials sent on the next line, and the
alert body (a confirmed identification naming a person, camera and timestamp), travel over a
channel any on-path attacker can terminate.

```
ssl._create_stdlib_context()   check_hostname=False  verify_mode=CERT_NONE
ssl.create_default_context()   check_hostname=True   verify_mode=CERT_REQUIRED
```

*Fix:* `server.starttls(context=ssl.create_default_context())`; consider refusing to send when
`smtp_use_tls` is false.

### SEC-05 · HIGH · [reproduced]
**`FRS_DATABASE_URL` is silently ignored — biometric data always lands in local SQLite.**
`app/core/config.py:314-319`, `app/api/main.py:61-66`

`Settings` declares no `database_url` field and sets `extra="ignore"`, so pydantic-settings
swallows the env var documented in `.env.example` and the README. `getattr(settings,
"database_url", None)` is always `None`, falling through to `sqlite:///data/faceless_frs.db`.
Someone who follows the docs to move the watchlist onto Postgres gets a local file with the
templates and the whole audit trail, no error, no warning. Only `serve.py --db-url` works — and
that puts the DB password in the process command line.

`job_workers` has the same problem: documented, never a field, always 1.

*Fix:* declare `database_url` and `job_workers`; set `extra="forbid"` so a mistyped `FRS_` var
fails loudly.

### SEC-06 · HIGH · [inspection]
**Encryption at rest is opt-in, so the insecure state is the default.**
`app/db/repository.py:87-100`, `app/matching/gallery.py:368-381`

Without `FRS_TEMPLATE_ENCRYPTION_KEY` the code writes raw float32 vectors and logs a warning.
Every default path produces plaintext biometric data. Because `Template.encrypted` is per-row a
deployment can end up half-encrypted and look compliant in a spot check.

*Fix:* refuse to write with no key, behind an explicit `FRS_ALLOW_PLAINTEXT_TEMPLATES=1` escape
hatch for local dev.

### SEC-07 · MEDIUM · [reproduced]
**Path traversal through `person_id` in the file-backed gallery.**
`app/matching/gallery.py:397-398`, reachable from `scripts/enroll.py --person-id`

```
'subject_a'              -> data/enrollment/subject_a
'../../../../tmp/pwned'  -> data/enrollment/../../../../tmp/pwned
'/tmp/absolute_escape'   -> /tmp/absolute_escape      (absolute path replaces the root)
```

The API path writes to the DB and is unaffected, but `person_id` is unvalidated there too
(`ingest.py:198` constrains only length). Upload filenames are handled correctly by
`Path(...).name` — this is the gap that one missed.

*Fix:* one shared validator, `^[A-Za-z0-9._-]{1,64}$`.

### SEC-08 · MEDIUM · [inspection]
**Job listing exposes every scan result; job errors leak server paths.**
`app/api/ingest.py:401-403`, `app/api/jobs.py:105`

`GET /api/jobs` returns the last 25 jobs to anyone with full `result` payloads — person IDs,
names, cameras, timestamps, scores. `job.error = str(exc)` puts raw exception text (temp paths,
model paths, DB messages) into the response.

### SEC-09 · MEDIUM · [inspection]
**Enrolment lifts the memory bound to 100,000 crops; preview blocks the event loop.**
`app/api/media.py:139-153`, `app/api/ingest.py:48, 146-159`

`observations_from_video` raises `max_observations` 64 -> 100,000 and `max_tracks` 50 -> 100.
With a 500 MB upload allowance that is tens of thousands of 256px crops in RAM — the failure
`track_buffer.py` was written to prevent, disabled on the one path taking untrusted input.
`preview_enrollment` is `async def` but runs YOLO synchronously, stalling every other request.

### SEC-10 · MEDIUM · [inspection]
**Preview mutates shared settings while a scan job may be reading them.**
`app/api/media.py:139-153` — `TrackBufferStore` holds a live reference to `settings.track_buffer`,
so a preview arriving mid-scan changes the running scan's eviction bounds and changes them back.
*Fix:* pass an explicit `TrackBufferSettings` (the constructor already accepts one).

### SEC-11 · LOW · [inspection]
**Uploaded footage is not always deleted.** `app/api/ingest.py:307, 383` — the `rmtree` lives in
the job body's `finally`, so if the job never runs (server stop, executor shutdown, cancelled
queue) the temp directory with the footage survives with nothing tracking it.

### SEC-12 · LOW · [inspection]
**Alert credentials are plain strings.** `app/core/config.py:284, 291` — `smtp_password` and
`twilio_auth_token` are ordinary `str`, so any `repr(settings)` or validation error prints them.
*Fix:* `SecretStr`.

---

## Correctness

### LOG-01 · HIGH · [reproduced]
**The fairness breakdown grades each group at its own threshold, hiding the gap it exists to find.**
`eval/metrics.py:322-338`; test that should catch it: `tests/test_eval.py:204-219`

The docstring says per-group TAR@FAR "at a *shared* threshold is what makes such a gap visible".
The implementation calls `evaluate_verification` per group, which derives a fresh threshold from
each group's own impostor distribution.

```
What fairness_breakdown() reports
  group A   TAR@FAR 0.01 = 1.000   (own threshold +0.311)
  group B   TAR@FAR 0.01 = 0.824   (own threshold +0.828)
  reads as: the system serves group A better

At the deployed shared threshold (+0.807)
  group A   TAR = 0.426   FAR = 0.000     <- missed 57% of the time
  group B   TAR = 0.916   FAR = 0.020     <- falsely flagged 2% of the time
```

The report inverts both. The existing test gives both groups identical impostor distributions
(so the thresholds coincide) and asserts on AUC, which is threshold-free — it cannot catch this.

*Fix:* one threshold on the pooled impostor set; report each group's TAR **and** FAR at it. Add a
test where the groups' impostor distributions differ.

### LOG-02 · HIGH · [inspection]
**The ablation prints its verdict from the table it has just told you not to compare.**
`eval/ablation.py:122-166`, `eval/run_evaluation.py:214-221`

`table()` emits "these AUCs are NOT directly comparable" and then, in the same output, prints
`Best by AUC:` and `attention - quality_weighted = …`. Both come from `best()`, a plain `max` over
rows of different coverage. `run_evaluation.py` calls this on the native-coverage table, so the
project's central claim gets its verdict from the comparison the module documents as invalid.
This is the mechanism behind the README's "face alone beats fusion (0.949 vs 0.811)" line.

*Fix:* compute `best()` and the attention delta only on the common-subset result.

### LOG-03 · HIGH · [reproduced]
**Re-ID staleness decay never runs in the web path — or in the evaluation.**
`app/api/scanning.py:284-288` (no `trusts=`), `eval/ablation.py:214` (no `trust`);
only caller: `scripts/match.py:53, 297`

`trust_at()` is what the README and `reid.py` both describe as stopping the system
"confidently misidentifying people by their coat". It has exactly one caller, in the CLI.
`gallery.rank()` defaults `trusts` to `{}` and `FusionInput.trust` to `1.0`, so every decision
created through the browser — the documented primary workflow — weights a three-week-old
clothing reference as if enrolled this morning. The ablation omits it too, so the harness
measures a `quality_weighted` the CLI does not deploy.

The CLI version has its own flaw: `reid_trusts()` uses the **oldest** enrolment in the whole
gallery as one global multiplier, so one stale person degrades re-ID for everyone. It is also
recomputed inside the per-track loop, re-parsing every timestamp on each re-match.

*Fix:* make trust per-person, resolved inside `Gallery.rank` from `PersonRecord.enrolled_at`;
drop the `trusts` parameter so no caller can forget it.

### LOG-04 · MEDIUM · [reproduced]
**The enrolment preview says gait is ready when the upload is mostly photographs.**
`app/api/media.py:253-262` — `summary.observations` counts video frames and photos together.

```
1 video (5 tracked frames) + 20 photographs, gait.min_frames = 20:
  face  ready=True   25 usable image(s)
  gait  ready=True   25 continuous frames available    <- only 5 came from video
  reid  ready=True   25 usable image(s)
```

This is the exact failure the endpoint exists to prevent.
*Fix:* track video-derived observations separately and gate gait on that count alone.

### LOG-05 · MEDIUM · [reproduced]
**Two uploaded videos are spliced into one fake continuous walk.**
`app/api/media.py:169-210` — the docstring promises monotonic renumbering; only *images* get an
offset.

```
two 4-frame videos ->
  frame indices : [0, 1, 2, 3, 0, 1, 2, 3]
  monotonic?      False
```

The combined list goes to `GaitEmbedder.embed()` as one sequence, so the autocorrelation and GEI
are computed across the join between two recordings, potentially at different frame rates and of
different dominant tracks.

### LOG-06 · MEDIUM · [inspection]
**Gait cadence assumes a gapless 25 fps stride-1 stream that nothing guarantees.**
`app/embeddings/silhouette.py:184-212`, `app/embeddings/gait.py:63-76, 117-166`, `config.yaml`

`extract` silently drops frames whose mask fails, but `cadence_signal` / `estimate_half_period`
treat the survivors as uniformly sampled. And `min_half_period=7` / `max_half_period=40` are in
*processed* frames: set `video.frame_stride: 5` and a real half-cycle of ~12 frames becomes ~2.4
samples, below the floor, so gait silently reports no signal for everyone and it looks like
footage quality rather than configuration.

*Fix:* derive lag bounds from `fps / frame_stride`; interpolate over or reject sequences with gaps.

### LOG-07 · MEDIUM · [inspection]
**Annotated video is written at an invented frame rate.** `app/pipeline.py::_open_writer` uses
`fps = 30.0 / max(1, frame_stride)`, ignoring `reader.meta.fps` which it already has. 25 fps
plays 20% fast, 60 fps plays at half speed, and burnt-in frame numbers stop matching wall-clock
position in the clip.

### LOG-08 · MEDIUM · [inspection]
**"Nothing is sent twice" holds only until the audit trail passes 5,000 events.**
`app/alerts/notifier.py:291-303` — `already_sent()` scans `audit_trail(limit=5000)`. Enrolments,
retirements, reviews and alerts share that table, so older `alert_sent` records fall out of the
window and those decisions get notified again. To the recipient it reads as a second sighting.
*Fix:* query `AuditEvent` filtered on `kind`, or add `alerted_at` to `MatchDecision`.

### LOG-09 · MEDIUM · [inspection]
**A decision can be re-reviewed indefinitely, flipping its actionable state.**
`app/db/repository.py:277-324` — `review()` never checks the decision's current state. Reviews
are genuinely append-only so the history survives, but `MatchDecision.status` is what `/alerts`
and `AlertDispatcher` gate on, and it is freely rewritable. With SEC-01 unfixed, by anyone.

### LOG-10 · LOW · [inspection]
**The CLI floods the review queue the API is careful not to flood.**
`scripts/match.py:325-340` writes a PENDING decision on *every* above-threshold re-match, while
`app/api/scanning.py:317-340` deliberately records one per track with a comment explaining why.
The README tells you to feed the queue with `match.py --record`.

### LOG-11 · LOW · [reproduced]
**The guard against unmeasurable TAR@FAR is a self-assignment.** `eval/metrics.py:239-245` —
`report.tar_at_far[far] = report.tar_at_far.get(far, 0.0)` changes nothing.
`smallest_resolvable_far` is set as an undeclared attribute and never read anywhere.

```
50 impostor pairs -> smallest resolvable FAR = 0.02
  FAR 0.0001   TAR 1.000   <- below resolution, reported anyway
```

### LOG-12 · LOW · [inspection]
**The audit record can name a fusion rule that was not used.** `app/fusion/baseline.py:204-209`
falls back to `AverageFusion(...).fuse(inputs)` (whose `FusionResult.strategy` is `"average"`),
but `app/api/scanning.py:314` records the *outer* `strategy.name` (`"quality_weighted"`).
*Fix:* record `result.strategy` from the `FusionResult`.

### LOG-13 · LOW · [inspection]
**Cross-group impostor pairs are attributed entirely to one group.** `eval/ablation.py:78-80` —
`Pair.group` returns `a.group or b.group`. Cross-group pairs are the majority of impostor pairs
in any multi-group set, so LOG-01's per-group FAR is measured on a mislabelled population.

### LOG-14 · LOW · [inspection]
**Uploads with the same filename overwrite each other.** `app/api/ingest.py:117-136` —
`destination = directory / name` with no de-duplication; the survivor is processed twice.

---

## Design & process

### DES-01 · HIGH · [inspection]
**The re-ID branch is running ImageNet weights, not re-ID weights.**
`app/embeddings/reid.py:258-264`, `data/models/osnet_x1_0_imagenet.pth`, `reid.py:300` (`num_classes=1000`)

Every URL in `PRETRAINED_URLS` is torchreid's *ImageNet* checkpoint, and the filename says so.
OSNet without re-ID fine-tuning gives generic visual features, which is a plausible explanation
for the figure quoted throughout `config.yaml` and the README: "different people 0.755, same
person 0.980". Two strangers scoring 0.755 is what generic features do. Everything anchored on
it — `reid_impostor`, `reid_threshold: 0.88`, the calibration mapping, the fusion weights — is
calibrated against the wrong model.

*Fix:* point at the Market-1501 / MSMT17 checkpoints, or state in the README that re-ID runs on
ImageNet features and its anchors are provisional for that reason.

### DES-02 · MEDIUM · [inspection]
**The reviewer confirms an identification without ever seeing the person.**
`frontend/src/App.jsx:109-180` — the review card shows a score, weight bars and a caution line.
No crop, no frame, no clip; the API returns no imagery either. The human-confirm step is the
safeguard the architecture is built around, and the human can evaluate *how* the system reached
its conclusion but not *whether* it is right.

*Fix:* store the matched crop with the decision (a crop is not a template) and show it beside the
enrolment reference.

### DES-03 · LOW · [inspection]
**A restart mid-scan leaves the browser polling forever.** `frontend/src/api.js:138-145` —
`waitForJob` is an unbounded `for(;;)` with no timeout or attempt cap.

---

## Fix order

1. **SEC-01 — authentication.** Nearly every other security finding is reachable because this is
   missing, and the guardrails cannot mean anything until the operator identity is real.
2. **SEC-02, 03, 04, 05.** Four small bounded diffs: copy settings per job, `weights_only=True`,
   an SSL context, two missing `Settings` fields.
3. **LOG-01, 02, 03.** These change what the evaluation reports and what the deployed matcher
   computes. Fix before the real-footage validation run — otherwise it measures the wrong thing
   and those numbers go into the writeup.
4. **SEC-06 and DES-01.** Decisions to make rather than bugs to fix.
5. **Everything else.** Mostly localised, mostly one function each.

---

## What holds up

- **Absence of signal is never conflated with a score of zero.** `ModalityEmbedding.similarity`
  returns `None`; every fusion strategy excludes rather than zeroes. The most common way to get
  multimodal fusion quietly wrong, avoided deliberately and consistently.
- **Uncentred gait refuses to compare** instead of returning a flattering number, with the
  reasoning measured and written down.
- **The `.gitignore` is comprehensive and it works.** `git ls-files` shows only `.gitkeep`
  placeholders and `.env.example` — no database, templates, footage or weights.
- **The template decode path is not a pickle sink** — `np.load` without `allow_pickle=True`.
- **No `dangerouslySetInnerHTML` anywhere** in the console; React's escaping is intact.
- **The comments are unusually honest about limits.** Several findings above are things the
  docstrings describe correctly and the implementation then fails to do — a much better position
  to be in than the reverse.
