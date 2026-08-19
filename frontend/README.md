# Faceless FRS — frontend

Placeholder. The dashboard is **Phase 8**; nothing is built here yet.

Planned (React, per section 3 of the build plan):

- **Enrollment flow** — upload a 360° rotation video, review the extracted
  face / gait / body crops, confirm before the reference profile is stored.
- **Live monitoring** — camera feeds with tracked people and their IDs.
- **Alerts** — a match raises an alert. Per section 8 of the build plan, the
  UI must require an explicit human confirmation before any action follows.
  No auto-dismiss, no auto-escalate.
- **Explainability view** — the attention weights (α face / β gait / γ re-ID)
  that produced the match, alongside the frames each modality actually used.
  An investigator has to be able to see *why* the system fired, and a match
  driven almost entirely by clothing should look obviously different from one
  driven by a clear face.

The backend API those screens consume is Phase 7 (`backend/app/api/`).
