# Faceless FRS — review console

React + Vite dashboard for reviewing candidate identifications.

```bash
npm install
npm run dev      # http://localhost:4173
```

The backend must be running (`cd backend && python scripts/serve.py`). Vite
proxies `/api` to `http://127.0.0.1:8000`, so the browser stays on one origin
and there is no CORS configuration involved.

## Screens

| Screen | Purpose |
|---|---|
| **Review queue** | Candidates awaiting a human verdict, each with its full per-modality breakdown |
| **Watchlist** | Who is enrolled and which signals are stored for them |
| **Confirmed** | Identifications a human has confirmed, with who confirmed them |
| **Audit trail** | Enrolments, retirements and reviews, newest first |

## Why the review queue looks the way it does

Section 8 of the build plan requires a human to confirm a match before anything
follows from it. That is only worth anything if the human can distinguish a
strong match from a merely plausible one, so every candidate shows which
modalities drove the score and what each of them actually measures — "Build and
clothing", not "reid".

When re-ID carries more than half the weight, the card says so outright.
Appearance is the weakest of the three signals and decays as people change
clothes, and a reviewer about to put a name to a face deserves to know that the
score in front of them mostly reflects a jacket.

## What is not here

- No live video. This shows decisions the matcher has already recorded.
- No enrollment form — the API has no upload endpoint yet, so enrolment runs
  through `backend/scripts/enroll.py`.
- No authentication. The operator name is recorded but not verified, so the
  audit trail shows a claimed identity.
