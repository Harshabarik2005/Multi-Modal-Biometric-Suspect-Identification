import { useCallback, useEffect, useState } from 'react'
import {
  api,
  clearToken,
  setUnauthorizedHandler,
  MODALITIES,
  appearanceShare,
} from './api'
import Enroll from './Enroll'
import Login from './Login'
import Scan from './Scan'

/* ------------------------------------------------------------------ *
 * Explainability
 * ------------------------------------------------------------------ */

/**
 * The per-modality breakdown behind a match.
 *
 * This is the screen the build plan's section 8 is really about. A reviewer
 * must be able to see *why* the system fired before confirming it, because a
 * match resting on a clear face and one resting on a jacket are very different
 * grounds for acting on an identification -- and they look identical if you
 * only show the final score.
 */
function WeightBreakdown({ weights = {}, calibrated = {}, strategy }) {
  const entries = Object.entries(weights).filter(([, w]) => w > 0)
  if (entries.length === 0) {
    return <p className="muted">No per-modality breakdown was recorded.</p>
  }

  const total = entries.reduce((sum, [, w]) => sum + w, 0)

  return (
    <div className="breakdown">
      <div className="breakdown-bar" role="img" aria-label="Modality contribution">
        {entries.map(([modality, weight]) => {
          const meta = MODALITIES[modality] || {}
          const pct = (weight / total) * 100
          return (
            <div
              key={modality}
              className="breakdown-segment"
              style={{ width: `${pct}%`, background: meta.colour || '#64748b' }}
              title={`${meta.label || modality}: ${pct.toFixed(0)}% of the decision`}
            />
          )
        })}
      </div>

      <table className="breakdown-table">
        <thead>
          <tr>
            <th>Modality</th>
            <th>Weight</th>
            <th>Score</th>
            <th>What it measures</th>
          </tr>
        </thead>
        <tbody>
          {entries
            .sort((a, b) => b[1] - a[1])
            .map(([modality, weight]) => {
              const meta = MODALITIES[modality] || {}
              return (
                <tr key={modality}>
                  <td>
                    <span
                      className="swatch"
                      style={{ background: meta.colour || '#64748b' }}
                    />
                    {meta.label || modality}
                  </td>
                  <td className="mono">{((weight / total) * 100).toFixed(0)}%</td>
                  <td className="mono">
                    {calibrated[modality] !== undefined
                      ? calibrated[modality].toFixed(2)
                      : '—'}
                  </td>
                  <td className="muted">{meta.blurb || ''}</td>
                </tr>
              )
            })}
        </tbody>
      </table>
      {strategy && <p className="muted small">Fusion strategy: {strategy}</p>}
    </div>
  )
}

/**
 * Warns when a match rests mostly on appearance.
 *
 * Not decoration. Re-ID encodes clothing more than the person, so a
 * confirmation on that basis deserves more scepticism than the raw score
 * suggests.
 */
function EvidenceCaution({ weights }) {
  const share = appearanceShare(weights)
  if (share < 0.5) return null
  return (
    <p className="caution">
      <strong>{(share * 100).toFixed(0)}% of this match rests on appearance
      (build and clothing).</strong>{' '}
      Appearance is the weakest of the three signals and goes stale as people
      change clothes. Treat this as a weaker identification than the score alone
      suggests.
    </p>
  )
}

/* ------------------------------------------------------------------ *
 * Review queue
 * ------------------------------------------------------------------ */

/**
 * The two pictures a reviewer needs: who was seen, and who they are supposed
 * to be (DES-02).
 *
 * Before this, the review card showed a score, some weight bars and a caution
 * line. The human confirmation is the safeguard the whole architecture is
 * built around, and the human could see *how* the system reached its
 * conclusion but had no way to judge *whether* it was right. Confirming an
 * identification of someone you have never seen is not a check.
 *
 * When an image is missing that is stated rather than hidden. A reviewer being
 * asked to decide without evidence needs to know that is what is happening.
 */
function EvidencePair({ decision, onError }) {
  const [images, setImages] = useState(null)

  useEffect(() => {
    let cancelled = false
    let urls = []

    Promise.all([
      decision.has_evidence ? api.evidenceImage(decision.id) : null,
      decision.has_reference ? api.referenceImage(decision.person_id) : null,
    ])
      .then(([seen, enrolled]) => {
        if (cancelled) {
          urls = [seen, enrolled].filter(Boolean)
          return
        }
        urls = [seen, enrolled].filter(Boolean)
        setImages({ seen, enrolled })
      })
      .catch((error) => {
        if (!cancelled) onError?.(error.message)
      })

    return () => {
      cancelled = true
      urls.forEach((url) => URL.revokeObjectURL(url))
    }
  }, [decision.id, decision.person_id, decision.has_evidence, decision.has_reference])

  return (
    <div className="evidence">
      <figure className="evidence-pane">
        <figcaption className="muted small">Seen on camera</figcaption>
        {images?.seen ? (
          <img src={images.seen} alt="The person detected in the footage" />
        ) : (
          <div className="evidence-missing">
            {decision.has_evidence
              ? 'Loading…'
              : 'No image was captured for this match.'}
          </div>
        )}
      </figure>

      <figure className="evidence-pane">
        <figcaption className="muted small">
          Enrolled as {decision.display_name}
        </figcaption>
        {images?.enrolled ? (
          <img src={images.enrolled} alt={`Enrolment reference for ${decision.display_name}`} />
        ) : (
          <div className="evidence-missing">
            {decision.has_reference
              ? 'Loading…'
              : 'No enrolment image was stored.'}
          </div>
        )}
      </figure>
    </div>
  )
}

function DecisionCard({ decision, onReviewed, onError }) {
  const [reason, setReason] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (verdict) => {
    setBusy(true)
    try {
      // No operator argument: the server takes it from the session, so a
      // decision records who was actually signed in.
      await api.review(decision.id, verdict, reason)
      onReviewed()
    } catch (error) {
      onError(error.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <article className="card">
      <header className="card-head">
        <div>
          <h3>{decision.display_name}</h3>
          <p className="muted small">
            {decision.person_id} · track {decision.track_id}
            {decision.camera_id && ` · ${decision.camera_id}`} · frame{' '}
            {decision.frame_index}
          </p>
        </div>
        <div className="score">
          <span className="score-value">{decision.score.toFixed(3)}</span>
          <span className="muted small">fused score</span>
        </div>
      </header>

      <EvidencePair decision={decision} onError={onError} />

      {!decision.has_evidence && (
        <p className="caution">
          There is no image of this sighting, so the only thing to judge is the
          score. Reject unless you have another way to verify it.
        </p>
      )}

      <WeightBreakdown
        weights={decision.weights}
        calibrated={decision.calibrated}
        strategy={decision.strategy}
      />
      <EvidenceCaution weights={decision.weights} />

      <div className="review">
        <input
          type="text"
          placeholder="Note (optional) — why you decided this"
          value={reason}
          onChange={(event) => setReason(event.target.value)}
          disabled={busy}
        />
        <div className="review-actions">
          <button
            className="btn btn-reject"
            onClick={() => submit('rejected')}
            disabled={busy}
          >
            Not a match
          </button>
          <button
            className="btn btn-confirm"
            onClick={() => submit('confirmed')}
            disabled={busy}
          >
            Confirm identification
          </button>
        </div>
      </div>
      <p className="muted small">
        Raised {new Date(decision.created_at).toLocaleString()}. Nothing has
        happened yet — confirming is what makes this actionable.
      </p>
    </article>
  )
}

function ReviewQueue({ onError }) {
  const [decisions, setDecisions] = useState([])
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      setDecisions(await api.decisions(true))
    } catch (error) {
      onError(error.message)
    } finally {
      setLoading(false)
    }
  }, [onError])

  useEffect(() => {
    load()
    const timer = setInterval(load, 10000)
    return () => clearInterval(timer)
  }, [load])

  if (loading && decisions.length === 0) return <p className="muted">Loading…</p>
  if (decisions.length === 0) {
    return (
      <div className="empty">
        <h3>Nothing awaiting review</h3>
        <p className="muted">
          Candidate matches appear here when the matcher runs with{' '}
          <code>--record</code>. They stay pending until someone reviews them.
        </p>
      </div>
    )
  }

  return (
    <>
      <p className="muted">
        {decisions.length} candidate{decisions.length === 1 ? '' : 's'} awaiting
        review. None of them has triggered anything.
      </p>
      {decisions.map((decision) => (
        <DecisionCard
          key={decision.id}
          decision={decision}
          onReviewed={load}
          onError={onError}
        />
      ))}
    </>
  )
}

/* ------------------------------------------------------------------ *
 * Watchlist, alerts, audit
 * ------------------------------------------------------------------ */

function Watchlist({ onError }) {
  const [people, setPeople] = useState([])

  useEffect(() => {
    api.watchlist().then(setPeople).catch((e) => onError(e.message))
  }, [onError])

  if (people.length === 0) {
    return (
      <div className="empty">
        <h3>Nobody is enrolled</h3>
        <p className="muted">
          Enroll with <code>scripts/enroll.py</code>, then import with{' '}
          <code>scripts/serve.py --import-enrollments</code>.
        </p>
      </div>
    )
  }

  return (
    <table className="table">
      <thead>
        <tr>
          <th>ID</th>
          <th>Name</th>
          <th>Stored signals</th>
          <th>Enrolled</th>
        </tr>
      </thead>
      <tbody>
        {people.map((person) => (
          <tr key={person.person_id}>
            <td className="mono">{person.person_id}</td>
            <td>{person.display_name}</td>
            <td>
              {person.templates.map((template) => {
                const meta = MODALITIES[template.modality] || {}
                return (
                  <span
                    key={template.modality}
                    className="pill"
                    style={{ borderColor: meta.colour }}
                    title={
                      template.encrypted
                        ? 'Encrypted at rest'
                        : 'NOT ENCRYPTED — set FRS_TEMPLATE_ENCRYPTION_KEY'
                    }
                  >
                    {meta.label || template.modality}
                    {!template.encrypted && ' ⚠'}
                  </span>
                )
              })}
            </td>
            <td className="muted small">
              {person.enrolled_at
                ? new Date(person.enrolled_at).toLocaleDateString()
                : '—'}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function Alerts({ onError }) {
  const [alerts, setAlerts] = useState([])

  useEffect(() => {
    api.alerts().then(setAlerts).catch((e) => onError(e.message))
  }, [onError])

  if (alerts.length === 0) {
    return (
      <div className="empty">
        <h3>No confirmed identifications</h3>
        <p className="muted">
          Only human-confirmed matches appear here. A pending decision is
          invisible to alerting by design.
        </p>
      </div>
    )
  }

  return (
    <>
      {alerts.map((alert) => (
        <article key={alert.id} className="card">
          <header className="card-head">
            <div>
              <h3>{alert.display_name}</h3>
              <p className="muted small">
                {alert.person_id} · track {alert.track_id}
                {alert.camera_id && ` · ${alert.camera_id}`}
              </p>
            </div>
            <div className="score">
              <span className="score-value">{alert.score.toFixed(3)}</span>
              <span className="muted small">confirmed</span>
            </div>
          </header>
          <WeightBreakdown
            weights={alert.weights}
            calibrated={alert.calibrated}
            strategy={alert.strategy}
          />
          {alert.reviews.map((review, index) => (
            <p key={index} className="muted small">
              {review.verdict} by <strong>{review.operator}</strong> ·{' '}
              {new Date(review.created_at).toLocaleString()}
              {review.reason && ` — “${review.reason}”`}
            </p>
          ))}
        </article>
      ))}
    </>
  )
}

function Audit({ onError }) {
  const [events, setEvents] = useState([])

  useEffect(() => {
    api.audit().then(setEvents).catch((e) => onError(e.message))
  }, [onError])

  if (events.length === 0) return <p className="muted">No audit events yet.</p>

  return (
    <table className="table">
      <thead>
        <tr>
          <th>When</th>
          <th>Event</th>
          <th>Actor</th>
          <th>Subject</th>
          <th>Detail</th>
        </tr>
      </thead>
      <tbody>
        {events.map((event, index) => (
          <tr key={index}>
            <td className="muted small">
              {new Date(event.created_at).toLocaleString()}
            </td>
            <td>
              <span className="pill">{event.kind}</span>
            </td>
            <td>{event.actor}</td>
            <td className="mono">{event.subject || '—'}</td>
            <td className="muted small mono">
              {Object.keys(event.detail).length
                ? JSON.stringify(event.detail)
                : '—'}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/* ------------------------------------------------------------------ *
 * Shell
 * ------------------------------------------------------------------ */

const TABS = [
  { id: 'enroll', label: 'Enrol someone' },
  { id: 'scan', label: 'Search footage' },
  { id: 'review', label: 'Review queue' },
  { id: 'watchlist', label: 'Watchlist' },
  { id: 'alerts', label: 'Confirmed' },
  { id: 'audit', label: 'Audit trail' },
]

export default function App() {
  const [session, setSession] = useState(null)
  const [tab, setTab] = useState('enroll')
  const [error, setError] = useState(null)
  const [stats, setStats] = useState(null)
  // Bumped to force the watchlist and queue to refetch after an enrolment or
  // scan, so the tabs are never stale.
  const [version, setVersion] = useState(0)

  const onError = useCallback((message) => setError(message), [])

  // A token can expire mid-session. Rather than showing a wall of failures,
  // drop straight back to the sign-in screen.
  useEffect(() => {
    setUnauthorizedHandler(() => {
      setSession(null)
      setError('Your session ended. Sign in again.')
    })
    return () => setUnauthorizedHandler(null)
  }, [])

  useEffect(() => {
    if (!session) return
    api
      .stats()
      .then(setStats)
      .catch(() => setStats(null))
  }, [session, version, tab])

  if (!session) {
    return (
      <>
        {error && (
          <div className="login-notice">
            <div className="error" onClick={() => setError(null)}>
              {error}
            </div>
          </div>
        )}
        <Login onSignedIn={setSession} />
      </>
    )
  }

  const signOut = () => {
    clearToken()
    setSession(null)
    setError(null)
  }

  return (
    <div className="app">
      <header className="masthead">
        <div>
          <h1>Faceless FRS</h1>
          <p className="muted small">
            Multi-modal identification — face, gait and appearance
          </p>
        </div>
        <div className="session">
          <span className="muted small">
            Signed in as <strong>{session.display_name || session.username}</strong>
          </span>
          <button className="btn btn-reject btn-small" onClick={signOut}>
            Sign out
          </button>
        </div>
      </header>

      <p className="notice">
        This system does not act on its own. Every candidate is a suggestion for
        a human to confirm or reject, the reasoning behind each one is shown so
        it can be judged rather than trusted, and whatever you decide is
        recorded against your account.
      </p>

      {stats?.warnings?.length > 0 && (
        <div className="caution banner">
          <strong>Scores on this deployment are not trustworthy yet.</strong>
          <ul>
            {stats.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </div>
      )}

      {error && (
        <div className="error" onClick={() => setError(null)}>
          {error} <span className="muted small">(click to dismiss)</span>
        </div>
      )}

      <nav className="tabs">
        {TABS.map((entry) => (
          <button
            key={entry.id}
            className={`tab ${tab === entry.id ? 'active' : ''}`}
            onClick={() => setTab(entry.id)}
          >
            {entry.label}
            {entry.id === 'review' && stats?.pending_decisions > 0 && (
              <span className="badge">{stats.pending_decisions}</span>
            )}
          </button>
        ))}
      </nav>

      <main>
        {tab === 'enroll' && (
          <Enroll onError={onError} onEnrolled={() => setVersion((v) => v + 1)} />
        )}
        {tab === 'scan' && (
          <Scan onError={onError} onScanned={() => setVersion((v) => v + 1)} />
        )}
        {tab === 'review' && <ReviewQueue key={version} onError={onError} />}
        {tab === 'watchlist' && <Watchlist key={version} onError={onError} />}
        {tab === 'alerts' && <Alerts key={version} onError={onError} />}
        {tab === 'audit' && <Audit key={version} onError={onError} />}
      </main>
    </div>
  )
}
