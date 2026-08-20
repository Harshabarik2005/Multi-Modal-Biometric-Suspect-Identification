import { useCallback, useEffect, useState } from 'react'
import { api, MODALITIES, appearanceShare } from './api'

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

function DecisionCard({ decision, operator, onReviewed, onError }) {
  const [reason, setReason] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (verdict) => {
    if (!operator.trim()) {
      onError('Enter your name before reviewing. Decisions record who made them.')
      return
    }
    setBusy(true)
    try {
      await api.review(decision.id, operator.trim(), verdict, reason)
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

function ReviewQueue({ operator, onError }) {
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
          operator={operator}
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
  { id: 'review', label: 'Review queue' },
  { id: 'watchlist', label: 'Watchlist' },
  { id: 'alerts', label: 'Confirmed' },
  { id: 'audit', label: 'Audit trail' },
]

export default function App() {
  const [tab, setTab] = useState('review')
  const [error, setError] = useState(null)
  const [health, setHealth] = useState(null)
  // Kept in localStorage so a reviewer does not retype it, but still recorded
  // on every decision -- an anonymous confirmation is not an audit trail.
  const [operator, setOperator] = useState(
    () => localStorage.getItem('frs-operator') || '',
  )

  useEffect(() => {
    localStorage.setItem('frs-operator', operator)
  }, [operator])

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch((e) => setError(`Cannot reach the API: ${e.message}`))
  }, [])

  const onError = useCallback((message) => setError(message), [])

  return (
    <div className="app">
      <header className="masthead">
        <div>
          <h1>Faceless FRS</h1>
          <p className="muted small">
            Multi-modal identification — face, gait and appearance
          </p>
        </div>
        <div className="operator">
          <label htmlFor="operator">Reviewing as</label>
          <input
            id="operator"
            type="text"
            placeholder="your name"
            value={operator}
            onChange={(event) => setOperator(event.target.value)}
          />
        </div>
      </header>

      <p className="notice">
        This system does not act on its own. Every candidate below is a
        suggestion for a human to confirm or reject, and the reasoning behind
        each one is shown so it can be judged rather than trusted.
      </p>

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
            {entry.id === 'review' && health?.pending_decisions > 0 && (
              <span className="badge">{health.pending_decisions}</span>
            )}
          </button>
        ))}
      </nav>

      <main>
        {tab === 'review' && <ReviewQueue operator={operator} onError={onError} />}
        {tab === 'watchlist' && <Watchlist onError={onError} />}
        {tab === 'alerts' && <Alerts onError={onError} />}
        {tab === 'audit' && <Audit onError={onError} />}
      </main>
    </div>
  )
}
