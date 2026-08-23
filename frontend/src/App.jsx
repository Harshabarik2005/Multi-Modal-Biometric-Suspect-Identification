import { useCallback, useEffect, useState } from 'react'
import {
  api,
  appearanceShare,
  clearToken,
  MODALITIES,
  setUnauthorizedHandler,
} from './api'
import Identify from './Identify'
import Login from './Login'
import PersonShot from './PersonShot'
import Records from './Records'
import Register from './Register'

/* ------------------------------------------------------------------ *
 * Shared explainability pieces
 * ------------------------------------------------------------------ */

/**
 * What the system relied on, and how much.
 *
 * Segments run widest-first, and each signal's grey is fixed by how much
 * evidence that signal carries — face near-black, appearance near-white. A
 * decision resting mostly on clothing therefore *looks* washed out, where a
 * coloured chart would have let it look as vivid as a face match.
 */
function WeightBreakdown({ weights = {}, calibrated = {}, strategy }) {
  const entries = Object.entries(weights).filter(([, weight]) => weight > 0)
  if (entries.length === 0) {
    return <p className="muted small">No per-signal breakdown was recorded.</p>
  }

  const total = entries.reduce((sum, [, weight]) => sum + weight, 0)
  const ordered = [...entries].sort((a, b) => b[1] - a[1])

  return (
    <div className="breakdown">
      <div className="breakdown-bar" role="img" aria-label="Signal contribution">
        {ordered.map(([modality, weight]) => {
          const meta = MODALITIES[modality] || {}
          const share = (weight / total) * 100
          return (
            <div
              key={modality}
              className="breakdown-segment"
              style={{ width: `${share}%`, background: meta.tone || '#999' }}
              title={`${meta.label || modality}: ${share.toFixed(0)}% of this decision`}
            />
          )
        })}
      </div>

      <table className="table">
        <thead>
          <tr>
            <th>Signal</th>
            <th>Weight</th>
            <th>Score</th>
            <th>What it measures</th>
          </tr>
        </thead>
        <tbody>
          {ordered.map(([modality, weight]) => {
            const meta = MODALITIES[modality] || {}
            return (
              <tr key={modality}>
                <td style={{ whiteSpace: 'nowrap' }}>
                  <span className="swatch" style={{ background: meta.tone }} />
                  {meta.label || modality}
                </td>
                <td className="num">{((weight / total) * 100).toFixed(0)}%</td>
                <td className="num">
                  {calibrated[modality] !== undefined
                    ? calibrated[modality].toFixed(2)
                    : '—'}
                </td>
                <td className="muted small">{meta.blurb || ''}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      {strategy && (
        <p className="muted small" style={{ marginTop: 8 }}>
          Fusion rule: <span className="mono">{strategy}</span>
        </p>
      )}
    </div>
  )
}

/**
 * Warns when a match rests mostly on appearance.
 *
 * Not decoration. Appearance encodes clothing more than the person, so a
 * confirmation on that basis deserves more scepticism than the score suggests.
 */
function AppearanceCaution({ weights }) {
  const share = appearanceShare(weights)
  if (share < 0.5) return null
  return (
    <p className="caution">
      <strong>
        {(share * 100).toFixed(0)}% of this rests on appearance — build and
        clothing.
      </strong>{' '}
      That is the weakest of the three signals and it goes stale as people
      change clothes. Treat it as a weaker identification than the score alone
      suggests.
    </p>
  )
}

/* ------------------------------------------------------------------ *
 * Review queue
 * ------------------------------------------------------------------ */

function DecisionCard({ decision, demoMode, onReviewed, onError }) {
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
            <span className="mono">{decision.person_id}</span> · track{' '}
            <span className="num">{decision.track_id}</span>
            {decision.camera_id && ` · ${decision.camera_id}`} · frame{' '}
            <span className="num">{decision.frame_index}</span>
          </p>
        </div>
        <div className="score">
          <span className="score-value">{decision.score.toFixed(3)}</span>
          <span className="score-label">score</span>
        </div>
      </header>

      <div className="evidence">
        <figure className="evidence-pane">
          <figcaption>Seen on camera</figcaption>
          <PersonShot
            key={`seen-${decision.id}`}
            alt="The person detected in the footage"
            missing="No image was captured for this sighting."
            fetcher={
              decision.has_evidence ? () => api.evidenceImage(decision.id) : null
            }
          />
        </figure>
        <figure className="evidence-pane">
          <figcaption>Registered as {decision.display_name}</figcaption>
          <PersonShot
            key={`ref-${decision.person_id}`}
            alt={`Registration photo for ${decision.display_name}`}
            missing="No registration photo stored."
            fetcher={
              decision.has_reference
                ? () => api.referenceImage(decision.person_id)
                : null
            }
          />
        </figure>
      </div>

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
      <AppearanceCaution weights={decision.weights} />

      <label className="stacked" style={{ marginTop: 14 }}>
        <span>
          Note <span className="hint">— optional, but it is what the record shows later</span>
        </span>
        <input
          type="text"
          placeholder="Why you decided this"
          value={reason}
          onChange={(event) => setReason(event.target.value)}
          disabled={busy}
        />
      </label>

      <div className="actions actions-end" style={{ marginTop: 12 }}>
        <button
          className="btn btn-quiet"
          onClick={() => submit('rejected')}
          disabled={busy}
        >
          Not this person
        </button>
        <button
          className="btn btn-primary"
          onClick={() => submit('confirmed')}
          disabled={busy}
        >
          Confirm identification
        </button>
      </div>

      <p className="muted small" style={{ marginTop: 12 }}>
        Raised {new Date(decision.created_at).toLocaleString()}. Nothing has
        happened yet — confirming is what makes this actionable, and it is
        recorded{' '}
        {demoMode
          ? 'against a shared demo operator, not against you'
          : 'against your account'}
        .
      </p>
    </article>
  )
}

function ReviewQueue({ demoMode, onError }) {
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

  if (loading && decisions.length === 0) {
    return <p className="muted small">Loading…</p>
  }

  return (
    <div className="division">
      <div className="division-head">
        <h2>Review</h2>
        <p>
          {decisions.length === 0
            ? 'Nothing is waiting. Candidates appear here after a search.'
            : `${decisions.length} candidate${decisions.length === 1 ? '' : 's'} awaiting a
               decision. None of them has triggered anything, and none will until
               you say so.`}
        </p>
      </div>

      {decisions.length === 0 ? (
        <div className="empty">
          <h3>Nothing awaiting review</h3>
          <p>
            Search some footage under Identify and any candidates will queue up
            here. They stay pending until a person decides.
          </p>
        </div>
      ) : (
        decisions.map((decision) => (
          <DecisionCard
            key={decision.id}
            decision={decision}
            demoMode={demoMode}
            onReviewed={load}
            onError={onError}
          />
        ))
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ *
 * Activity: confirmed identifications and the audit trail
 * ------------------------------------------------------------------ */

function Activity({ onError }) {
  const [alerts, setAlerts] = useState([])
  const [events, setEvents] = useState([])

  useEffect(() => {
    api.alerts().then(setAlerts).catch((e) => onError(e.message))
    api.audit().then(setEvents).catch((e) => onError(e.message))
  }, [onError])

  return (
    <div className="division">
      <div className="division-head">
        <h2>Activity</h2>
        <p>
          Confirmed identifications, and the record of everything this system
          has been asked to do. The trail is append-only — a change of mind adds
          an entry rather than replacing one.
        </p>
      </div>

      <div className="card-title-rule">Confirmed identifications</div>
      {alerts.length === 0 ? (
        <div className="empty">
          <h3>None confirmed</h3>
          <p>
            Only human-confirmed matches appear here. A pending candidate is
            invisible to alerting by design.
          </p>
        </div>
      ) : (
        alerts.map((alert) => (
          <article key={alert.id} className="card">
            <header className="card-head">
              <div>
                <h3>{alert.display_name}</h3>
                <p className="muted small">
                  <span className="mono">{alert.person_id}</span> · track{' '}
                  <span className="num">{alert.track_id}</span>
                  {alert.camera_id && ` · ${alert.camera_id}`}
                </p>
              </div>
              <div className="score">
                <span className="score-value">{alert.score.toFixed(3)}</span>
                <span className="score-label">confirmed</span>
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
        ))
      )}

      <hr className="rule" />

      <div className="card-title-rule">Audit trail</div>
      {events.length === 0 ? (
        <p className="muted small">No events yet.</p>
      ) : (
        <div className="card">
          <table className="table">
            <thead>
              <tr>
                <th>When</th>
                <th>Event</th>
                <th>Who</th>
                <th>Subject</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {events.map((event, index) => (
                <tr key={index}>
                  <td className="muted small" style={{ whiteSpace: 'nowrap' }}>
                    {new Date(event.created_at).toLocaleString()}
                  </td>
                  <td>
                    <span className="pill">{event.kind}</span>
                  </td>
                  <td className="small">{event.actor}</td>
                  <td className="mono small">{event.subject || '—'}</td>
                  <td className="muted small mono">
                    {Object.keys(event.detail).length
                      ? JSON.stringify(event.detail)
                      : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ *
 * Shell
 * ------------------------------------------------------------------ */

const TABS = [
  { id: 'register', label: 'Register' },
  { id: 'records', label: 'Records' },
  { id: 'identify', label: 'Identify' },
  { id: 'review', label: 'Review' },
  { id: 'activity', label: 'Activity' },
]

export default function App() {
  const [session, setSession] = useState(null)
  const [checking, setChecking] = useState(true)
  const [tab, setTab] = useState('register')
  const [error, setError] = useState(null)
  const [stats, setStats] = useState(null)
  // Bumped to force the records and queue to refetch after a registration or
  // a search, so no tab shows a stale list.
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

  // Ask who we are before showing anything. With sign-in switched off the
  // server answers without a token, and the login screen is skipped entirely
  // — a prototype should not make someone type credentials that mean nothing.
  useEffect(() => {
    api
      .me()
      .then(setSession)
      .catch(() => setSession(null))
      .finally(() => setChecking(false))
  }, [])

  useEffect(() => {
    if (!session) return
    api.stats().then(setStats).catch(() => setStats(null))
  }, [session, version, tab])

  if (checking) {
    return (
      <div className="app">
        <p className="muted small" style={{ paddingTop: 32 }}>
          Loading…
        </p>
      </div>
    )
  }

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

  const bump = () => setVersion((value) => value + 1)

  return (
    <div className="app">
      <header className="masthead">
        <div className="wordmark">
          <h1>Faceless FRS</h1>
          <span className="tagline">Face · Gait · Appearance</span>
        </div>
        <div className="session">
          {/* One line, and one sentence. The name and the words "signed in"
              used to be split by a <br>, which read out as
              "Demo Operatorsigned in" with no pause and no space. */}
          <span className="session-who">
            {session.demo_mode ? (
              <>
                <strong>Demo</strong> — sign-in off
              </>
            ) : (
              <>
                Signed in as{' '}
                <strong>{session.display_name || session.username}</strong>
              </>
            )}
          </span>
          {!session.demo_mode && (
            <button className="btn btn-quiet btn-small" onClick={signOut}>
              Sign out
            </button>
          )}
        </div>
      </header>

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

      {stats?.warnings?.length > 0 && (
        <div className="banner">
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
          {error} <span className="muted">(click to dismiss)</span>
        </div>
      )}

      <main>
        {tab === 'register' && <Register onError={onError} onRegistered={bump} />}
        {tab === 'records' && <Records key={version} onError={onError} />}
        {tab === 'identify' && <Identify onError={onError} onScanned={bump} />}
        {tab === 'review' && (
          <ReviewQueue
            key={version}
            demoMode={session.demo_mode}
            onError={onError}
          />
        )}
        {tab === 'activity' && <Activity key={version} onError={onError} />}
      </main>

      <p className="notice" style={{ marginTop: 40 }}>
        This system does not act on its own. Every candidate is a suggestion for
        a person to confirm or reject, the reasoning behind each one is shown so
        it can be judged rather than trusted, and whatever you decide is
        recorded{' '}
        {session.demo_mode
          ? 'against a shared demo operator — this build cannot tell who you are.'
          : 'against your account.'}
      </p>
    </div>
  )
}
