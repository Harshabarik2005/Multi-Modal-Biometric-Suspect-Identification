import { useState } from 'react'
import { api, MODALITIES, appearanceShare, waitForJob } from './api'

/**
 * One person found in the footage.
 *
 * Deliberately not called a "match". The system produced a candidate; whether
 * it is the person is a judgement someone still has to make, and the wording
 * here should not make that judgement sound already settled.
 */
function Finding({ finding }) {
  const weights = finding.weights || {}
  const entries = Object.entries(weights).filter(([, weight]) => weight > 0)
  const total = entries.reduce((sum, [, weight]) => sum + weight, 0) || 1
  const appearance = appearanceShare(weights)

  return (
    <article className="card">
      <header className="card-head">
        <div>
          <h3>{finding.display_name}</h3>
          <p className="muted small">
            {finding.person_id} · {finding.video} · {finding.timestamp_s}s into
            the clip (frame {finding.frame_index}) · track {finding.track_id}
          </p>
        </div>
        <div className="score">
          <span className="score-value">{finding.score.toFixed(3)}</span>
          <span className="muted small">score</span>
        </div>
      </header>

      {entries.length > 0 && (
        <>
          <div className="breakdown-bar">
            {entries.map(([modality, weight]) => (
              <div
                key={modality}
                className="breakdown-segment"
                style={{
                  width: `${(weight / total) * 100}%`,
                  background: MODALITIES[modality]?.colour || '#64748b',
                }}
                title={`${MODALITIES[modality]?.label || modality}: ${Math.round(
                  (weight / total) * 100,
                )}%`}
              />
            ))}
          </div>
          <p className="muted small">
            Driven by{' '}
            {entries
              .sort((a, b) => b[1] - a[1])
              .map(
                ([modality, weight]) =>
                  `${MODALITIES[modality]?.label || modality} ${Math.round(
                    (weight / total) * 100,
                  )}%`,
              )
              .join(', ')}
          </p>
        </>
      )}

      {appearance >= 0.5 && (
        <p className="caution">
          <strong>
            {Math.round(appearance * 100)}% of this rests on appearance (build
            and clothing).
          </strong>{' '}
          That is the weakest of the three signals and it goes stale as people
          change clothes.
        </p>
      )}

      <p className="muted small">
        {finding.decision_id
          ? `Queued for review as decision #${finding.decision_id}. Nothing has
             happened yet — it needs a person to confirm or reject it.`
          : 'Not recorded — this was a preview run.'}
      </p>
    </article>
  )
}

export default function Scan({ onError, onScanned }) {
  const [files, setFiles] = useState([])
  const [cameraId, setCameraId] = useState('')
  const [job, setJob] = useState(null)
  const [result, setResult] = useState(null)

  const submit = async () => {
    if (!files.length) {
      onError('Choose some footage to scan.')
      return
    }
    setResult(null)
    try {
      const started = await api.scan({ files, cameraId })
      setJob(started)
      const finished = await waitForJob(started.id, setJob)
      setJob(null)

      if (finished.status === 'failed') {
        onError(finished.error || 'The scan failed.')
        return
      }
      setResult(finished.result)
      setFiles([])
      onScanned?.()
    } catch (exception) {
      setJob(null)
      onError(exception.message)
    }
  }

  const busy = Boolean(job)

  return (
    <>
      <div className="card">
        <h3>Search footage</h3>
        <p className="muted small">
          Upload CCTV or any other video. Everyone in it is detected and tracked,
          then compared against the watchlist. Anyone found is queued for review
          — nothing is acted on automatically.
        </p>

        <input
          type="file"
          multiple
          accept="video/*"
          onChange={(event) => {
            setFiles(Array.from(event.target.files || []))
            setResult(null)
          }}
          disabled={busy}
        />

        <label className="stacked" style={{ marginTop: 10 }}>
          <span>Camera label (optional)</span>
          <input
            type="text"
            placeholder="e.g. lobby-north"
            value={cameraId}
            onChange={(event) => setCameraId(event.target.value)}
            disabled={busy}
          />
        </label>

        {files.length > 0 && (
          <p className="muted small" style={{ marginTop: 8 }}>
            {files.length} file(s):{' '}
            {files.map((file) => file.name).join(', ')}
          </p>
        )}

        <div className="review-actions" style={{ marginTop: 12 }}>
          <button className="btn btn-confirm" onClick={submit} disabled={busy}>
            {busy ? 'Scanning…' : 'Scan for watchlist people'}
          </button>
        </div>
        <p className="muted small" style={{ marginTop: 8 }}>
          Scanning runs detection, tracking and three recognition models over
          every frame, so expect roughly a minute per thousand frames.
        </p>
      </div>

      {job && (
        <div className="card">
          <h3>Scanning…</h3>
          <div className="progress">
            <div
              className="progress-bar"
              style={{ width: `${Math.round((job.progress || 0) * 100)}%` }}
            />
          </div>
          <p className="muted small">{job.message}</p>
        </div>
      )}

      {result && (
        <>
          <div className="card">
            <h3>
              {result.findings.length === 0
                ? 'Nobody on the watchlist was found'
                : `${result.findings.length} person(s) found`}
            </h3>
            <p className="muted small">
              Searched {result.videos.join(', ')}
              {result.camera_id && ` · camera ${result.camera_id}`} · threshold{' '}
              {result.threshold.toFixed(2)}
            </p>
            {result.findings.length === 0 && (
              <p className="muted small">
                That means nobody scored above the threshold — not that nobody
                in the footage is on the watchlist. Someone whose face was never
                visible may simply not have produced a strong enough signal.
              </p>
            )}
          </div>

          {result.findings.map((finding, index) => (
            <Finding key={index} finding={finding} />
          ))}

          {result.findings.length > 0 && (
            <p className="notice">
              These are candidates, not identifications. Open the review queue to
              confirm or reject each one.
            </p>
          )}
        </>
      )}
    </>
  )
}
