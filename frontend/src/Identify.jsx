import { useState } from 'react'
import { api, MODALITIES, appearanceShare, waitForJob } from './api'
import PersonShot from './PersonShot'

/**
 * Identify people in footage.
 *
 * The result of a scan is a list of *candidates*, not a list of matches, and
 * the wording here holds that line: the system produced a suggestion and
 * somebody still has to decide. Each one shows the crop it was taken from
 * beside the registration photo, because "is this the same person" is a
 * question about two pictures, and answering it from a similarity score is
 * not the same thing at all.
 */

const humanSize = (bytes) =>
  bytes > 1024 * 1024
    ? `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    : `${Math.max(1, Math.round(bytes / 1024))} KB`

function WeightBar({ weights }) {
  const entries = Object.entries(weights || {}).filter(([, weight]) => weight > 0)
  if (!entries.length) return null

  const total = entries.reduce((sum, [, weight]) => sum + weight, 0) || 1
  const ordered = [...entries].sort((a, b) => b[1] - a[1])

  return (
    <div className="breakdown">
      <div className="breakdown-bar">
        {ordered.map(([modality, weight]) => (
          <div
            key={modality}
            className="breakdown-segment"
            style={{
              width: `${(weight / total) * 100}%`,
              background: MODALITIES[modality]?.tone || '#999',
            }}
          />
        ))}
      </div>
      <div className="breakdown-legend">
        {ordered.map(([modality, weight]) => (
          <span key={modality}>
            <span
              className="swatch"
              style={{ background: MODALITIES[modality]?.tone }}
            />
            {MODALITIES[modality]?.label || modality}{' '}
            <span className="num">{Math.round((weight / total) * 100)}%</span>
          </span>
        ))}
      </div>
    </div>
  )
}

/** One candidate found in the footage. */
function Candidate({ finding }) {
  const appearance = appearanceShare(finding.weights || {})

  return (
    <article className="card">
      <header className="card-head">
        <div>
          <h3>{finding.display_name}</h3>
          <p className="muted small">
            <span className="mono">{finding.person_id}</span> · {finding.video} ·{' '}
            <span className="num">{finding.timestamp_s}s</span> into the clip ·
            frame <span className="num">{finding.frame_index}</span> · track{' '}
            <span className="num">{finding.track_id}</span>
          </p>
        </div>
        <div className="score">
          <span className="score-value">{finding.score.toFixed(3)}</span>
          <span className="score-label">score</span>
        </div>
      </header>

      <div className="evidence">
        <figure className="evidence-pane">
          <figcaption>Found in this footage</figcaption>
          <PersonShot
            key={`seen-${finding.decision_id}`}
            alt="The person detected in the uploaded footage"
            missing="No image was captured for this sighting."
            fetcher={
              finding.decision_id
                ? () => api.evidenceImage(finding.decision_id)
                : null
            }
          />
        </figure>
        <figure className="evidence-pane">
          <figcaption>Registered as {finding.display_name}</figcaption>
          <PersonShot
            key={`ref-${finding.person_id}`}
            alt={`Registration photo for ${finding.display_name}`}
            missing="No registration photo stored."
            fetcher={() => api.referenceImage(finding.person_id)}
          />
        </figure>
      </div>

      <WeightBar weights={finding.weights} />

      {appearance >= 0.5 && (
        <p className="caution">
          <strong>
            {Math.round(appearance * 100)}% of this rests on appearance — build
            and clothing.
          </strong>{' '}
          That is the weakest of the three signals and it goes stale as people
          change clothes. Look at the faces, not the score.
        </p>
      )}

      {finding.not_compared && Object.keys(finding.not_compared).length > 0 && (
        <p className="caution">
          Some signals were present but could not be compared:{' '}
          {Object.entries(finding.not_compared)
            .map(
              ([modality, reason]) =>
                `${MODALITIES[modality]?.label || modality} — ${reason}`,
            )
            .join('; ')}
        </p>
      )}

      <p className="muted small">
        {finding.decision_id
          ? `Queued for review as decision #${finding.decision_id}. Nothing has happened
             yet — it stays a suggestion until someone confirms or rejects it.`
          : 'Not recorded — this was a preview run.'}
      </p>
    </article>
  )
}

export default function Identify({ onError, onScanned }) {
  const [files, setFiles] = useState([])
  const [cameraId, setCameraId] = useState('')
  const [over, setOver] = useState(false)
  const [job, setJob] = useState(null)
  const [result, setResult] = useState(null)

  const busy = Boolean(job)

  const addFiles = (list) => {
    const incoming = Array.from(list || [])
    if (incoming.length) {
      setFiles((current) => [...current, ...incoming])
      setResult(null)
    }
  }

  const submit = async (event) => {
    event.preventDefault()
    if (!files.length) {
      onError('Choose footage to search.')
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

  const findings = result?.findings || []

  return (
    <div className="division">
      <div className="division-head">
        <h2>Identify</h2>
        <p>
          Upload footage — CCTV, a clip, a photograph — and every person in it
          is detected, tracked, and compared against the watchlist. What comes
          back is a list of candidates for you to judge, with the picture each
          one was taken from.
        </p>
      </div>

      <form className="card" onSubmit={submit}>
        <div className="card-title-rule">Footage to search</div>

        <label
          className={`dropzone ${over ? 'over' : ''}`}
          onDragOver={(event) => {
            event.preventDefault()
            setOver(true)
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(event) => {
            event.preventDefault()
            setOver(false)
            addFiles(event.dataTransfer.files)
          }}
        >
          <input
            type="file"
            multiple
            accept="image/*,video/*"
            disabled={busy}
            onChange={(event) => {
              addFiles(event.target.files)
              event.target.value = ''
            }}
          />
          <div className="dropzone-label">Drop footage here, or click to choose</div>
          <div className="dropzone-hint">
            Video or photographs · several files at once is fine
          </div>
        </label>

        {files.length > 0 && (
          <ul className="filelist">
            {files.map((file, index) => (
              <li key={`${file.name}-${index}`}>
                <span className="filename mono">{file.name}</span>
                <span className="filesize">{humanSize(file.size)}</span>
                <button
                  type="button"
                  className="btn-icon"
                  title="Remove"
                  disabled={busy}
                  onClick={() =>
                    setFiles((current) =>
                      current.filter((_, position) => position !== index),
                    )
                  }
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
        )}

        <div style={{ maxWidth: 260, marginTop: 16 }}>
          <label className="stacked">
            <span>
              Camera or location <span className="hint">— optional</span>
            </span>
            <input
              type="text"
              value={cameraId}
              onChange={(event) => setCameraId(event.target.value)}
              placeholder="north-entrance"
              disabled={busy}
            />
          </label>
        </div>

        <div className="actions actions-end" style={{ marginTop: 18 }}>
          <button className="btn btn-primary" type="submit" disabled={busy || !files.length}>
            {busy ? 'Searching…' : 'Search this footage'}
          </button>
        </div>

        {job && (
          <>
            <div className="progress">
              <div
                className="progress-bar"
                style={{ width: `${Math.round((job.progress || 0) * 100)}%` }}
              />
            </div>
            <p className="muted small">{job.message || 'working…'}</p>
          </>
        )}
      </form>

      {result && (
        <>
          <hr className="rule" />
          <div className="division-head">
            <h2>
              {findings.length === 0
                ? 'Nobody from the watchlist was found'
                : `${findings.length} candidate${findings.length > 1 ? 's' : ''}`}
            </h2>
            <p>
              {(result.videos || []).length} file(s) searched
              {result.camera_id ? ` from ${result.camera_id}` : ''} at a
              threshold of <span className="num">{result.threshold?.toFixed(2)}</span>.
              {findings.length > 0 &&
                ' Each is a suggestion, not a finding. Compare the two pictures before you decide.'}
            </p>
          </div>

          {findings.length === 0 ? (
            <div className="empty">
              <h3>No candidates</h3>
              <p>
                Nobody in this footage scored above the threshold. That is not
                proof they were absent — a person can be present and unmatched
                if their face is hidden, they never walk, or they are simply not
                registered.
              </p>
            </div>
          ) : (
            findings.map((finding, index) => (
              <Candidate
                key={finding.decision_id || `${finding.track_id}-${index}`}
                finding={finding}
              />
            ))
          )}
        </>
      )}
    </div>
  )
}
