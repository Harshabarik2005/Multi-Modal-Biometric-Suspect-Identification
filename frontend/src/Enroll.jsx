import { useCallback, useEffect, useRef, useState } from 'react'
import { api, MODALITIES, waitForJob } from './api'

/**
 * What each signal needs from an upload.
 *
 * Shown before anything is uploaded, because the three modalities have
 * genuinely different requirements and the difference is not obvious. Someone
 * who uploads five good photographs has enrolled a perfectly usable face
 * profile and *no gait profile at all* — a still image contains no gait
 * information — and they should learn that here rather than discover it later.
 */
const REQUIREMENTS = [
  {
    modality: 'face',
    accepts: 'Photos or video',
    needs: 'Face visible and roughly front-on. Several angles beat one.',
    watch: 'A distant or blurred face will not enrol — it needs the pixels.',
  },
  {
    modality: 'gait',
    accepts: 'Video only — of them WALKING',
    needs: 'About two seconds or more of walking, whole body in frame.',
    watch:
      'Photos give nothing. Standing still or turning on the spot gives nothing. ' +
      'A side-on view is best.',
  },
  {
    modality: 'reid',
    accepts: 'Photos or video',
    needs: 'Whole body in frame.',
    watch:
      'Largely describes clothing, so it goes stale. Re-enrol if it needs to ' +
      'stay current.',
  },
]

function RequirementsCard() {
  return (
    <div className="card">
      <h3>What to record</h3>
      <p className="muted small">
        The three signals need different things. You do not have to provide all
        of them, but a profile missing a signal simply cannot match on it.
      </p>
      <table className="table">
        <thead>
          <tr>
            <th>Signal</th>
            <th>Accepts</th>
            <th>Needs</th>
          </tr>
        </thead>
        <tbody>
          {REQUIREMENTS.map((entry) => {
            const meta = MODALITIES[entry.modality] || {}
            return (
              <tr key={entry.modality}>
                <td>
                  <span className="swatch" style={{ background: meta.colour }} />
                  {meta.label || entry.modality}
                </td>
                <td className="small">{entry.accepts}</td>
                <td className="small">
                  {entry.needs}
                  <div className="muted" style={{ marginTop: 2 }}>
                    {entry.watch}
                  </div>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="caution" style={{ marginTop: 12 }}>
        <strong>One person per enrolment.</strong> If several people are in the
        footage the longest-visible one is used. A reference built from two
        people produces confident wrong matches from then on.
      </p>
    </div>
  )
}

/** Record enrolment footage straight from the webcam. */
function WebcamRecorder({ onRecorded, disabled }) {
  const videoRef = useRef(null)
  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const streamRef = useRef(null)

  const [active, setActive] = useState(false)
  const [recording, setRecording] = useState(false)
  const [seconds, setSeconds] = useState(0)
  const [error, setError] = useState('')

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    setActive(false)
    setRecording(false)
  }, [])

  // Releasing the camera on unmount matters: a webcam left streaming keeps its
  // light on, which looks exactly like covert recording.
  useEffect(() => stop, [stop])

  useEffect(() => {
    if (!recording) return undefined
    const timer = setInterval(() => setSeconds((value) => value + 1), 1000)
    return () => clearInterval(timer)
  }, [recording])

  const start = async () => {
    setError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) videoRef.current.srcObject = stream
      setActive(true)
    } catch (exception) {
      setError(
        `Could not open the camera: ${exception.message}. Browsers only allow ` +
          'camera access over https or on localhost.',
      )
    }
  }

  const beginRecording = () => {
    if (!streamRef.current) return
    chunksRef.current = []
    setSeconds(0)

    const recorder = new MediaRecorder(streamRef.current, {
      mimeType: MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
        ? 'video/webm;codecs=vp9'
        : 'video/webm',
    })
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data)
    }
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: 'video/webm' })
      const stamp = new Date().toISOString().replace(/[:.]/g, '-')
      onRecorded(new File([blob], `recording-${stamp}.webm`, { type: 'video/webm' }))
    }
    recorder.start()
    recorderRef.current = recorder
    setRecording(true)
  }

  const endRecording = () => {
    recorderRef.current?.stop()
    setRecording(false)
  }

  return (
    <div className="card">
      <h3>Record from the camera</h3>
      {!active && (
        <>
          <p className="muted small">
            To capture gait as well as face, walk across the frame from one side
            to the other for a few seconds, with your whole body visible.
          </p>
          <button className="btn btn-confirm" onClick={start} disabled={disabled}>
            Open camera
          </button>
        </>
      )}

      {error && <p className="caution">{error}</p>}

      {active && (
        <>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="preview-video"
          />
          <div className="review-actions" style={{ marginTop: 10 }}>
            <button className="btn btn-reject" onClick={stop}>
              Close camera
            </button>
            {recording ? (
              <button className="btn btn-reject" onClick={endRecording}>
                Stop ({seconds}s)
              </button>
            ) : (
              <button className="btn btn-confirm" onClick={beginRecording}>
                Start recording
              </button>
            )}
          </div>
          {recording && seconds < 3 && (
            <p className="muted small" style={{ marginTop: 8 }}>
              Keep going — gait needs roughly two seconds of walking at minimum.
            </p>
          )}
        </>
      )}
    </div>
  )
}

function ReadinessReport({ preview }) {
  if (!preview) return null

  return (
    <div className="card">
      <h3>What this upload can support</h3>
      <p className="muted small">
        {preview.videos} video(s), {preview.images} photo(s) —{' '}
        {preview.observations} usable view(s) of the person found.
      </p>

      {preview.warning && <p className="caution">{preview.warning}</p>}

      <table className="table">
        <tbody>
          {preview.readiness.map((entry) => {
            const meta = MODALITIES[entry.modality] || {}
            return (
              <tr key={entry.modality}>
                <td style={{ width: 90 }}>
                  <span className="swatch" style={{ background: meta.colour }} />
                  {meta.label || entry.modality}
                </td>
                <td style={{ width: 90 }}>
                  <span className={entry.ready ? 'ok' : 'not-ok'}>
                    {entry.ready ? 'ready' : 'not ready'}
                  </span>
                </td>
                <td className="small">
                  {entry.reason}
                  {!entry.ready && (
                    <div className="muted" style={{ marginTop: 2 }}>
                      {entry.requirement}
                    </div>
                  )}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>

      {preview.rejected.length > 0 && (
        <>
          <p className="muted small" style={{ marginTop: 10 }}>
            Files that gave nothing:
          </p>
          <ul className="muted small">
            {preview.rejected.map((entry, index) => (
              <li key={index}>
                <code>{entry.file}</code> — {entry.reason}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  )
}

export default function Enroll({ operator, onError, onEnrolled }) {
  const [files, setFiles] = useState([])
  const [personId, setPersonId] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [notes, setNotes] = useState('')
  const [replace, setReplace] = useState(false)

  const [preview, setPreview] = useState(null)
  const [checking, setChecking] = useState(false)
  const [job, setJob] = useState(null)
  const [done, setDone] = useState(null)

  const addFiles = (incoming) => {
    setFiles((current) => [...current, ...incoming])
    setPreview(null)
    setDone(null)
  }

  const removeFile = (index) =>
    setFiles((current) => current.filter((_, position) => position !== index))

  const check = async () => {
    if (!files.length) {
      onError('Add some photos or video first.')
      return
    }
    setChecking(true)
    setPreview(null)
    try {
      setPreview(await api.previewEnrollment(files))
    } catch (exception) {
      onError(exception.message)
    } finally {
      setChecking(false)
    }
  }

  const submit = async () => {
    if (!personId.trim() || !displayName.trim()) {
      onError('An ID and a name are both required.')
      return
    }
    if (!files.length) {
      onError('Add some photos or video first.')
      return
    }

    setDone(null)
    try {
      const started = await api.enroll({
        files,
        personId: personId.trim(),
        displayName: displayName.trim(),
        notes,
        operator,
        replace,
      })
      setJob(started)
      const finished = await waitForJob(started.id, setJob)
      setJob(null)

      if (finished.status === 'failed') {
        onError(finished.error || 'Enrolment failed.')
        return
      }
      setDone(finished.result)
      setFiles([])
      setPreview(null)
      onEnrolled?.()
    } catch (exception) {
      setJob(null)
      onError(exception.message)
    }
  }

  const busy = Boolean(job)

  return (
    <>
      <RequirementsCard />

      <div className="card">
        <h3>Who is this?</h3>
        <div className="field-grid">
          <label>
            <span>ID</span>
            <input
              type="text"
              placeholder="e.g. ravi"
              value={personId}
              onChange={(event) => setPersonId(event.target.value)}
              disabled={busy}
            />
          </label>
          <label>
            <span>Name</span>
            <input
              type="text"
              placeholder="e.g. Ravi Kumar"
              value={displayName}
              onChange={(event) => setDisplayName(event.target.value)}
              disabled={busy}
            />
          </label>
        </div>
        <label className="stacked">
          <span>Notes (optional)</span>
          <input
            type="text"
            placeholder="Why this person is on the watchlist"
            value={notes}
            onChange={(event) => setNotes(event.target.value)}
            disabled={busy}
          />
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={replace}
            onChange={(event) => setReplace(event.target.checked)}
            disabled={busy}
          />
          <span>Replace an existing profile with this ID</span>
        </label>
      </div>

      <div className="card">
        <h3>Photos and video</h3>
        <input
          type="file"
          multiple
          accept="image/*,video/*"
          onChange={(event) => {
            addFiles(Array.from(event.target.files || []))
            event.target.value = ''
          }}
          disabled={busy}
        />

        {files.length > 0 && (
          <table className="table" style={{ marginTop: 10 }}>
            <tbody>
              {files.map((file, index) => (
                <tr key={`${file.name}-${index}`}>
                  <td className="mono small">{file.name}</td>
                  <td className="muted small" style={{ width: 90 }}>
                    {(file.size / (1024 * 1024)).toFixed(1)} MB
                  </td>
                  <td style={{ width: 70 }}>
                    <button
                      className="btn btn-reject btn-small"
                      onClick={() => removeFile(index)}
                      disabled={busy}
                    >
                      remove
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        <div className="review-actions" style={{ marginTop: 12 }}>
          <button
            className="btn btn-reject"
            onClick={check}
            disabled={busy || checking || !files.length}
          >
            {checking ? 'Checking…' : 'Check what this covers'}
          </button>
          <button className="btn btn-confirm" onClick={submit} disabled={busy}>
            {busy ? 'Enrolling…' : 'Enrol this person'}
          </button>
        </div>
        <p className="muted small" style={{ marginTop: 8 }}>
          Checking first takes a few seconds and tells you which signals your
          files can support. Enrolling runs all three models and takes longer.
        </p>
      </div>

      <WebcamRecorder onRecorded={(file) => addFiles([file])} disabled={busy} />

      <ReadinessReport preview={preview} />

      {job && (
        <div className="card">
          <h3>Enrolling…</h3>
          <div className="progress">
            <div
              className="progress-bar"
              style={{ width: `${Math.round((job.progress || 0) * 100)}%` }}
            />
          </div>
          <p className="muted small">{job.message}</p>
        </div>
      )}

      {done && (
        <div className="card">
          <h3>Enrolled {done.display_name}</h3>
          <p className="muted small">
            From {done.videos} video(s) and {done.images} photo(s),{' '}
            {done.observations} usable view(s).
          </p>
          <p>
            Stored:{' '}
            {done.stored.map((modality) => {
              const meta = MODALITIES[modality] || {}
              return (
                <span
                  key={modality}
                  className="pill"
                  style={{ borderColor: meta.colour }}
                >
                  {meta.label || modality}
                </span>
              )
            })}
          </p>
          {done.skipped.length > 0 && (
            <>
              <p className="caution">
                Not stored — this profile cannot match on these signals:
              </p>
              <ul className="muted small">
                {done.skipped.map((entry) => (
                  <li key={entry.modality}>
                    <strong>{MODALITIES[entry.modality]?.label || entry.modality}</strong>{' '}
                    — {entry.reason}
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </>
  )
}
