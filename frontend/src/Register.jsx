import { useMemo, useState } from 'react'
import { api, MODALITIES, waitForJob } from './api'
import Recorder from './Recorder'

/**
 * Registering a person.
 *
 * Footage is collected section by section rather than as one pile of files,
 * because the three signals want genuinely different recordings and the
 * difference is not guessable. Somebody who uploads five excellent portraits
 * has enrolled a good face profile and *no gait profile at all* — a still
 * image contains no gait — and a single "add files" box gives them no way to
 * discover that until it is too late to re-record.
 *
 * Each section carries its own angle slots, because one view of a face is a
 * much weaker reference than three, and the interface should ask for the
 * second and third rather than accept the first and call it done.
 *
 * All the files are sent together: the backend runs every branch over every
 * observation, so the sections are about eliciting the right footage, not
 * about routing it. What each section does buy is *specific* feedback — check
 * a gait section and you learn whether that walk actually produced gait,
 * rather than whether the upload as a whole did.
 */
const SECTIONS = [
  {
    key: 'face',
    title: 'Face',
    subtitle: 'Photos or video · the identity signal, and the strongest one',
    prompt: 'Look at the camera and turn your head slowly left, then right.',
    angles: ['Front', 'Left profile', 'Right profile'],
    needs: 'Face visible and roughly front-on, close enough to have real pixels.',
    watch: 'A distant or motion-blurred face will not enrol. Several angles beat one good one.',
    required: false,
  },
  {
    key: 'gait',
    title: 'Gait',
    subtitle: 'Video only · of them WALKING',
    prompt: 'Walk across the frame from one side to the other, whole body visible.',
    angles: ['Side-on walk', 'Walking towards camera'],
    needs: 'Two seconds or more of continuous walking, whole body in frame.',
    watch:
      'Photos give nothing at all. Standing still or turning on the spot gives ' +
      'nothing. A side-on view carries the most.',
    required: false,
  },
  {
    key: 'reid',
    title: 'Appearance',
    subtitle: 'Photos or video · build and clothing',
    prompt: 'Stand a few steps back so your whole body is in frame.',
    angles: ['Full body'],
    needs: 'Whole body in frame, head to feet.',
    watch:
      'Largely describes what they are wearing, so it goes stale within days. ' +
      'Re-register if it has to stay current.',
    required: false,
  },
]

const newAngle = (label) => ({
  id: `${label}-${Math.random().toString(36).slice(2, 8)}`,
  label,
  files: [],
})

const initialSections = () =>
  Object.fromEntries(
    SECTIONS.map((section) => [
      section.key,
      section.angles.map((label) => newAngle(label)),
    ]),
  )

const humanSize = (bytes) =>
  bytes > 1024 * 1024
    ? `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    : `${Math.max(1, Math.round(bytes / 1024))} KB`

/**
 * Rename a file to say where it came from.
 *
 * The server reports rejected files by name, and "IMG_0007.jpg gave nothing"
 * is not actionable while "gait--side-on-walk--IMG_0007.jpg gave nothing"
 * tells you which recording to redo. It also lands in the enrolment's audit
 * record, so the provenance of a stored template survives the session.
 */
function label(file, sectionKey, angleLabel) {
  const slug = (text) =>
    text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
  const name = `${slug(sectionKey)}--${slug(angleLabel)}--${file.name}`
  try {
    return new File([file], name, { type: file.type })
  } catch {
    return file // very old browsers: the plain file still works
  }
}

/* ---------------------------------------------------------------- dropzone */

function DropZone({ accept, onFiles, disabled }) {
  const [over, setOver] = useState(false)

  const take = (list) => {
    const files = Array.from(list || [])
    if (files.length) onFiles(files)
  }

  return (
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
        take(event.dataTransfer.files)
      }}
    >
      <input
        type="file"
        multiple
        accept={accept}
        disabled={disabled}
        onChange={(event) => {
          take(event.target.files)
          event.target.value = ''
        }}
      />
      <div className="dropzone-label">Drop files here, or click to choose</div>
      <div className="dropzone-hint">
        {accept.includes('image') ? 'Video or photos' : 'Video only'}
      </div>
    </label>
  )
}

/* ----------------------------------------------------------------- section */

function CaptureSection({ section, index, angles, setAngles, onError, busy }) {
  const [checking, setChecking] = useState(false)
  const [report, setReport] = useState(null)

  const files = angles.flatMap((angle) => angle.files)
  const accept = section.key === 'gait' ? 'video/*' : 'image/*,video/*'

  const update = (angleId, changes) =>
    setAngles(
      angles.map((angle) =>
        angle.id === angleId ? { ...angle, ...changes } : angle,
      ),
    )

  const addFiles = (angleId, incoming) => {
    const angle = angles.find((entry) => entry.id === angleId)
    update(angleId, { files: [...angle.files, ...incoming] })
    setReport(null)
  }

  const removeFile = (angleId, position) => {
    const angle = angles.find((entry) => entry.id === angleId)
    update(angleId, {
      files: angle.files.filter((_, index) => index !== position),
    })
    setReport(null)
  }

  /**
   * Check what this section alone produced.
   *
   * Per-section rather than per-upload on purpose: told "gait: not ready" for
   * a whole submission, you do not know which recording failed. Told it about
   * the walking video specifically, you know exactly what to record again.
   */
  const check = async () => {
    if (!files.length) {
      onError(`Add something to the ${section.title.toLowerCase()} section first.`)
      return
    }
    setChecking(true)
    setReport(null)
    try {
      const preview = await api.previewEnrollment(
        angles.flatMap((angle) =>
          angle.files.map((file) => label(file, section.key, angle.label)),
        ),
      )
      setReport(preview.readiness.find((entry) => entry.modality === section.key))
    } catch (exception) {
      onError(exception.message)
    } finally {
      setChecking(false)
    }
  }

  return (
    <section className="capture-section">
      <header className="capture-head">
        <span className="capture-index">{index + 1}</span>
        <div className="capture-title">
          <h3>{section.title}</h3>
          <p>{section.subtitle}</p>
        </div>
        <span className={`capture-status ${files.length ? 'filled' : ''}`}>
          {files.length ? `${files.length} file${files.length > 1 ? 's' : ''}` : 'empty'}
        </span>
      </header>

      <div className="capture-body">
        <p className="muted small" style={{ marginBottom: 14 }}>
          <strong style={{ color: 'var(--ink-2)' }}>{section.needs}</strong>{' '}
          {section.watch}
        </p>

        <div className="angles">
          {angles.map((angle) => (
            <div className="angle" key={angle.id}>
              <div className="angle-head">
                <input
                  type="text"
                  value={angle.label}
                  onChange={(event) => update(angle.id, { label: event.target.value })}
                  aria-label="Angle name"
                  disabled={busy}
                />
                <span className="muted small" style={{ flex: 1 }}>
                  {angle.files.length
                    ? `${angle.files.length} file${angle.files.length > 1 ? 's' : ''}`
                    : ''}
                </span>
                {angles.length > 1 && (
                  <button
                    type="button"
                    className="btn-icon"
                    title="Remove this angle"
                    onClick={() =>
                      setAngles(angles.filter((entry) => entry.id !== angle.id))
                    }
                    disabled={busy}
                  >
                    ✕
                  </button>
                )}
              </div>

              <DropZone
                accept={accept}
                disabled={busy}
                onFiles={(incoming) => addFiles(angle.id, incoming)}
              />

              {angle.files.length > 0 && (
                <ul className="filelist">
                  {angle.files.map((file, position) => (
                    <li key={`${file.name}-${position}`}>
                      <span className="filename mono">{file.name}</span>
                      <span className="filesize">{humanSize(file.size)}</span>
                      <button
                        type="button"
                        className="btn-icon"
                        title="Remove"
                        onClick={() => removeFile(angle.id, position)}
                        disabled={busy}
                      >
                        ✕
                      </button>
                    </li>
                  ))}
                </ul>
              )}

              <div style={{ marginTop: 10 }}>
                <Recorder
                  prompt={section.prompt}
                  disabled={busy}
                  onRecorded={(file) => addFiles(angle.id, [file])}
                />
              </div>
            </div>
          ))}
        </div>

        <div className="actions" style={{ marginTop: 12 }}>
          <button
            type="button"
            className="btn btn-quiet btn-small"
            onClick={() => setAngles([...angles, newAngle('Another angle')])}
            disabled={busy}
          >
            + Add angle
          </button>
          <button
            type="button"
            className="btn btn-quiet btn-small"
            onClick={check}
            disabled={busy || checking || !files.length}
          >
            {checking ? 'Checking…' : `Check ${section.title.toLowerCase()}`}
          </button>
        </div>

        {report && (
          <p className={report.ready ? 'notice' : 'caution'} style={{ marginBottom: 0 }}>
            <span className={report.ready ? 'ok' : 'not-ok'}>
              {report.ready ? 'ready' : 'not ready'}
            </span>{' '}
            {report.reason}
            {!report.ready && (
              <span className="muted"> {report.requirement}</span>
            )}
          </p>
        )}
      </div>
    </section>
  )
}

/* -------------------------------------------------------------------- page */

export default function Register({ onError, onRegistered }) {
  const [personId, setPersonId] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [notes, setNotes] = useState('')
  const [replace, setReplace] = useState(false)
  const [sections, setSections] = useState(initialSections)

  const [job, setJob] = useState(null)
  const [done, setDone] = useState(null)

  const busy = Boolean(job)

  const allFiles = useMemo(
    () =>
      SECTIONS.flatMap((section) =>
        sections[section.key].flatMap((angle) =>
          angle.files.map((file) => label(file, section.key, angle.label)),
        ),
      ),
    [sections],
  )

  const reset = () => {
    setPersonId('')
    setDisplayName('')
    setNotes('')
    setReplace(false)
    setSections(initialSections())
  }

  const submit = async (event) => {
    event.preventDefault()

    if (!personId.trim() || !displayName.trim()) {
      onError('A name and an ID are both required — a record with neither cannot be reviewed later.')
      return
    }
    if (!allFiles.length) {
      onError('Add footage to at least one section.')
      return
    }

    setDone(null)
    try {
      const started = await api.enroll({
        files: allFiles,
        personId: personId.trim(),
        displayName: displayName.trim(),
        notes,
        replace,
      })
      setJob(started)
      const finished = await waitForJob(started.id, setJob)
      setJob(null)

      if (finished.status === 'failed') {
        onError(finished.error || 'Registration failed.')
        return
      }
      setDone(finished.result)
      reset()
      onRegistered?.()
    } catch (exception) {
      setJob(null)
      onError(exception.message)
    }
  }

  return (
    <form className="division" onSubmit={submit}>
      <div className="division-head">
        <h2>Register a person</h2>
        <p>
          Record each signal separately. You do not have to provide all three,
          but a record missing a signal simply cannot match on it later — and
          you will not find that out at the moment it matters.
        </p>
      </div>

      <div className="card">
        <div className="card-title-rule">Identity</div>
        <div className="field-grid">
          <label>
            <span>Full name</span>
            <input
              type="text"
              value={displayName}
              onChange={(event) => setDisplayName(event.target.value)}
              placeholder="Ravi Kumar"
              disabled={busy}
            />
          </label>
          <label>
            <span>
              Reference ID <span className="hint">— letters, digits, . _ -</span>
            </span>
            <input
              type="text"
              value={personId}
              onChange={(event) => setPersonId(event.target.value)}
              placeholder="case-2291"
              className="mono"
              disabled={busy}
            />
          </label>
        </div>

        <label className="stacked">
          <span>
            Notes <span className="hint">— case reference, source, authority</span>
          </span>
          <textarea
            value={notes}
            onChange={(event) => setNotes(event.target.value)}
            placeholder="Why this person is on the watchlist, and on whose authority."
            disabled={busy}
          />
        </label>

        <label className="checkbox" style={{ marginTop: 14 }}>
          <input
            type="checkbox"
            checked={replace}
            onChange={(event) => setReplace(event.target.checked)}
            disabled={busy}
          />
          <span>
            Replace an existing record with this ID. Their stored signals are
            overwritten; past decisions about them are kept.
          </span>
        </label>
      </div>

      <div className="card-title-rule" style={{ marginTop: 22 }}>
        Footage
      </div>

      {SECTIONS.map((section, index) => (
        <CaptureSection
          key={section.key}
          section={section}
          index={index}
          angles={sections[section.key]}
          setAngles={(angles) =>
            setSections((current) => ({ ...current, [section.key]: angles }))
          }
          onError={onError}
          busy={busy}
        />
      ))}

      <p className="caution">
        <strong>One person per record.</strong> Where footage contains several
        people the longest-visible one is used. A reference built from two
        people produces confident wrong matches from then on, and nothing
        downstream can detect that it happened.
      </p>

      {job && (
        <div className="card">
          <div className="card-title-rule">Registering</div>
          <div className="progress">
            <div
              className="progress-bar"
              style={{ width: `${Math.round((job.progress || 0) * 100)}%` }}
            />
          </div>
          <p className="muted small">{job.message || 'working…'}</p>
        </div>
      )}

      {done && (
        <div className="card">
          <div className="card-title-rule">Registered</div>
          <h3>
            {done.display_name} <span className="mono muted">{done.person_id}</span>
          </h3>
          <p className="muted small">
            {done.observations} usable view(s) of the person across the upload.
          </p>

          <div style={{ marginTop: 10 }}>
            {done.stored.map((modality) => (
              <span className="pill" key={modality}>
                <span
                  className="swatch"
                  style={{ background: MODALITIES[modality]?.tone }}
                />
                {MODALITIES[modality]?.label || modality} stored
              </span>
            ))}
          </div>

          {done.skipped?.length > 0 && (
            <>
              <p className="muted small" style={{ marginTop: 14, marginBottom: 4 }}>
                Not stored — this record cannot match on these:
              </p>
              <table className="table">
                <tbody>
                  {done.skipped.map((entry) => (
                    <tr key={entry.modality}>
                      <td style={{ width: 110 }}>
                        {MODALITIES[entry.modality]?.label || entry.modality}
                      </td>
                      <td className="small muted">{entry.reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </div>
      )}

      <div className="actions actions-end" style={{ marginTop: 18 }}>
        <button
          type="button"
          className="btn btn-quiet"
          onClick={reset}
          disabled={busy}
        >
          Clear
        </button>
        <button className="btn btn-primary" type="submit" disabled={busy}>
          {busy ? 'Registering…' : `Register${allFiles.length ? ` — ${allFiles.length} file(s)` : ''}`}
        </button>
      </div>
    </form>
  )
}
