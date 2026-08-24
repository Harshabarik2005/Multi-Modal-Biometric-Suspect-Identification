import { useMemo, useState } from 'react'
import { api, MODALITIES, waitForJob } from './api'
import Recorder from './Recorder'

/**
 * Registering a person.
 *
 * Footage is collected section by section rather than as one pile of files,
 * because the three signals want genuinely different recordings and the
 * difference is not guessable. Somebody who uploads five excellent portraits
 * has registered a good face profile and *no gait profile at all* — a still
 * image contains no gait — and a single "add files" box gives them no way to
 * discover that until it is too late to re-record.
 *
 * Each section offers the same choice two ways round:
 *
 *   **One video** that sweeps through everything the signal needs, which is
 *   the better input and not obviously so. Face enrolment keeps the best 40%
 *   of at least ten frames, so a slow head turn hands the branch a spread to
 *   choose from where three held poses hand it three. It is also one recording
 *   instead of three uploads.
 *
 *   **Separate photos**, for when video is not available — someone working
 *   from a case file has stills and nothing else.
 *
 * Both are collected if both are given; the backend runs every branch over
 * every observation, so nothing is thrown away for being in the wrong slot.
 * The mode switch changes what is asked for, not what is accepted.
 */
const SECTIONS = [
  {
    key: 'face',
    title: 'Face',
    subtitle: 'The strongest signal, and the fussiest',
    // ArcFace scores a frame by det_score x frontality x resolution.
    // Frontality is full credit to face.frontal_yaw_deg (20°) and reaches ZERO
    // at face.max_yaw_deg (65°) -- a true profile contributes nothing, so
    // asking for one wastes the operator's time. Resolution caps at
    // face.ideal_face_height (112px of face, not of person).
    spec: 'Face at least 112 px tall in frame, turned no more than 20° off centre for full credit.',
    limit:
      'Past 65° off centre a frame counts for nothing — a true side profile and ' +
      'the back of a head are worth zero, so there is no point recording them.',
    video: {
      instruction:
        'Look straight at the camera, then turn your head slowly to the left ' +
        'and back, then slowly to the right and back. Ten seconds is plenty. ' +
        'Stop turning well before your nose leaves the frame.',
      covers: [
        'Looking straight at the camera',
        'Head turned a little to the left',
        'Head turned a little to the right',
      ],
      // enroll_top_fraction 0.4 over enroll_min_frames 10: the best 40% of at
      // least ten frames survive, so a sweep beats a held pose.
      why: 'A slow turn gives the model dozens of frames to pick its best from. Three photos give it three.',
    },
    photos: [
      {
        label: 'Looking straight at the camera',
        guide: 'Eyes level, face filling a good part of the frame.',
      },
      {
        label: 'Head turned a little to the left',
        guide: 'About a quarter turn — you should still see one whole ear. Not a side-on profile.',
      },
      {
        label: 'Head turned a little to the right',
        guide: 'The same, the other way.',
      },
    ],
  },
  {
    key: 'gait',
    title: 'Gait',
    subtitle: 'How they walk — video only',
    // The cadence signal is the width of the LOWER THIRD of the silhouette --
    // the legs. Only a side-on walk swings that width; walking at the camera
    // keeps the legs inside the body outline and fails min_swing_ratio. Full
    // quality needs cycles >= 2 (cycle_factor = min(1, cycles / 2)), and every
    // clipped frame -- body touching the top or bottom edge -- is subtracted
    // through the `unclipped` factor.
    spec: 'Four seconds or more of unbroken walking, seen from the side, whole body inside the frame.',
    limit:
      'Gait is read from how far the legs swing apart, so it only works side on. ' +
      'Walking towards or away from the camera gives almost nothing. Keep a gap ' +
      'above the head and below the feet — a body touching the edge of frame is ' +
      'marked down, not just cropped.',
    videoOnly: true,
    video: {
      instruction:
        'Stand side on to the camera and walk across the frame at a normal ' +
        'pace, then turn round and walk back. Keep the whole body visible the ' +
        'whole way across.',
      covers: ['Walking one way, side on', 'Walking back the other way'],
      why: 'A photograph contains no gait at all, and neither does standing still or turning on the spot.',
    },
    photos: [],
  },
  {
    key: 'reid',
    title: 'Appearance',
    subtitle: 'Build and clothing, from any side',
    // OSNet quality is resolution x aspect-ratio x detection confidence. It is
    // deliberately pose-blind -- a back view is fully usable, which is the
    // whole point of the modality when a face is never visible. Resolution
    // caps at reid.ideal_box_height (192px of person); the aspect score peaks
    // at reid.ideal_aspect 2.5, a normal standing figure.
    spec: 'Whole body, head to feet, at least 192 px tall. Standing normally, arms down.',
    limit:
      'This one does not care which way they face — a view from behind is as ' +
      'good as one from the front, and is often all a real camera gets. It does ' +
      'care that the whole body is in frame: half a person, or two people in one ' +
      'box, describes neither. It also reads clothing more than the person, so ' +
      'it goes stale within days.',
    video: {
      instruction:
        'Stand a few steps back so the whole body is in frame, arms at your ' +
        'sides, and turn slowly all the way round.',
      covers: ['From the front', 'From behind', 'From each side'],
      why: 'One turn covers every side, which is what this signal wants — it is the one that has to work when no face is ever visible.',
    },
    photos: [
      { label: 'From the front', guide: 'Head to feet, standing straight, arms down.' },
      {
        label: 'From behind',
        guide: 'Genuinely worth having — most cameras never get a face.',
      },
      { label: 'From the side', guide: 'Fills in the shape from the third direction.' },
    ],
  },
]

const newAngle = ({ label, guide = '' }) => ({
  id: `${label}-${Math.random().toString(36).slice(2, 8)}`,
  label,
  guide,
  files: [],
})

const initialSections = () =>
  Object.fromEntries(
    SECTIONS.map((section) => [
      section.key,
      {
        mode: 'video',
        video: [newAngle({ label: 'Video' })],
        photos: section.photos.map(newAngle),
      },
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
 * is not actionable while "gait--walking-one-way--IMG_0007.jpg gave nothing"
 * names the recording to redo. It also lands in the enrolment's audit record,
 * so the provenance of a stored template survives the session.
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

const filesOf = (state) => [...state.video, ...state.photos].flatMap((a) => a.files)

const labelledFiles = (sectionKey, state) =>
  [...state.video, ...state.photos].flatMap((angle) =>
    angle.files.map((file) => label(file, sectionKey, angle.label)),
  )

/* ---------------------------------------------------------------- dropzone */

function DropZone({ accept, onFiles, disabled, prompt, hint }) {
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
      <span className="dropzone-label">{prompt}</span>
      <span className="dropzone-hint">{hint}</span>
    </label>
  )
}

function FileList({ files, onRemove, disabled }) {
  if (!files.length) return null
  return (
    <ul className="filelist">
      {files.map((file, position) => (
        <li key={`${file.name}-${position}`}>
          <span className="filename mono">{file.name}</span>
          <span className="filesize">{humanSize(file.size)}</span>
          <button
            type="button"
            className="btn-icon"
            aria-label={`Remove ${file.name}`}
            title="Remove"
            onClick={() => onRemove(position)}
            disabled={disabled}
          >
            ✕
          </button>
        </li>
      ))}
    </ul>
  )
}

/* ----------------------------------------------------------------- section */

function CaptureSection({ section, index, state, setState, onError, busy }) {
  const [checking, setChecking] = useState(false)
  const [report, setReport] = useState(null)

  const files = filesOf(state)
  const accept = section.key === 'gait' ? 'video/*' : 'image/*,video/*'
  const mode = section.videoOnly ? 'video' : state.mode
  const angles = state[mode]
  const otherCount = filesOf({ ...state, [mode]: [] }).length

  const setAngles = (next) => setState({ ...state, [mode]: next })

  const update = (angleId, changes) =>
    setAngles(
      angles.map((angle) => (angle.id === angleId ? { ...angle, ...changes } : angle)),
    )

  const addFiles = (angleId, incoming) => {
    const angle = angles.find((entry) => entry.id === angleId)
    update(angleId, { files: [...angle.files, ...incoming] })
    setReport(null)
  }

  const removeFile = (angleId, position) => {
    const angle = angles.find((entry) => entry.id === angleId)
    update(angleId, { files: angle.files.filter((_, i) => i !== position) })
    setReport(null)
  }

  /**
   * Check what this section alone produced.
   *
   * Per-section rather than per-upload on purpose: told "gait: not ready" for
   * a whole submission you do not know which recording failed. Told it about
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
      const preview = await api.previewEnrollment(labelledFiles(section.key, state))
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
        <p className="spec">{section.spec}</p>
        <p className="muted small" style={{ marginBottom: 16 }}>
          {section.limit}
        </p>

        {!section.videoOnly && (
          <div className="mode-switch" role="group" aria-label="How to provide footage">
            <button
              type="button"
              className={`mode ${mode === 'video' ? 'active' : ''}`}
              onClick={() => setState({ ...state, mode: 'video' })}
              disabled={busy}
            >
              One video
            </button>
            <button
              type="button"
              className={`mode ${mode === 'photos' ? 'active' : ''}`}
              onClick={() => setState({ ...state, mode: 'photos' })}
              disabled={busy}
            >
              Separate photos
            </button>
          </div>
        )}

        {mode === 'video' ? (
          <div className="angle">
            <p className="video-instruction">{section.video.instruction}</p>
            <ul className="covers">
              {section.video.covers.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
            <p className="muted small" style={{ marginBottom: 10 }}>
              {section.video.why}
            </p>

            <DropZone
              accept={accept}
              disabled={busy}
              prompt="Drop the video here, or click to choose"
              hint={section.videoOnly ? 'Video only' : 'Video — photos are on the other tab'}
              onFiles={(incoming) => addFiles(angles[0].id, incoming)}
            />
            <FileList
              files={angles[0].files}
              disabled={busy}
              onRemove={(position) => removeFile(angles[0].id, position)}
            />

            <div style={{ marginTop: 10 }}>
              <Recorder
                prompt={section.video.instruction}
                disabled={busy}
                allowVideo
                allowPhoto={!section.videoOnly}
                onCapture={(file) => addFiles(angles[0].id, [file])}
              />
            </div>
          </div>
        ) : (
          <div className="angles">
            {angles.map((angle) => (
              <div className="angle" key={angle.id}>
                <div className="angle-head">
                  <input
                    type="text"
                    value={angle.label}
                    onChange={(event) => update(angle.id, { label: event.target.value })}
                    aria-label="What this photo shows"
                    disabled={busy}
                  />
                  {angle.files.length > 0 && (
                    <span className="angle-count muted small">
                      {angle.files.length} file{angle.files.length > 1 ? 's' : ''}
                    </span>
                  )}
                  {angles.length > 1 && (
                    <button
                      type="button"
                      className="btn-icon"
                      aria-label={`Remove the "${angle.label}" slot`}
                      title="Remove this slot"
                      onClick={() =>
                        setAngles(angles.filter((entry) => entry.id !== angle.id))
                      }
                      disabled={busy}
                    >
                      ✕
                    </button>
                  )}
                </div>

                {angle.guide && <p className="angle-guide">{angle.guide}</p>}

                <DropZone
                  accept={accept}
                  disabled={busy}
                  prompt="Drop photos here, or click to choose"
                  hint="Photos"
                  onFiles={(incoming) => addFiles(angle.id, incoming)}
                />
                <FileList
                  files={angle.files}
                  disabled={busy}
                  onRemove={(position) => removeFile(angle.id, position)}
                />

                {/* Photo mode had no camera at all, so anyone without files
                    already on disk could not use this tab. Video capture is
                    off here: a recording does not belong in a slot labelled
                    "Looking straight at the camera". */}
                <div style={{ marginTop: 10 }}>
                  <Recorder
                    prompt={`Line up the shot — ${angle.label.toLowerCase()} — then take it.`}
                    disabled={busy}
                    allowVideo={false}
                    allowPhoto
                    onCapture={(file) => addFiles(angle.id, [file])}
                  />
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="actions" style={{ marginTop: 14 }}>
          {mode === 'photos' && (
            <button
              type="button"
              className="btn btn-quiet btn-small"
              onClick={() => setAngles([...angles, newAngle({ label: 'Another view' })])}
              disabled={busy}
            >
              + Add another view
            </button>
          )}
          <button
            type="button"
            className="btn btn-quiet btn-small"
            onClick={check}
            disabled={busy || checking || !files.length}
          >
            {checking ? 'Checking…' : `Check ${section.title.toLowerCase()}`}
          </button>
        </div>

        {otherCount > 0 && (
          <p className="muted small" style={{ marginTop: 10 }}>
            {otherCount} file{otherCount > 1 ? 's' : ''} added under{' '}
            {mode === 'video' ? '“Separate photos”' : '“One video”'} will be sent
            too — switching tabs hides them, it does not drop them.
          </p>
        )}

        {report && (
          <p className={report.ready ? 'notice' : 'caution'} style={{ marginBottom: 0 }}>
            <span className={report.ready ? 'ok' : 'not-ok'}>
              {report.ready ? 'ready' : 'not ready'}
            </span>{' '}
            {report.reason}
            {!report.ready && <span className="muted"> {report.requirement}</span>}
          </p>
        )}
      </div>
    </section>
  )
}

/* -------------------------------------------------------------------- page */

/**
 * @param prefill  When Records sends someone here to have their footage
 *   re-recorded: their existing identity, with `replace` already on. The ID is
 *   locked in that case -- the whole point is to overwrite THIS record, and a
 *   typo in the field would quietly enrol a second person instead.
 */
export default function Register({ onError, onRegistered, prefill = null }) {
  const [personId, setPersonId] = useState(prefill?.person_id || '')
  const [displayName, setDisplayName] = useState(prefill?.display_name || '')
  const [notes, setNotes] = useState(prefill?.notes || '')
  const [replace, setReplace] = useState(Boolean(prefill))
  const [sections, setSections] = useState(initialSections)

  const [job, setJob] = useState(null)
  const [done, setDone] = useState(null)

  const busy = Boolean(job)

  const allFiles = useMemo(
    () => SECTIONS.flatMap((section) => labelledFiles(section.key, sections[section.key])),
    [sections],
  )

  const reset = () => {
    // Back to the prefilled identity rather than to blank, when there is one:
    // clearing the form mid-update should not turn an overwrite into a new
    // registration without saying so.
    setPersonId(prefill?.person_id || '')
    setDisplayName(prefill?.display_name || '')
    setNotes(prefill?.notes || '')
    setReplace(Boolean(prefill))
    setSections(initialSections())
  }

  const submit = async (event) => {
    event.preventDefault()

    if (!personId.trim() || !displayName.trim()) {
      onError(
        'A name and an ID are both required — a record with neither cannot be reviewed later.',
      )
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
        <h2>{prefill ? 'Update stored footage' : 'Register a person'}</h2>
        <p>
          Record each signal separately. You do not have to provide all three,
          but a record missing a signal simply cannot match on it later — and
          you will not find that out at the moment it matters.
        </p>
      </div>

      {prefill && (
        <p className="caution">
          <strong>Overwriting {prefill.display_name}.</strong> Whatever you
          record here replaces their stored signals entirely — signals you do
          not supply this time are lost, not kept from before. Past decisions
          about them are unaffected.
        </p>
      )}

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
              disabled={busy || Boolean(prefill)}
              readOnly={Boolean(prefill)}
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
            disabled={busy || Boolean(prefill)}
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
          state={sections[section.key]}
          setState={(next) =>
            setSections((current) => ({ ...current, [section.key]: next }))
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
        <button type="button" className="btn btn-quiet" onClick={reset} disabled={busy}>
          Clear
        </button>
        <button className="btn btn-primary" type="submit" disabled={busy}>
          {busy
            ? prefill
              ? 'Updating…'
              : 'Registering…'
            : `${prefill ? 'Replace stored footage' : 'Register'}${
                allFiles.length ? ` — ${allFiles.length} file(s)` : ''
              }`}
        </button>
      </div>
    </form>
  )
}
