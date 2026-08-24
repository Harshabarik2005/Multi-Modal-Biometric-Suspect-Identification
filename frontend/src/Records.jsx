import { useCallback, useEffect, useState } from 'react'
import { api, MODALITIES } from './api'
import PersonShot from './PersonShot'

/**
 * The watchlist, and the operations that change it.
 *
 * Shown as faces rather than as a table of identifiers. Everyone on this list
 * is a real person, and a screen that renders them as `case-2291` makes it
 * fractionally easier to forget that. It is also the practical view: someone
 * checking whether a person is already registered recognises the photograph
 * long before they recognise the ID.
 *
 * Which signals are stored is on the card because it decides what this record
 * can do. A record holding only appearance will match a jacket and little
 * else, and the moment to know that is now, not when a match arrives.
 *
 * Four ways to change a record, deliberately distinct rather than one "delete"
 * that quietly picks for you:
 *
 * * **Edit** -- name and notes. Never the ID: it is what every match decision
 *   was filed under, so changing it would detach a person from their history.
 * * **Retire** -- off the active list, reversible, destroys nothing.
 * * **Erase biometrics** -- destroys the templates and the enrolment photo for
 *   good, keeps the record and its decisions so past identifications stay
 *   reviewable. This is what an erasure request actually asks for.
 * * **Delete** -- removes the record entirely. The server refuses once anyone
 *   has been matched against it, because that would leave the review queue
 *   pointing at nobody.
 */

/** Destructive actions state their consequence in the button, not just a colour. */
const CONFIRMATIONS = {
  retire: {
    title: 'Take off the watchlist?',
    body: (person) => (
      <>
        <strong>{person.display_name}</strong> stops being matched against.
        Nothing is destroyed — their signals and past decisions are kept, and
        you can put them back at any time.
      </>
    ),
    confirm: 'Retire',
    danger: false,
  },
  erase: {
    title: 'Destroy their biometric data?',
    body: (person) => (
      <>
        Permanently destroys the stored signals and the registration photograph
        for <strong>{person.display_name}</strong>. They can never be matched
        again unless re-registered from footage.
        <br />
        <br />
        The record and any past decisions about them are kept, so what this
        system already decided stays reviewable. <strong>This cannot be
        undone.</strong>
      </>
    ),
    confirm: 'Erase biometrics',
    danger: true,
    typed: true,
  },
  delete: {
    title: 'Delete this record entirely?',
    body: (person) => (
      <>
        Removes <strong>{person.display_name}</strong> and everything stored
        about them, leaving no trace in the watchlist.
        <br />
        <br />
        Possible because nobody has been matched against this record yet — once
        somebody has, deleting is refused, because the decisions would be left
        pointing at nobody. <strong>This cannot be undone.</strong>
      </>
    ),
    confirm: 'Delete permanently',
    danger: true,
    typed: true,
  },
}

/**
 * Why this record cannot be deleted outright, or '' when it can.
 *
 * Checked before the button is offered rather than after it is pressed. The
 * server refuses either way, but finding out by typing your way through an
 * irreversible-looking confirmation and receiving an error teaches people that
 * the confirmation is bluffing -- which is the opposite of what it is for.
 */
function deleteBlockedReason(person) {
  if (!person.decision_count) return ''
  const n = person.decision_count
  const named = `Named in ${n} match decision${n === 1 ? '' : 's'}, so the record has to stay — deleting it would leave those pointing at nobody.`
  // Once the templates are gone there is nothing left to suggest erasing, and
  // repeating the advice would read as though the erasure had not worked.
  return person.templates.length
    ? `${named} Erase their biometrics instead.`
    : `${named} Their biometrics are already erased.`
}

/**
 * Confirmation for an action that cannot be taken back.
 *
 * The irreversible ones ask for the person's ID to be typed. A dialog that is
 * dismissed with one click is dismissed reflexively, and the cost of getting
 * this wrong is biometric data that only exists in footage nobody kept.
 */
function Confirm({ kind, person, onCancel, onConfirm, busy }) {
  const spec = CONFIRMATIONS[kind]
  const [typed, setTyped] = useState('')
  const satisfied = !spec.typed || typed.trim() === person.person_id

  useEffect(() => {
    const onKey = (event) => {
      if (event.key === 'Escape' && !busy) onCancel()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onCancel, busy])

  return (
    <div className="modal-backdrop" onMouseDown={busy ? undefined : onCancel}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <h3 id="confirm-title">{spec.title}</h3>
        <p className="small">{spec.body(person)}</p>

        {spec.typed && (
          <label className="stacked" style={{ marginTop: 16 }}>
            <span>
              Type <span className="mono">{person.person_id}</span> to confirm
            </span>
            <input
              type="text"
              className="mono"
              value={typed}
              autoFocus
              onChange={(event) => setTyped(event.target.value)}
              disabled={busy}
            />
          </label>
        )}

        <div className="actions actions-end" style={{ marginTop: 20 }}>
          <button
            type="button"
            className="btn btn-quiet"
            onClick={onCancel}
            disabled={busy}
          >
            Cancel
          </button>
          <button
            type="button"
            className={`btn ${spec.danger ? 'btn-danger' : 'btn-primary'}`}
            onClick={onConfirm}
            disabled={busy || !satisfied}
          >
            {busy ? 'Working…' : spec.confirm}
          </button>
        </div>
      </div>
    </div>
  )
}

/** Inline editor for the fields that are safe to change. */
function EditPanel({ person, onCancel, onSaved, onError }) {
  const [displayName, setDisplayName] = useState(person.display_name)
  const [notes, setNotes] = useState(person.notes || '')
  const [saving, setSaving] = useState(false)

  const dirty =
    displayName !== person.display_name || notes !== (person.notes || '')

  const save = async (event) => {
    event.preventDefault()
    if (!displayName.trim()) {
      onError('A record still needs a name — leave it blank and nobody can recognise it in a review queue.')
      return
    }
    setSaving(true)
    try {
      await api.updatePerson(person.person_id, {
        display_name: displayName.trim(),
        notes,
      })
      onSaved()
    } catch (exception) {
      onError(exception.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <form className="record-edit" onSubmit={save}>
      <label className="stacked">
        <span>Full name</span>
        <input
          type="text"
          value={displayName}
          onChange={(event) => setDisplayName(event.target.value)}
          disabled={saving}
          autoFocus
        />
      </label>

      <label className="stacked">
        <span>Notes</span>
        <textarea
          value={notes}
          rows={3}
          onChange={(event) => setNotes(event.target.value)}
          placeholder="Why this person is on the watchlist, and on whose authority."
          disabled={saving}
        />
      </label>

      <p className="hint" style={{ marginTop: 2 }}>
        The reference ID <span className="mono">{person.person_id}</span> cannot
        be changed — every decision about this person was filed under it.
      </p>

      <div className="actions actions-end" style={{ marginTop: 12 }}>
        <button
          type="button"
          className="btn btn-quiet"
          onClick={onCancel}
          disabled={saving}
        >
          Cancel
        </button>
        <button className="btn btn-primary" type="submit" disabled={saving || !dirty}>
          {saving ? 'Saving…' : 'Save changes'}
        </button>
      </div>
    </form>
  )
}

function RecordCard({ person, onError, onChanged, onReplaceFootage }) {
  const [editing, setEditing] = useState(false)
  const [confirming, setConfirming] = useState(null)
  const [busy, setBusy] = useState(false)

  const retired = !person.is_active
  const hasSignals = person.templates.length > 0
  const blockedReason = deleteBlockedReason(person)

  const act = async () => {
    setBusy(true)
    try {
      if (confirming === 'retire') await api.retirePerson(person.person_id)
      if (confirming === 'erase') await api.eraseBiometrics(person.person_id)
      if (confirming === 'delete') await api.deletePerson(person.person_id)
      setConfirming(null)
      onChanged()
    } catch (exception) {
      setConfirming(null)
      onError(exception.message)
    } finally {
      setBusy(false)
    }
  }

  const restore = async () => {
    setBusy(true)
    try {
      await api.restorePerson(person.person_id)
      onChanged()
    } catch (exception) {
      onError(exception.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <article className={`record ${retired ? 'record-retired' : ''}`}>
      <PersonShot
        key={`${person.person_id}-${person.has_reference}`}
        variant="tile"
        alt={`Registration photo for ${person.display_name}`}
        missing={hasSignals ? 'No photo stored' : 'Biometrics erased'}
        fetcher={
          person.has_reference
            ? () => api.referenceImage(person.person_id)
            : null
        }
      />

      <div className="record-body">
        <h3>{person.display_name}</h3>
        <div className="record-id mono">{person.person_id}</div>

        {retired && (
          <p className="record-flag">
            Retired{' '}
            {person.retired_at
              ? new Date(person.retired_at).toLocaleDateString()
              : ''}{' '}
            — not matched against.
          </p>
        )}

        <div className="record-signals">
          {!hasSignals && (
            <span className="muted small">
              No signals stored — this record cannot match anyone.
            </span>
          )}
          {person.templates.map((template) => (
            <span
              className="pill"
              key={template.modality}
              title={
                template.encrypted
                  ? 'Encrypted at rest'
                  : 'NOT ENCRYPTED — set FRS_TEMPLATE_ENCRYPTION_KEY'
              }
            >
              <span
                className="swatch"
                style={{ background: MODALITIES[template.modality]?.tone }}
              />
              {MODALITIES[template.modality]?.label || template.modality}
              {!template.encrypted && ' ⚠'}
            </span>
          ))}
        </div>

        <p className="muted small" style={{ marginTop: 8 }}>
          Registered{' '}
          {person.enrolled_at
            ? new Date(person.enrolled_at).toLocaleDateString()
            : '—'}
        </p>

        {editing ? (
          <EditPanel
            person={person}
            onError={onError}
            onCancel={() => setEditing(false)}
            onSaved={() => {
              setEditing(false)
              onChanged()
            }}
          />
        ) : (
          <>
            {person.notes && (
              <p className="small" style={{ marginTop: 6, color: 'var(--ink-2)' }}>
                {person.notes}
              </p>
            )}

            <div className="record-actions">
              <button
                type="button"
                className="btn btn-quiet btn-small"
                onClick={() => setEditing(true)}
                disabled={busy}
              >
                Edit details
              </button>

              <button
                type="button"
                className="btn btn-quiet btn-small"
                onClick={() => onReplaceFootage(person)}
                disabled={busy}
                title="Re-record this person's signals from new footage"
              >
                Update footage
              </button>

              {retired ? (
                <button
                  type="button"
                  className="btn btn-quiet btn-small"
                  onClick={restore}
                  disabled={busy}
                >
                  {busy ? 'Restoring…' : 'Put back'}
                </button>
              ) : (
                <button
                  type="button"
                  className="btn btn-quiet btn-small"
                  onClick={() => setConfirming('retire')}
                  disabled={busy}
                >
                  Retire
                </button>
              )}

              {hasSignals && (
                <button
                  type="button"
                  className="btn btn-quiet btn-small btn-danger-quiet"
                  onClick={() => setConfirming('erase')}
                  disabled={busy}
                  title="Destroy the stored signals and photograph. Cannot be undone."
                >
                  Erase biometrics
                </button>
              )}

              <button
                type="button"
                className="btn btn-quiet btn-small btn-danger-quiet"
                onClick={() => setConfirming('delete')}
                disabled={busy || Boolean(blockedReason)}
                title={
                  blockedReason ||
                  'Remove the record entirely. Cannot be undone.'
                }
              >
                Delete
              </button>
            </div>

            {blockedReason && (
              <p className="hint" style={{ marginTop: 8 }}>
                {blockedReason}
              </p>
            )}
          </>
        )}
      </div>

      {confirming && (
        <Confirm
          kind={confirming}
          person={person}
          busy={busy}
          onCancel={() => setConfirming(null)}
          onConfirm={act}
        />
      )}
    </article>
  )
}

export default function Records({ onError, onReplaceFootage }) {
  const [people, setPeople] = useState(null)
  const [query, setQuery] = useState('')
  const [showRetired, setShowRetired] = useState(false)

  const load = useCallback(() => {
    api
      .watchlist(true)
      .then(setPeople)
      .catch((exception) => {
        setPeople([])
        onError(exception.message)
      })
  }, [onError])

  useEffect(load, [load])

  if (people === null) {
    return <p className="muted small">Loading…</p>
  }

  const active = people.filter((person) => person.is_active)
  const retiredCount = people.length - active.length

  if (people.length === 0) {
    return (
      <div className="division">
        <div className="division-head">
          <h2>Records</h2>
          <p>Everyone currently on the watchlist.</p>
        </div>
        <div className="empty">
          <h3>Nobody is registered</h3>
          <p>
            Use the Register tab to add someone. Nothing can be identified until
            there is at least one record to identify them against.
          </p>
        </div>
      </div>
    )
  }

  const term = query.trim().toLowerCase()
  const shown = people
    .filter((person) => showRetired || person.is_active)
    .filter(
      (person) =>
        !term ||
        person.display_name.toLowerCase().includes(term) ||
        person.person_id.toLowerCase().includes(term),
    )

  return (
    <div className="division">
      <div className="division-head">
        <h2>Records</h2>
        <p>
          {active.length} {active.length === 1 ? 'person' : 'people'} on the
          watchlist
          {retiredCount > 0 && `, ${retiredCount} retired`}. The signals listed
          under each are what that record can actually be matched on.
        </p>
      </div>

      <div className="record-toolbar">
        <input
          type="text"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search by name or ID"
        />
        {retiredCount > 0 && (
          <label className="checkbox">
            <input
              type="checkbox"
              checked={showRetired}
              onChange={(event) => setShowRetired(event.target.checked)}
            />
            <span>
              Show {retiredCount} retired {retiredCount === 1 ? 'record' : 'records'}
            </span>
          </label>
        )}
      </div>

      {shown.length === 0 ? (
        <div className="empty">
          <h3>No match</h3>
          <p>
            {term
              ? `Nobody on the watchlist matches “${query}”.`
              : 'Every record is retired. Tick the box above to see them.'}
          </p>
        </div>
      ) : (
        <div className="record-grid">
          {shown.map((person) => (
            <RecordCard
              key={person.person_id}
              person={person}
              onError={onError}
              onChanged={load}
              onReplaceFootage={onReplaceFootage}
            />
          ))}
        </div>
      )}
    </div>
  )
}
