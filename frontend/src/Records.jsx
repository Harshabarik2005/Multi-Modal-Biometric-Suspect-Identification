import { useEffect, useState } from 'react'
import { api, MODALITIES } from './api'
import PersonShot from './PersonShot'

/**
 * The watchlist.
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
 */
export default function Records({ onError }) {
  const [people, setPeople] = useState(null)
  const [query, setQuery] = useState('')

  useEffect(() => {
    api
      .watchlist()
      .then(setPeople)
      .catch((exception) => {
        setPeople([])
        onError(exception.message)
      })
  }, [onError])

  if (people === null) {
    return <p className="muted small">Loading…</p>
  }

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
  const shown = term
    ? people.filter(
        (person) =>
          person.display_name.toLowerCase().includes(term) ||
          person.person_id.toLowerCase().includes(term),
      )
    : people

  return (
    <div className="division">
      <div className="division-head">
        <h2>Records</h2>
        <p>
          {people.length} {people.length === 1 ? 'person' : 'people'} on the
          watchlist. The signals listed under each are what that record can
          actually be matched on.
        </p>
      </div>

      <div style={{ maxWidth: 320, marginBottom: 18 }}>
        <input
          type="text"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search by name or ID"
        />
      </div>

      {shown.length === 0 ? (
        <div className="empty">
          <h3>No match</h3>
          <p>Nobody on the watchlist matches “{query}”.</p>
        </div>
      ) : (
        <div className="record-grid">
          {shown.map((person) => (
            <article className="record" key={person.person_id}>
              <PersonShot
                key={person.person_id}
                variant="tile"
                alt={`Registration photo for ${person.display_name}`}
                missing="No photo stored"
                fetcher={
                  person.has_reference
                    ? () => api.referenceImage(person.person_id)
                    : null
                }
              />
              <div className="record-body">
                <h3>{person.display_name}</h3>
                <div className="record-id mono">{person.person_id}</div>

                <div className="record-signals">
                  {person.templates.length === 0 && (
                    <span className="muted small">No signals stored</span>
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
                        style={{
                          background: MODALITIES[template.modality]?.tone,
                        }}
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

                {person.notes && (
                  <p className="small" style={{ marginTop: 6, color: 'var(--ink-2)' }}>
                    {person.notes}
                  </p>
                )}
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  )
}
