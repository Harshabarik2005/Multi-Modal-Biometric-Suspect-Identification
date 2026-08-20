/**
 * Thin client for the Faceless FRS API.
 *
 * Every call goes through `/api`, which Vite proxies to the backend in
 * development. No biometric vectors ever cross this boundary -- the API
 * describes templates but never returns them -- so nothing here needs to
 * handle embedding data.
 */

async function request(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (body.detail) {
        detail =
          typeof body.detail === 'string'
            ? body.detail
            : JSON.stringify(body.detail)
      }
    } catch {
      // Response had no JSON body; the status line is all we have.
    }
    throw new Error(detail)
  }
  return response.json()
}

export const api = {
  health: () => request('/health'),

  watchlist: (includeRetired = false) =>
    request(`/watchlist?include_retired=${includeRetired}`),

  person: (personId) => request(`/watchlist/${encodeURIComponent(personId)}`),

  updatePerson: (personId, payload) =>
    request(`/watchlist/${encodeURIComponent(personId)}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),

  retirePerson: (personId, operator) =>
    request(
      `/watchlist/${encodeURIComponent(personId)}?operator=${encodeURIComponent(operator)}`,
      { method: 'DELETE' },
    ),

  decisions: (pendingOnly = true, limit = 100) =>
    request(`/decisions?pending_only=${pendingOnly}&limit=${limit}`),

  /**
   * Record a human verdict. This is the only route to a confirmed match --
   * nothing in the system confirms one on its own.
   */
  review: (decisionId, operator, verdict, reason = '') =>
    request(`/decisions/${decisionId}/review`, {
      method: 'POST',
      body: JSON.stringify({ operator, verdict, reason }),
    }),

  alerts: (limit = 100) => request(`/alerts?limit=${limit}`),

  audit: (limit = 200) => request(`/audit?limit=${limit}`),
}

/** Display metadata per modality. Colours are reused by the weight bars. */
export const MODALITIES = {
  face: { label: 'Face', colour: '#3b82f6', blurb: 'ArcFace identity' },
  gait: { label: 'Gait', colour: '#a855f7', blurb: 'How they walk' },
  reid: { label: 'Re-ID', colour: '#f59e0b', blurb: 'Build and clothing' },
}

/**
 * How much of a match rested on appearance alone.
 *
 * Re-ID largely encodes clothing, so a match driven mostly by it is far weaker
 * evidence than one driven by a face -- and a reviewer about to confirm an
 * identification needs that distinction in front of them, not buried in a
 * number.
 */
export function appearanceShare(weights = {}) {
  const total = Object.values(weights).reduce((sum, w) => sum + w, 0)
  if (total <= 0) return 0
  return (weights.reid || 0) / total
}
