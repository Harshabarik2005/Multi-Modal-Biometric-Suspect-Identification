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

  /**
   * Check what an upload can support before committing to it.
   *
   * Runs detection only, so it returns in seconds. Worth calling first: a
   * photograph contains no gait information at all, and without this the
   * person only finds out after waiting through a full enrolment that
   * quietly stored a face-only profile.
   */
  previewEnrollment: (files) => upload('/enroll/preview', { files }),

  enroll: ({ files, personId, displayName, notes, operator, replace }) =>
    upload('/enroll', {
      files,
      fields: {
        person_id: personId,
        display_name: displayName,
        notes: notes || '',
        operator: operator || 'unknown',
        replace: replace ? 'true' : 'false',
      },
    }),

  scan: ({ files, cameraId, threshold }) =>
    upload('/scan', {
      files,
      fields: {
        camera_id: cameraId || '',
        threshold: threshold === undefined ? '-1' : String(threshold),
        record: 'true',
      },
    }),

  job: (jobId) => request(`/jobs/${jobId}`),
}

/** POST multipart form data. Cannot use `request`, which sets a JSON header. */
async function upload(path, { files = [], fields = {} }) {
  const body = new FormData()
  for (const [key, value] of Object.entries(fields)) body.append(key, value)
  for (const file of files) body.append('files', file, file.name)

  const response = await fetch(`/api${path}`, { method: 'POST', body })
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const parsed = await response.json()
      if (parsed.detail) {
        detail =
          typeof parsed.detail === 'string'
            ? parsed.detail
            : JSON.stringify(parsed.detail)
      }
    } catch {
      // No JSON body; the status line is all there is.
    }
    throw new Error(detail)
  }
  return response.json()
}

/**
 * Poll a background job until it finishes.
 *
 * Enrolment and scanning run three models over every frame and take minutes,
 * so they cannot happen inside a request. `onProgress` is called with each
 * update so the UI can show what stage it has reached.
 */
export async function waitForJob(jobId, onProgress, intervalMs = 1000) {
  for (;;) {
    const job = await api.job(jobId)
    onProgress?.(job)
    if (job.status === 'succeeded' || job.status === 'failed') return job
    await new Promise((resolve) => setTimeout(resolve, intervalMs))
  }
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
