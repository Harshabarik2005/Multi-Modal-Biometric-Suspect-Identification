/**
 * Thin client for the Faceless FRS API.
 *
 * Every call goes through `/api`, which Vite proxies to the backend in
 * development. No biometric vectors ever cross this boundary -- the API
 * describes templates but never returns them -- so nothing here needs to
 * handle embedding data.
 */

/**
 * The bearer token for the signed-in operator.
 *
 * Held in memory, not localStorage. A token in localStorage is readable by any
 * script that ends up on the page, and this one can confirm an identification.
 * The cost is that a refresh signs you out, which is the right trade for what
 * this token authorises.
 */
let authToken = null
let onUnauthorized = null

export function setToken(token) {
  authToken = token
}

export function clearToken() {
  authToken = null
}

export function setUnauthorizedHandler(handler) {
  onUnauthorized = handler
}

function authHeaders(extra = {}) {
  return authToken
    ? { ...extra, Authorization: `Bearer ${authToken}` }
    : { ...extra }
}

async function request(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    ...options,
    headers: authHeaders({ 'Content-Type': 'application/json', ...(options.headers || {}) }),
  })

  if (response.status === 401) {
    clearToken()
    onUnauthorized?.()
    throw httpError('Your session has ended. Sign in again.', 401)
  }

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
    throw httpError(detail, response.status)
  }
  return response.json()
}

/**
 * An Error that remembers its HTTP status.
 *
 * Callers that retry need to distinguish "the server hiccuped" from "you are
 * signed out" -- retrying the second is pointless and hides the real problem.
 */
function httpError(message, status) {
  const error = new Error(message)
  error.status = status
  return error
}

export const api = {
  health: () => request('/health'),

  login: async (username, password) => {
    const session = await request('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    })
    setToken(session.token)
    return session
  },

  me: () => request('/auth/me'),

  stats: () => request('/stats'),

  watchlist: (includeRetired = false) =>
    request(`/watchlist?include_retired=${includeRetired}`),

  person: (personId) => request(`/watchlist/${encodeURIComponent(personId)}`),

  updatePerson: (personId, payload) =>
    request(`/watchlist/${encodeURIComponent(personId)}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),

  retirePerson: (personId) =>
    request(`/watchlist/${encodeURIComponent(personId)}`, { method: 'DELETE' }),

  decisions: (pendingOnly = true, limit = 100) =>
    request(`/decisions?pending_only=${pendingOnly}&limit=${limit}`),

  /**
   * Record a human verdict. The only route to a confirmed match.
   *
   * The operator is NOT sent: the server takes it from the authenticated
   * session, so a decision records who was actually signed in rather than
   * whatever name the page claimed.
   */
  review: (decisionId, verdict, reason = '') =>
    request(`/decisions/${decisionId}/review`, {
      method: 'POST',
      body: JSON.stringify({ verdict, reason }),
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

  enroll: ({ files, personId, displayName, notes, replace }) =>
    upload('/enroll', {
      files,
      fields: {
        person_id: personId,
        display_name: displayName,
        notes: notes || '',
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

  /**
   * The crop a match was made on, and the enrolment crop to compare it with
   * (DES-02).
   *
   * Fetched rather than pointed at with an <img src>, because these routes
   * need the bearer token and an img tag cannot send a header. Returns an
   * object URL the caller must revoke, or null when no image was stored --
   * which is not an error, and the reviewer needs to be told about it.
   */
  evidenceImage: (decisionId) => imageUrl(`/decisions/${decisionId}/evidence`),
  referenceImage: (personId) =>
    imageUrl(`/watchlist/${encodeURIComponent(personId)}/reference`),
}

async function imageUrl(path) {
  const response = await fetch(`/api${path}`, { headers: authHeaders() })

  if (response.status === 401) {
    clearToken()
    onUnauthorized?.()
    throw httpError('Your session has ended. Sign in again.', 401)
  }
  if (response.status === 404) return null
  if (!response.ok) {
    throw httpError(`${response.status} ${response.statusText}`, response.status)
  }
  return URL.createObjectURL(await response.blob())
}

/** POST multipart form data. Cannot use `request`, which sets a JSON header. */
async function upload(path, { files = [], fields = {} }) {
  const body = new FormData()
  for (const [key, value] of Object.entries(fields)) body.append(key, value)
  for (const file of files) body.append('files', file, file.name)

  const response = await fetch(`/api${path}`, {
    method: 'POST',
    body,
    headers: authHeaders(),
  })

  if (response.status === 401) {
    clearToken()
    onUnauthorized?.()
    throw httpError('Your session has ended. Sign in again.', 401)
  }
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
    throw httpError(detail, response.status)
  }
  return response.json()
}

/**
 * Poll a background job until it finishes.
 *
 * Enrolment and scanning run three models over every frame and take minutes,
 * so they cannot happen inside a request. `onProgress` is called with each
 * update so the UI can show what stage it has reached.
 *
 * Bounded, deliberately. This was an unbounded `for(;;)`: restart the server
 * mid-scan and the job is gone, every poll 404s, and the browser sits on
 * "scanning..." forever with no way to tell the difference between slow and
 * dead (DES-03). It also tolerates a few consecutive failures first, because
 * one dropped request during a minutes-long job is not a reason to give up.
 */
export const JOB_TIMEOUT_MS = 30 * 60 * 1000
const MAX_CONSECUTIVE_ERRORS = 5

export async function waitForJob(
  jobId,
  onProgress,
  intervalMs = 1000,
  timeoutMs = JOB_TIMEOUT_MS,
) {
  const deadline = Date.now() + timeoutMs
  let consecutiveErrors = 0

  for (;;) {
    let job
    try {
      job = await api.job(jobId)
      consecutiveErrors = 0
    } catch (error) {
      // A 401 means the session ended; retrying cannot fix that, and the
      // caller has already been dropped back to sign-in.
      if (error.status === 401) throw error

      consecutiveErrors += 1
      if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        throw new Error(
          `Lost contact with the server while the job was running (${error.message}). ` +
            'It may have restarted. Check the results before re-running — ' +
            'the work may have completed.',
        )
      }
      await sleep(intervalMs)
      continue
    }

    onProgress?.(job)
    if (job.status === 'succeeded' || job.status === 'failed') return job

    if (Date.now() > deadline) {
      throw new Error(
        `The job did not finish within ${Math.round(timeoutMs / 60000)} minutes. ` +
          'It may still be running on the server — check before starting again.',
      )
    }
    await sleep(intervalMs)
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
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
