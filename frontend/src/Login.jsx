import { useState } from 'react'
import { api } from './api'

/**
 * Sign-in screen.
 *
 * Before authentication existed, the "operator" was a name typed into a box at
 * the top of the console and sent with each request. That made the audit trail
 * a record of unverified claims: an identification confirmed by "whoever typed
 * alice" is not confirmed by anyone. Now the console cannot be used at all
 * without an account, and a decision records the session that made it.
 */
export default function Login({ onSignedIn }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const submit = async (event) => {
    event.preventDefault()
    if (!username.trim() || !password) {
      setError('Enter your username and password.')
      return
    }

    setBusy(true)
    setError('')
    try {
      onSignedIn(await api.login(username.trim(), password))
    } catch (exception) {
      setError(exception.message)
      setPassword('')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="login-shell">
      <form className="card login-card" onSubmit={submit}>
        <h1>Faceless FRS</h1>
        <p className="muted small">
          Multi-modal identification — face, gait and appearance
        </p>

        <label className="stacked">
          <span>Username</span>
          <input
            type="text"
            autoComplete="username"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            disabled={busy}
            autoFocus
          />
        </label>

        <label className="stacked">
          <span>Password</span>
          <input
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            disabled={busy}
          />
        </label>

        {error && <p className="caution">{error}</p>}

        <button className="btn btn-confirm login-submit" type="submit" disabled={busy}>
          {busy ? 'Signing in…' : 'Sign in'}
        </button>

        <p className="muted small">
          Accounts are created on the server, not here:{' '}
          <code>python scripts/manage_operators.py --create &lt;username&gt;</code>
        </p>
        <p className="muted small">
          Everything you confirm or reject is recorded against this account.
        </p>
      </form>
    </div>
  )
}
