import { useEffect, useState } from 'react'

/**
 * A photograph of a person, fetched with the session token.
 *
 * Not a plain `<img src>`: the image routes require a bearer token and an img
 * tag cannot send a header, so the bytes are fetched and turned into an object
 * URL, which is released when the component goes away.
 *
 * A missing image is stated rather than hidden. Somebody being asked to judge
 * an identification without a picture needs to know that is what is happening
 * — a blank space reads as "still loading" and quietly becomes "fine".
 */
export default function PersonShot({ fetcher, alt, variant = '', missing }) {
  const [url, setUrl] = useState(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    if (!fetcher) {
      setUrl(null)
      return undefined
    }

    let cancelled = false
    let objectUrl = null

    setUrl(null)
    setFailed(false)

    fetcher()
      .then((value) => {
        if (cancelled) {
          // Unmounted while the image was in flight. The cleanup below has
          // already run and saw nothing to release, so it has to happen here
          // or the blob is held for the life of the page.
          if (value) URL.revokeObjectURL(value)
          return
        }
        objectUrl = value
        if (value) setUrl(value)
        else setFailed(true)
      })
      .catch(() => {
        if (!cancelled) setFailed(true)
      })

    return () => {
      cancelled = true
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
    // `fetcher` is rebuilt on every render by most callers, so keying the
    // effect on it would refetch forever. The caller passes a `key` instead.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  if (url) {
    return <img className={`person-shot ${variant}`} src={url} alt={alt} />
  }

  return (
    <div className={`shot-missing ${variant}`}>
      {failed || !fetcher ? missing || 'No image' : 'Loading…'}
    </div>
  )
}
