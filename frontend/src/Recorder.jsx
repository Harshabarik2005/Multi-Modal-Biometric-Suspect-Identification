import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Capture enrolment footage from the camera.
 *
 * `prompt` is the section's own instruction — a gait section wants the person
 * walking across the frame, a face section wants them close and turning slowly
 * — so the guidance on screen is specific to what is being captured rather
 * than generic advice nobody reads.
 *
 * The preview is deliberately NOT mirrored. Video-call apps mirror self-view
 * because it matches your own proprioception, but this is mostly one person
 * recording another, and more importantly the captured file is not mirrored:
 * somebody checking "did I get the left-turn shot" needs the preview and the
 * saved frame to agree.
 */
export default function Recorder({
  prompt,
  onCapture,
  disabled,
  allowVideo = true,
  allowPhoto = true,
}) {
  const videoRef = useRef(null)
  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const streamRef = useRef(null)

  const [active, setActive] = useState(false)
  const [recording, setRecording] = useState(false)
  const [seconds, setSeconds] = useState(0)
  const [shots, setShots] = useState(0)
  const [capturing, setCapturing] = useState(false)
  const [error, setError] = useState('')

  const stop = useCallback(() => {
    recorderRef.current?.state === 'recording' && recorderRef.current.stop()
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    setActive(false)
    setRecording(false)
    setShots(0)
    setCapturing(false)
  }, [])

  // Releasing the camera on unmount matters: a webcam left streaming keeps its
  // light on, which looks exactly like covert recording.
  useEffect(() => stop, [stop])

  /**
   * Attach the stream once the <video> is actually on the page.
   *
   * This cannot happen in `start()`. The element only renders while `active`
   * is true, so at the moment the stream arrives `videoRef.current` is still
   * null — and the assignment used to be wrapped in `if (videoRef.current)`,
   * which turned that into silence rather than an error. The camera opened,
   * its light came on, and the preview stayed blank.
   */
  useEffect(() => {
    const video = videoRef.current
    if (!active || !video || !streamRef.current) return
    video.srcObject = streamRef.current
    // autoPlay usually covers this, but a tab restored from background needs
    // the explicit nudge, and a rejected promise here is not worth surfacing.
    video.play().catch(() => {})
  }, [active])

  useEffect(() => {
    if (!recording) return undefined
    const timer = setInterval(() => setSeconds((value) => value + 1), 1000)
    return () => clearInterval(timer)
  }, [recording])

  const start = async () => {
    setError('')
    if (!navigator.mediaDevices?.getUserMedia) {
      setError(
        'This browser will not give a page camera access. Chrome, Edge, Firefox ' +
          'and Safari all will, over https or on localhost.',
      )
      return
    }

    try {
      streamRef.current = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      })
      setActive(true)
    } catch (exception) {
      const reason =
        {
          NotAllowedError:
            'Permission was refused. Allow camera access for this site and try again.',
          NotFoundError: 'No camera was found on this device.',
          NotReadableError:
            'The camera is already in use by another application.',
        }[exception.name] || exception.message

      setError(
        `Could not open the camera: ${reason} Browsers only allow camera ` +
          'access over https or on localhost.',
      )
    }
  }

  /**
   * Grab the current frame as a JPEG.
   *
   * `toBlob` encodes asynchronously and can take the better part of a second
   * on a slow machine, so the button reports itself busy for the duration. A
   * control that does nothing visible for a second reads as broken, and the
   * person will press it again and get two of the same frame.
   */
  const takePhoto = () => {
    const video = videoRef.current
    if (!video || !video.videoWidth || capturing) return

    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)

    setCapturing(true)
    canvas.toBlob(
      (blob) => {
        setCapturing(false)
        if (!blob) {
          setError('The camera frame could not be saved. Try again.')
          return
        }
        const stamp = new Date().toISOString().replace(/[:.]/g, '-')
        onCapture(new File([blob], `photo-${stamp}.jpg`, { type: 'image/jpeg' }))
        setShots((count) => count + 1)
      },
      'image/jpeg',
      0.92,
    )
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
      onCapture(new File([blob], `recording-${stamp}.webm`, { type: 'video/webm' }))
    }
    recorder.start()
    recorderRef.current = recorder
    setRecording(true)
  }

  const endRecording = () => {
    recorderRef.current?.stop()
    setRecording(false)
  }

  if (!active) {
    return (
      <>
        {error && <p className="caution">{error}</p>}
        <button
          type="button"
          className="btn btn-quiet btn-small"
          onClick={start}
          disabled={disabled}
        >
          Use camera
        </button>
      </>
    )
  }

  return (
    <div className="recorder">
      <div className="recorder-stage">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="preview-video"
        />
        {recording && (
          <span className="rec-badge">
            <span className="rec-dot" aria-hidden="true" />
            REC {seconds}s
          </span>
        )}
      </div>

      {prompt && <p className="muted small recorder-prompt">{prompt}</p>}

      <div className="actions">
        {allowVideo &&
          (recording ? (
            <button type="button" className="btn btn-primary" onClick={endRecording}>
              Stop recording — {seconds}s
            </button>
          ) : (
            <button type="button" className="btn btn-primary" onClick={beginRecording}>
              Start recording
            </button>
          ))}

        {allowPhoto && !recording && (
          <button
            type="button"
            className="btn"
            onClick={takePhoto}
            disabled={capturing}
          >
            {capturing ? 'Saving…' : 'Take photo'}
          </button>
        )}

        <button type="button" className="btn btn-quiet" onClick={stop}>
          Close camera
        </button>
      </div>

      {shots > 0 && (
        <p className="muted small" style={{ marginTop: 8 }}>
          {shots === 1
            ? '1 photo taken — it is in the list above.'
            : `${shots} photos taken — they are in the list above.`}
        </p>
      )}
      {recording && seconds < 3 && (
        <p className="muted small" style={{ marginTop: 8 }}>
          Keep going — gait needs at least two seconds of walking, and four is
          better.
        </p>
      )}
    </div>
  )
}
