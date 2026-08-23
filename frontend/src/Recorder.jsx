import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Record enrolment footage straight from the camera.
 *
 * `prompt` is the section's own instruction — a gait section wants the person
 * walking across the frame, a face section wants them close and still — so the
 * guidance is specific to what is being captured rather than generic advice
 * nobody reads.
 */
export default function Recorder({ prompt, onRecorded, disabled }) {
  const videoRef = useRef(null)
  const recorderRef = useRef(null)
  const chunksRef = useRef([])
  const streamRef = useRef(null)

  const [active, setActive] = useState(false)
  const [recording, setRecording] = useState(false)
  const [seconds, setSeconds] = useState(0)
  const [error, setError] = useState('')

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    setActive(false)
    setRecording(false)
  }, [])

  // Releasing the camera on unmount matters: a webcam left streaming keeps its
  // light on, which looks exactly like covert recording.
  useEffect(() => stop, [stop])

  useEffect(() => {
    if (!recording) return undefined
    const timer = setInterval(() => setSeconds((value) => value + 1), 1000)
    return () => clearInterval(timer)
  }, [recording])

  const start = async () => {
    setError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) videoRef.current.srcObject = stream
      setActive(true)
    } catch (exception) {
      setError(
        `Could not open the camera: ${exception.message}. Browsers only allow ` +
          'camera access over https or on localhost.',
      )
    }
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
      onRecorded(
        new File([blob], `recording-${stamp}.webm`, { type: 'video/webm' }),
      )
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
    <div style={{ marginTop: 10 }}>
      <video ref={videoRef} autoPlay playsInline muted className="preview-video" />
      {prompt && (
        <p className="muted small" style={{ marginTop: 8 }}>
          {prompt}
        </p>
      )}
      <div className="actions" style={{ marginTop: 10 }}>
        {recording ? (
          <button type="button" className="btn btn-primary" onClick={endRecording}>
            Stop — {seconds}s
          </button>
        ) : (
          <button type="button" className="btn btn-primary" onClick={beginRecording}>
            Start recording
          </button>
        )}
        <button type="button" className="btn btn-quiet" onClick={stop}>
          Close camera
        </button>
      </div>
      {recording && seconds < 3 && (
        <p className="muted small" style={{ marginTop: 8 }}>
          Keep going — gait needs at least two seconds of walking.
        </p>
      )}
    </div>
  )
}
