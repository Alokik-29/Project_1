import { useState, useRef, useEffect } from 'react'
import './App.css'

const API_URL = 'http://127.0.0.1:8000/predict'
const BAR_COUNT = 24

function App() {
  const [recording, setRecording] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [levels, setLevels] = useState(Array(BAR_COUNT).fill(4))

  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])
  const audioCtxRef = useRef(null)
  const analyserRef = useRef(null)
  const rafRef = useRef(null)

  const stopVisualizer = () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    audioCtxRef.current?.close()
    setLevels(Array(BAR_COUNT).fill(4))
  }

  useEffect(() => stopVisualizer, [])

  const runVisualizer = (stream) => {
    const ctx = new (window.AudioContext || window.webkitAudioContext)()
    const source = ctx.createMediaStreamSource(stream)
    const analyser = ctx.createAnalyser()
    analyser.fftSize = 64
    source.connect(analyser)
    audioCtxRef.current = ctx
    analyserRef.current = analyser

    const data = new Uint8Array(analyser.frequencyBinCount)
    const tick = () => {
      analyser.getByteFrequencyData(data)
      const step = Math.floor(data.length / BAR_COUNT)
      const next = Array.from({ length: BAR_COUNT }, (_, i) => {
        const v = data[i * step] || 0
        return 4 + (v / 255) * 36
      })
      setLevels(next)
      rafRef.current = requestAnimationFrame(tick)
    }
    tick()
  }

  const startRecording = async () => {
    setError(null)
    setResult(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (e) => chunksRef.current.push(e.data)

      mediaRecorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        stream.getTracks().forEach((track) => track.stop())
        stopVisualizer()
        await sendAudio(blob)
      }

      mediaRecorder.start()
      runVisualizer(stream)
      setRecording(true)
    } catch (err) {
      setError('Microphone access denied — ' + err.message)
    }
  }

  const stopRecording = () => {
    mediaRecorderRef.current?.stop()
    setRecording(false)
  }

  const handleFileUpload = (e) => {
    const file = e.target.files[0]
    if (file) {
      setError(null)
      setResult(null)
      sendAudio(file)
    }
  }

  const sendAudio = async (audioBlob) => {
    setLoading(true)
    try {
      const formData = new FormData()
      formData.append('file', audioBlob, 'recording.webm')

      const response = await fetch(API_URL, { method: 'POST', body: formData })
      if (!response.ok) throw new Error(`server responded ${response.status}`)

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError('Analysis failed — ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="unit">
      <div className="unit-label">
        <span className="unit-name">VOXMETER</span>
        <span className="unit-model">mk.1 — emotion analysis</span>
      </div>

      <div className="screen">
        {result ? (
          <div className="readout">
            <div className="readout-row">
              <span className="readout-label">emotion</span>
              <span className="readout-value">{result.emotion}</span>
            </div>
            <div className="meter">
              <div
                className="meter-fill"
                style={{ width: `${result.confidence * 100}%` }}
              />
            </div>
            <div className="readout-row">
              <span className="readout-label">confidence</span>
              <span className="readout-value small">
                {(result.confidence * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        ) : error ? (
          <div className="screen-message error">{error}</div>
        ) : loading ? (
          <div className="screen-message">analyzing signal…</div>
        ) : (
          <div className="bars" aria-hidden="true">
            {levels.map((h, i) => (
              <span key={i} className="bar" style={{ height: `${h}px` }} />
            ))}
          </div>
        )}
        {!result && !error && !loading && (
          <div className="screen-caption">
            {recording ? 'listening' : 'awaiting input'}
          </div>
        )}
      </div>

      <div className="controls">
        {!recording ? (
          <button onClick={startRecording} disabled={loading} className="btn primary">
            <span className="dot" /> record
          </button>
        ) : (
          <button onClick={stopRecording} className="btn stop">
            <span className="square" /> stop
          </button>
        )}

        <label className="btn ghost">
          upload clip
          <input
            type="file"
            accept="audio/*"
            onChange={handleFileUpload}
            disabled={loading}
            hidden
          />
        </label>
      </div>

      <div className="unit-footer">
        <span>AST transfer-learned · RAVDESS</span>
      </div>
    </div>
  )
}

export default App
