import { useState } from 'react'

function App() {
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const runAnalysis = async () => {
    setLoading(true)
    setError(null)
    setReport(null)

    try {
      const response = await fetch('http://localhost:8002/analyze', {
        method: 'POST',
      })

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`)
      }

      // The /analyze endpoint returns the VerificationReport at the top level
      // ({ citations, flags, degraded_agents }), not wrapped in a `report` key.
      const data = await response.json()
      setReport(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: '800px', margin: '40px auto', padding: '0 20px', fontFamily: 'system-ui, sans-serif' }}>
      <h1>BS Detector</h1>
      <p>Legal brief verification pipeline</p>

      <button
        onClick={runAnalysis}
        disabled={loading}
        style={{
          padding: '10px 24px',
          fontSize: '16px',
          cursor: loading ? 'not-allowed' : 'pointer',
        }}
      >
        {loading ? 'Analyzing...' : 'Run Analysis'}
      </button>

      {error && (
        <div style={{ marginTop: '20px', color: 'red' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {report && (
        <div style={{ marginTop: '20px' }}>
          <h2>Report</h2>
          <pre style={{
            background: '#f5f5f5',
            padding: '20px',
            borderRadius: '4px',
            overflow: 'auto',
            whiteSpace: 'pre-wrap',
            wordWrap: 'break-word',
          }}>
            {typeof report === 'string' ? report : JSON.stringify(report, null, 2)}
          </pre>
        </div>
      )}

      {report === null && !loading && !error && (
        <p style={{ marginTop: '20px', color: '#888' }}>
          Click "Run Analysis" to analyze the case documents.
        </p>
      )}
    </div>
  )
}

export default App
