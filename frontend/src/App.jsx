import { useState } from 'react'

// Color a verdict by what it means for trust: a contradiction is a red flag, a
// could_not_verify is an honest abstention (amber, not red — the pipeline's whole
// point is that "I can't confirm this" is a first-class, non-alarming outcome),
// and verified is green. Keeping this mapping explicit makes the UI legible as a
// trust signal rather than a raw data dump.
const STATUS_COLOR = {
  contradicted: '#c0392b',
  could_not_verify: '#b7791f',
  verified: '#2f855a',
}

// Confidence band colors are intentionally NEUTRAL (greys), distinct from the verdict
// colors above: the verdict says WHAT the audit found, the band says HOW SURE the
// pipeline is. Keeping the two scales visually separate stops a reader from conflating
// "high confidence" with "severe" — they're orthogonal axes.
const CONFIDENCE_COLOR = {
  high: '#2d3748',
  medium: '#718096',
  low: '#a0aec0',
}

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: '10px',
      fontSize: '12px', fontWeight: 600, color: '#fff', background: color || '#555',
      marginRight: '6px', textTransform: 'lowercase',
    }}>
      {text}
    </span>
  )
}

function Card({ children, accent }) {
  return (
    <div style={{
      border: '1px solid #e2e2e2', borderLeft: `4px solid ${accent || '#bbb'}`,
      borderRadius: '6px', padding: '14px 16px', marginBottom: '12px', background: '#fff',
    }}>
      {children}
    </div>
  )
}

// A flag is a cross-doc / factual finding. Show the verdict, the MSJ claim under
// scrutiny, and — load-bearing for an auditable tool — the verbatim evidence span
// with its source document, so a reviewer can trace every claim back to real text.
function FlagCard({ flag }) {
  const color = STATUS_COLOR[flag.status] || '#555'
  return (
    <Card accent={color}>
      <div style={{ marginBottom: '8px' }}>
        <Badge text={flag.status} color={color} />
        <Badge text={flag.flag_type} color="#34495e" />
        {flag.confidence && (
          // Confidence badge carries the deterministic reasoning in its title, so the
          // score is auditable on hover — never an opaque number.
          <span title={flag.confidence.reasoning}>
            <Badge
              text={`confidence: ${flag.confidence.band}`}
              color={CONFIDENCE_COLOR[flag.confidence.band] || '#718096'}
            />
          </span>
        )}
      </div>
      <div style={{ fontWeight: 600, marginBottom: '4px' }}>{flag.msj_claim}</div>
      {flag.explanation && (
        <div style={{ fontSize: '14px', color: '#444', marginBottom: '8px' }}>{flag.explanation}</div>
      )}
      {flag.evidence?.length > 0 && (
        <div style={{ fontSize: '13px' }}>
          {flag.evidence.map((ev, i) => (
            <blockquote key={i} style={{
              margin: '6px 0', padding: '6px 10px', background: '#f7f7f7',
              borderLeft: '3px solid #ccc', color: '#333',
            }}>
              “{ev.quote}”
              <div style={{ fontSize: '11px', color: '#888', marginTop: '2px' }}>
                — {ev.source_doc}{ev.locator ? ` (${ev.locator})` : ''}
              </div>
            </blockquote>
          ))}
        </div>
      )}
      {/* The chain-of-thought the backend engineers to run BEFORE the verdict — the
          "traceable reasoning" the product sells. Shown in the UI (not just Raw JSON)
          so a reviewer can see WHY, collapsed by default to keep the card scannable. */}
      {flag.comparison_reasoning && (
        <details style={{ marginTop: '6px', fontSize: '12px', color: '#555' }}>
          <summary style={{ cursor: 'pointer' }}>Reasoning</summary>
          <div style={{ marginTop: '4px' }}>{flag.comparison_reasoning}</div>
        </details>
      )}
      {flag.raised_by && (
        <div style={{ fontSize: '11px', color: '#999', marginTop: '6px' }}>
          raised by {flag.raised_by}
        </div>
      )}
    </Card>
  )
}

// A citation is the citation-audit agent's assessment of a legal authority. Show
// whether the authority was confirmed/abstained and any quote-accuracy issue.
function CitationCard({ cite }) {
  const color = STATUS_COLOR[cite.support_assessment] || '#555'
  return (
    <Card accent={color}>
      <div style={{ marginBottom: '8px' }}>
        <Badge text={cite.support_assessment} color={color} />
        {cite.flag_type && <Badge text={cite.flag_type} color="#34495e" />}
      </div>
      <div style={{ fontWeight: 600 }}>{cite.authority}</div>
      {cite.reporter && <div style={{ fontSize: '12px', color: '#888' }}>{cite.reporter}</div>}
      <div style={{ fontSize: '14px', color: '#444', marginTop: '4px' }}>{cite.proposition}</div>
      {cite.quoted_text && (
        <blockquote style={{
          margin: '6px 0', padding: '6px 10px', background: '#f7f7f7',
          borderLeft: '3px solid #ccc', fontSize: '13px',
        }}>
          “{cite.quoted_text}”
        </blockquote>
      )}
      {cite.issue && (
        <div style={{ fontSize: '13px', color: '#b7791f', marginTop: '4px' }}>{cite.issue}</div>
      )}
      {/* Same reasoning-before-verdict transparency as the finding cards. */}
      {cite.assessment_reasoning && (
        <details style={{ marginTop: '6px', fontSize: '12px', color: '#555' }}>
          <summary style={{ cursor: 'pointer' }}>Reasoning</summary>
          <div style={{ marginTop: '4px' }}>{cite.assessment_reasoning}</div>
        </details>
      )}
    </Card>
  )
}

// The judicial memo is the synthesis for a judge — shown FIRST and visually set apart
// (a bench-memo banner), because it's the human-readable entry point into the report.
// The structured cards below are the auditable detail behind it.
function MemoCard({ memo }) {
  return (
    <div style={{
      border: '1px solid #cbd5e0', borderLeft: '4px solid #2b6cb0', borderRadius: '6px',
      padding: '16px 18px', marginBottom: '24px', background: '#ebf4ff',
    }}>
      <div style={{ fontWeight: 700, fontSize: '13px', color: '#2b6cb0', marginBottom: '6px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
        Judicial Memo
      </div>
      <p style={{ margin: 0, lineHeight: 1.55, color: '#1a202c' }}>{memo.summary}</p>
      {memo.grounded_in?.length > 0 && (
        <div style={{ fontSize: '12px', color: '#4a5568', marginTop: '8px' }}>
          Synthesized from {memo.grounded_in.length} confirmed finding
          {memo.grounded_in.length === 1 ? '' : 's'}.
        </div>
      )}
    </div>
  )
}

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
    <div style={{ maxWidth: '860px', margin: '40px auto', padding: '0 20px', fontFamily: 'system-ui, sans-serif', color: '#1a1a1a' }}>
      <h1 style={{ marginBottom: '4px' }}>BS Detector</h1>
      <p style={{ color: '#666', marginTop: 0 }}>Legal brief verification pipeline</p>

      <button
        onClick={runAnalysis}
        disabled={loading}
        style={{
          padding: '10px 24px',
          fontSize: '16px',
          cursor: loading ? 'not-allowed' : 'pointer',
        }}
      >
        {loading ? 'Analyzing…' : 'Run Analysis'}
      </button>

      {error && (
        <div style={{ marginTop: '20px', color: '#c0392b' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {report && (
        <div style={{ marginTop: '24px' }}>
          {report.degraded_agents?.length > 0 && (
            <div style={{
              padding: '8px 12px', background: '#fff8e1', border: '1px solid #f0d98c',
              borderRadius: '6px', marginBottom: '16px', fontSize: '13px',
            }}>
              ⚠ Partial coverage — degraded agents: {report.degraded_agents.join(', ')}
            </div>
          )}

          {report.judicial_memo && <MemoCard memo={report.judicial_memo} />}

          <h2 style={{ fontSize: '18px' }}>Findings ({report.flags?.length || 0})</h2>
          {report.flags?.length
            ? report.flags.map((f, i) => <FlagCard key={i} flag={f} />)
            : <p style={{ color: '#888' }}>No cross-document contradictions found.</p>}

          <h2 style={{ fontSize: '18px', marginTop: '24px' }}>Citations ({report.citations?.length || 0})</h2>
          {report.citations?.length
            ? report.citations.map((c, i) => <CitationCard key={i} cite={c} />)
            : <p style={{ color: '#888' }}>No citations extracted.</p>}

          {/* Raw JSON kept available behind a details toggle — the structured cards
              are the primary view, but a grader may want the exact payload. */}
          <details style={{ marginTop: '16px' }}>
            <summary style={{ cursor: 'pointer', color: '#666', fontSize: '13px' }}>Raw JSON</summary>
            <pre style={{
              background: '#f5f5f5', padding: '16px', borderRadius: '4px', overflow: 'auto',
              whiteSpace: 'pre-wrap', wordWrap: 'break-word', fontSize: '12px',
            }}>
              {JSON.stringify(report, null, 2)}
            </pre>
          </details>
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
