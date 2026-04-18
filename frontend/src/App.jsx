import { useState, useRef, useCallback } from "react";

const API = "http://localhost:5000/api";

const STATUS_STEPS = ["downloading", "extracting", "generating", "done"];
const STATUS_LABELS = {
  queued:      "Queued",
  downloading: "Downloading video",
  extracting:  "Extracting features",
  generating:  "Generating summary",
  done:        "Complete",
  error:       "Error",
};

// ── Styles ────────────────────────────────────
const css = `
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Mono:wght@300;400&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #080c14;
    --surface:  #0e1420;
    --border:   #1c2535;
    --accent:   #4f8ef7;
    --accent2:  #22d3a5;
    --text:     #e2e8f4;
    --muted:    #5a6a82;
    --error:    #f87171;
    --success:  #34d399;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Mono', monospace;
    min-height: 100vh;
  }

  .app {
    max-width: 900px;
    margin: 0 auto;
    padding: 48px 24px;
  }

  /* Header */
  .header { margin-bottom: 52px; }
  .header-tag {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--accent2);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 14px;
  }
  .header h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(28px, 5vw, 48px);
    font-weight: 800;
    line-height: 1.1;
    color: var(--text);
  }
  .header h1 span { color: var(--accent); }
  .header p {
    margin-top: 14px;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.7;
    max-width: 520px;
  }

  /* Tabs */
  .tabs {
    display: flex;
    gap: 2px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 4px;
    width: fit-content;
    margin-bottom: 28px;
  }
  .tab {
    padding: 8px 20px;
    border-radius: 7px;
    border: none;
    cursor: pointer;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    transition: all 0.2s;
    background: transparent;
    color: var(--muted);
  }
  .tab.active {
    background: var(--accent);
    color: #fff;
  }

  /* Input card */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 28px;
    margin-bottom: 24px;
  }

  /* Drop zone */
  .dropzone {
    border: 2px dashed var(--border);
    border-radius: 10px;
    padding: 48px 24px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    position: relative;
  }
  .dropzone:hover, .dropzone.drag { border-color: var(--accent); background: rgba(79,142,247,0.04); }
  .dropzone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
  .dropzone-icon { font-size: 36px; margin-bottom: 12px; }
  .dropzone-label { font-size: 13px; color: var(--muted); }
  .dropzone-label strong { color: var(--accent); }
  .dropzone-file {
    margin-top: 14px;
    padding: 10px 14px;
    background: rgba(79,142,247,0.08);
    border: 1px solid rgba(79,142,247,0.2);
    border-radius: 8px;
    font-size: 12px;
    color: var(--accent);
  }

  /* URL input */
  .url-row { display: flex; gap: 10px; }
  .url-input {
    flex: 1;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    color: var(--text);
    font-family: 'DM Mono', monospace;
    font-size: 13px;
    outline: none;
    transition: border-color 0.2s;
  }
  .url-input:focus { border-color: var(--accent); }
  .url-input::placeholder { color: var(--muted); }

  /* Button */
  .btn {
    padding: 12px 28px;
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
  }
  .btn:hover { background: #6ba3f9; transform: translateY(-1px); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
  .btn-full { width: 100%; margin-top: 16px; }

  /* Progress */
  .progress-card { padding: 24px 28px; }
  .progress-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
  }
  .progress-title { font-family: 'Syne', sans-serif; font-weight: 600; font-size: 14px; }
  .status-badge {
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 20px;
    background: rgba(79,142,247,0.12);
    color: var(--accent);
    letter-spacing: 1px;
    text-transform: uppercase;
  }
  .status-badge.done { background: rgba(34,211,165,0.12); color: var(--accent2); }
  .status-badge.error { background: rgba(248,113,113,0.12); color: var(--error); }

  .steps { display: flex; gap: 0; margin-bottom: 8px; }
  .step {
    flex: 1;
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    margin-right: 4px;
    transition: background 0.4s;
    overflow: hidden;
    position: relative;
  }
  .step.active::after {
    content: '';
    position: absolute;
    inset: 0;
    background: var(--accent);
    animation: pulse-bar 1.2s ease-in-out infinite;
  }
  .step.complete { background: var(--accent2); }
  @keyframes pulse-bar {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  .step-labels {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--muted);
    margin-top: 6px;
  }

  /* Results */
  .results-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
  }
  @media (max-width: 600px) { .results-grid { grid-template-columns: 1fr; } }

  .result-label {
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
  }
  .summary-text {
    font-size: 15px;
    line-height: 1.7;
    color: var(--text);
    font-family: 'Syne', sans-serif;
    font-weight: 400;
  }

  /* ROUGE scores */
  .rouge-row { display: flex; gap: 10px; }
  .rouge-chip {
    flex: 1;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
  }
  .rouge-val {
    font-family: 'Syne', sans-serif;
    font-size: 20px;
    font-weight: 800;
    color: var(--accent2);
  }
  .rouge-name { font-size: 10px; color: var(--muted); margin-top: 2px; }

  /* Frames */
  .frames-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-top: 10px;
  }
  @media (max-width: 600px) { .frames-grid { grid-template-columns: repeat(2, 1fr); } }
  .frame-img {
    width: 100%;
    aspect-ratio: 16/9;
    object-fit: cover;
    border-radius: 6px;
    border: 1px solid var(--border);
    transition: transform 0.2s;
  }
  .frame-img:hover { transform: scale(1.03); }

  /* Error */
  .error-box {
    padding: 14px 18px;
    background: rgba(248,113,113,0.08);
    border: 1px solid rgba(248,113,113,0.2);
    border-radius: 8px;
    color: var(--error);
    font-size: 13px;
  }

  /* Divider */
  .divider {
    height: 1px;
    background: var(--border);
    margin: 20px 0;
  }

  .reset-btn {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
    padding: 8px 18px;
    border-radius: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.2s;
    margin-top: 16px;
  }
  .reset-btn:hover { border-color: var(--accent); color: var(--accent); }
`;

// ── Component ─────────────────────────────────
export default function App() {
  const [tab, setTab]           = useState("upload");
  const [file, setFile]         = useState(null);
  const [url, setUrl]           = useState("");
  const [drag, setDrag]         = useState(false);
  const [jobId, setJobId]       = useState(null);
  const [jobData, setJobData]   = useState(null);
  const [loading, setLoading]   = useState(false);
  const pollRef                 = useRef(null);

  // ── Poll status ─────────────────────────────
  const startPolling = useCallback((id) => {
    setJobId(id);
    pollRef.current = setInterval(async () => {
      try {
        const res  = await fetch(`${API}/status/${id}`);
        const data = await res.json();
        setJobData(data);
        if (data.status === "done" || data.status === "error") {
          clearInterval(pollRef.current);
          setLoading(false);
        }
      } catch (e) {
        clearInterval(pollRef.current);
        setLoading(false);
      }
    }, 2000);
  }, []);

  // ── Submit upload ────────────────────────────
  const submitUpload = async () => {
    if (!file) return;
    setLoading(true);
    setJobData({ status: "queued", message: "Queued..." });

    const form = new FormData();
    form.append("video", file);
    const res  = await fetch(`${API}/summarise/upload`, { method: "POST", body: form });
    const data = await res.json();
    if (data.job_id) startPolling(data.job_id);
    else { setLoading(false); setJobData({ status: "error", error: data.error }); }
  };

  // ── Submit YouTube URL ───────────────────────
  const submitUrl = async () => {
    if (!url) return;
    setLoading(true);
    setJobData({ status: "downloading", message: "Downloading video..." });

    const res  = await fetch(`${API}/summarise/youtube`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();
    if (data.job_id) startPolling(data.job_id);
    else { setLoading(false); setJobData({ status: "error", error: data.error }); }
  };

  // ── Reset ────────────────────────────────────
  const reset = () => {
    clearInterval(pollRef.current);
    setFile(null); setUrl(""); setJobId(null); setJobData(null); setLoading(false);
  };

  // ── Step index ───────────────────────────────
  const stepIdx = jobData ? STATUS_STEPS.indexOf(jobData.status) : -1;

  const isDone  = jobData?.status === "done";
  const isError = jobData?.status === "error";

  return (
    <>
      <style>{css}</style>
      <div className="app">

        {/* Header */}
        <div className="header">
          <div className="header-tag">AI · Vision · Language</div>
          <h1>Multimodal<br /><span>Video Summariser</span></h1>
          <p>Upload a video or paste a YouTube URL to generate an AI summary using CLIP, Wav2Vec2, and a Conformer-GPT2 pipeline.</p>
        </div>

        {/* Input */}
        {!jobData && (
          <div className="card">
            <div className="tabs">
              <button className={`tab ${tab === "upload" ? "active" : ""}`} onClick={() => setTab("upload")}>Upload file</button>
              <button className={`tab ${tab === "youtube" ? "active" : ""}`} onClick={() => setTab("youtube")}>YouTube URL</button>
            </div>

            {tab === "upload" ? (
              <>
                <div
                  className={`dropzone ${drag ? "drag" : ""}`}
                  onDragOver={e => { e.preventDefault(); setDrag(true); }}
                  onDragLeave={() => setDrag(false)}
                  onDrop={e => { e.preventDefault(); setDrag(false); setFile(e.dataTransfer.files[0]); }}
                >
                  <input type="file" accept="video/*" onChange={e => setFile(e.target.files[0])} />
                  <div className="dropzone-icon">🎬</div>
                  <div className="dropzone-label">
                    <strong>Click to upload</strong> or drag and drop<br />
                    MP4, MOV, AVI — max 200MB
                  </div>
                  {file && <div className="dropzone-file">📎 {file.name} ({(file.size/1024/1024).toFixed(1)}MB)</div>}
                </div>
                <button className="btn btn-full" onClick={submitUpload} disabled={!file || loading}>
                  {loading ? "Processing..." : "Generate Summary"}
                </button>
              </>
            ) : (
              <>
                <div className="url-row">
                  <input
                    className="url-input"
                    placeholder="https://www.youtube.com/watch?v=..."
                    value={url}
                    onChange={e => setUrl(e.target.value)}
                    onKeyDown={e => e.key === "Enter" && submitUrl()}
                  />
                  <button className="btn" onClick={submitUrl} disabled={!url || loading}>
                    {loading ? "..." : "Summarise"}
                  </button>
                </div>
              </>
            )}
          </div>
        )}

        {/* Progress + Results */}
        {jobData && (
          <div className="card progress-card">
            <div className="progress-header">
              <div className="progress-title">
                {isDone ? "Summary ready" : isError ? "Something went wrong" : "Processing your video"}
              </div>
              <div className={`status-badge ${isDone ? "done" : isError ? "error" : ""}`}>
                {STATUS_LABELS[jobData.status] || jobData.status}
              </div>
            </div>

            {/* Step bar */}
            {!isError && (
              <>
                <div className="steps">
                  {STATUS_STEPS.map((s, i) => (
                    <div
                      key={s}
                      className={`step ${i < stepIdx ? "complete" : i === stepIdx ? "active" : ""}`}
                    />
                  ))}
                </div>
                <div className="step-labels">
                  <span>Download</span><span>Extract</span><span>Generate</span><span>Done</span>
                </div>
              </>
            )}

            {/* Error */}
            {isError && (
              <div className="error-box">⚠ {jobData.error || "An unexpected error occurred."}</div>
            )}

            {/* Results */}
            {isDone && (
              <>
                <div className="divider" />

                <div className="results-grid">
                  {/* Summary */}
                  <div>
                    <div className="result-label">Generated Summary</div>
                    <div className="summary-text">
                      {jobData.summary || "No summary generated."}
                    </div>
                  </div>

                  {/* ROUGE */}
                  <div>
                    <div className="result-label">ROUGE Scores</div>
                    {jobData.rouge && Object.keys(jobData.rouge).length > 0 ? (
                      <div className="rouge-row">
                        {["rouge1","rouge2","rougeL"].map(k => (
                          <div className="rouge-chip" key={k}>
                            <div className="rouge-val">{((jobData.rouge[k] || 0) * 100).toFixed(1)}</div>
                            <div className="rouge-name">{k.toUpperCase()}</div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.6 }}>
                        ROUGE scores require a reference summary.<br />
                        Available when evaluating on YouCook2 val set.
                      </div>
                    )}
                  </div>
                </div>

                {/* Frames */}
                {jobData.frames?.length > 0 && (
                  <>
                    <div className="result-label" style={{ marginTop: 20 }}>Key Frames</div>
                    <div className="frames-grid">
                      {jobData.frames.map(f => (
                        <img
                          key={f}
                          src={`${API}/frames/${f}`}
                          className="frame-img"
                          alt="frame"
                        />
                      ))}
                    </div>
                  </>
                )}
              </>
            )}

            <button className="reset-btn" onClick={reset}>← New video</button>
          </div>
        )}

      </div>
    </>
  );
}
