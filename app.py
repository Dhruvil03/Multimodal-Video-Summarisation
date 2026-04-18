"""
Flask Backend API
Multimodal Video Summarisation — Full Stack

Endpoints:
    POST /api/summarise/upload   — upload a video file
    POST /api/summarise/youtube  — paste a YouTube URL
    GET  /api/status/<job_id>    — poll job status
    GET  /api/frames/<job_id>    — get important frames

Install:
    pip install flask flask-cors yt-dlp

Run:
    python app.py
"""

import os
import uuid
import json
import subprocess
import threading
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────
UPLOAD_DIR   = Path("./uploads")
RESULTS_DIR  = Path("./results")
CACHE_DIR    = Path(".feature_cache")
CHECKPOINT   = "./checkpoints/best.pt"
MAX_FILE_MB  = 200

for d in [UPLOAD_DIR, RESULTS_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# In-memory job store  {job_id: {status, result, error}}
jobs = {}


# ─────────────────────────────────────────────
# Background worker
# ─────────────────────────────────────────────

def run_summarisation(job_id: str, video_path: str):
    """Runs the full pipeline in a background thread."""
    try:
        import torch
        import torch.nn.functional as F
        from evaluate import load_model, generate_summary
        from speech_feature_extractor import extract_all_features

        jobs[job_id]["status"] = "extracting"

        # Extract features
        feats  = extract_all_features(
            video_path,
            num_visual_frames=32,
            speech_segment_duration=2.0,
            cache_dir=str(CACHE_DIR),
        )
        visual = feats["visual"].unsqueeze(0)
        speech = feats["speech"].unsqueeze(0)

        jobs[job_id]["status"] = "generating"

        # Load model and generate
        device  = "cuda" if torch.cuda.is_available() else "cpu"
        model   = load_model(CHECKPOINT, device)
        summary = generate_summary(
            model, visual, speech, device,
            temperature=0.7,
            top_p=0.9,
            max_len=100,
        )

        # Save top frames (evenly sampled indices shown to user)
        import cv2
        cap     = cv2.VideoCapture(video_path)
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = [int(total * i / 8) for i in range(8)]
        frames  = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                fname = f"{job_id}_frame_{idx}.jpg"
                cv2.imwrite(str(RESULTS_DIR / fname), frame)
                frames.append(fname)
        cap.release()

        # ROUGE scores (if reference available — skip silently if not)
        rouge_scores = {}
        try:
            from rouge_score import rouge_scorer as rs
            # Dummy ref for demo — in production pass real reference
            scorer = rs.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
            # We skip real ROUGE here since no reference available for arbitrary video
        except Exception:
            pass

        jobs[job_id].update({
            "status":  "done",
            "summary": summary,
            "frames":  frames,
            "rouge":   rouge_scores,
        })
        logger.info(f"Job {job_id} complete: {summary[:60]}...")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        jobs[job_id].update({"status": "error", "error": str(e)})


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/summarise/upload", methods=["POST"])
def summarise_upload():
    """Accept a video file upload and start summarisation."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Check size
    file.seek(0, 2)
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)
    if size_mb > MAX_FILE_MB:
        return jsonify({"error": f"File too large (max {MAX_FILE_MB}MB)"}), 400

    job_id     = str(uuid.uuid4())[:8]
    video_path = str(UPLOAD_DIR / f"{job_id}.mp4")
    file.save(video_path)

    jobs[job_id] = {"status": "queued", "summary": None, "frames": [], "rouge": {}}
    thread = threading.Thread(
        target=run_summarisation, args=(job_id, video_path), daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/summarise/youtube", methods=["POST"])
def summarise_youtube():
    """Accept a YouTube URL, download, and start summarisation."""
    data = request.get_json()
    url  = data.get("url", "").strip()

    if not url or "youtube.com" not in url and "youtu.be" not in url:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    job_id     = str(uuid.uuid4())[:8]
    video_path = str(UPLOAD_DIR / f"{job_id}.mp4")

    jobs[job_id] = {"status": "downloading", "summary": None, "frames": [], "rouge": {}}

    def download_and_run():
        try:
            result = subprocess.run([
                "yt-dlp",
                "-f", "bestvideo[vcodec^=avc][ext=mp4]+bestaudio[ext=m4a]/mp4",
                "--merge-output-format", "mp4",
                "-o", video_path,
                "--quiet", "--no-warnings", url,
            ], capture_output=True, text=True)

            if not Path(video_path).exists():
                jobs[job_id].update({
                    "status": "error",
                    "error": "Failed to download video"
                })
                return

            run_summarisation(job_id, video_path)

        except Exception as e:
            jobs[job_id].update({"status": "error", "error": str(e)})

    thread = threading.Thread(target=download_and_run, daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>", methods=["GET"])
def get_status(job_id):
    """Poll job status."""
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    job = jobs[job_id]

    # Map status to user-friendly message
    messages = {
        "queued":      "Queued...",
        "downloading": "Downloading video...",
        "extracting":  "Extracting visual & speech features...",
        "generating":  "Generating summary...",
        "done":        "Complete",
        "error":       "Error",
    }

    return jsonify({
        "status":  job["status"],
        "message": messages.get(job["status"], "Processing..."),
        "summary": job.get("summary"),
        "frames":  job.get("frames", []),
        "rouge":   job.get("rouge", {}),
        "error":   job.get("error"),
    })


@app.route("/api/frames/<filename>", methods=["GET"])
def get_frame(filename):
    """Serve a saved frame image."""
    path = RESULTS_DIR / filename
    if not path.exists():
        return jsonify({"error": "Frame not found"}), 404
    return send_file(str(path), mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
