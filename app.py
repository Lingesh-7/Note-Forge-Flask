"""
NoteForge-AI — Flask entry point
=================================
Replaces Streamlit with a pure Flask + SSE (Server-Sent Events) stack.
- /                  → main page
- /generate          → POST: kicks off background generation, returns job_id
- /stream/<job_id>   → GET SSE: streams status updates
- /download/<job_id> → GET: streams the generated PDF
- /validate-key      → POST: quick Groq key check
"""

from __future__ import annotations

import io
import json
import os
import queue
import tempfile
import threading
import time
import uuid

from fpdf import FPDF
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    stream_with_context,
)
from groq import RateLimitError
from langchain_groq import ChatGroq

from graph.graph_builder import build_graph
from rag.vectordb import create_vectorstore

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "noteforge-secret-2024")

TMP_DIR = tempfile.gettempdir()

SECTION_LABELS = frozenset({
    "Definition", "Intuition", "Detailed Explanation",
    "Example", "Key Points", "Connection",
})
MARK_LABELS = frozenset({"2 Marks", "5 Marks", "10 Marks"})

# ── In-memory job store ───────────────────────────────────────────────────────
# job_id → {"queue": Queue, "status": str, "doc": str, "pdf_path": str, "error": str}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


# ── PDF builder ───────────────────────────────────────────────────────────────
def _build_pdf(text: str, dest: str) -> None:
    pdf = FPDF()
    pdf.set_margins(left=15, top=15, right=15)
    pdf.add_page()

    font_dir = os.path.join(os.path.dirname(__file__), "fonts")
    regular_font = os.path.join(font_dir, "DejaVuSans.ttf")
    bold_font    = os.path.join(font_dir, "DejaVuSans-Bold.ttf")

    if os.path.exists(regular_font) and os.path.exists(bold_font):
        pdf.add_font("DejaVu", "",  regular_font, uni=True)
        pdf.add_font("DejaVu", "B", bold_font,    uni=True)
        font_name = "DejaVu"
    else:
        font_name = "Helvetica"

    pdf.set_auto_page_break(auto=True, margin=15)

    effective_width = pdf.w - pdf.l_margin - pdf.r_margin

    for raw in text.split("\n"):
        # Sanitise: replace characters that can break fpdf2 layout
        line = raw.strip()
        line = line.replace("\x00", "").replace("\r", "")
        if not line:
            pdf.ln(3)
            continue

        try:
            if line.startswith("# "):
                pdf.set_font(font_name, "B", 16)
                pdf.multi_cell(effective_width, 10, line[2:])
                pdf.ln(1)
            elif line in SECTION_LABELS or line in MARK_LABELS:
                pdf.set_font(font_name, "B", 13)
                pdf.multi_cell(effective_width, 8, line)
            else:
                pdf.set_font(font_name, "", 11)
                pdf.multi_cell(effective_width, 6, line)
        except Exception:
            # Skip any single line that still can't render rather than aborting
            pdf.ln(3)

    pdf.output(dest)

# ── Background generation worker ──────────────────────────────────────────────
def _run_generation(job_id: str, syllabus: str, api_key: str,
                    pdf_path_upload: str | None) -> None:
    job = _jobs[job_id]
    q: queue.Queue = job["queue"]

    def push(event: str, data: dict) -> None:
        q.put({"event": event, "data": data})

    try:
        # Build vectorstore if PDF was uploaded
        vectorstore = None
        if pdf_path_upload and os.path.exists(pdf_path_upload):
            push("status", {"msg": "Indexing PDF…", "pct": 2})
            vectorstore = create_vectorstore(pdf_path_upload)

        graph = build_graph()

        run_state = {
            "syllabus":            syllabus,
            "has_book":            bool(vectorstore),
            "vectorstore":         vectorstore,
            "api_key":             api_key,
            "mode":                "In-Depth",
            "topics":              [],
            "current_topic_index": 0,
            "current_topic":       "",
            "all_notes":           [],
            "all_questions":       [],
            "retry_count":         0,
            "critic_feedback":     "",
            "critic_pass":         False,
            "research_content":    "",
            "draft_notes":         "",
            "unit_questions":      "",
            "final_document":      "",
        }

        push("status", {"msg": "Initialising agents…", "pct": 5})
        current_state = None
        rate_limited = False

        for step in graph.stream(run_state):
            node_name     = next(iter(step))
            current_state = step[node_name]

            topic = current_state.get("current_topic", "")
            idx   = current_state.get("current_topic_index", 0)
            total = len(current_state.get("topics", [])) or 1
            pct   = max(5, int(((idx + 1) / total) * 90))

            label = node_name.replace("_", " ").title()
            push("status", {
                "msg":   f"{label} — {topic}",
                "pct":   pct,
                "node":  node_name,
                "topic": topic,
                "idx":   idx,
                "total": total,
            })

    except RateLimitError:
        rate_limited = True
        push("status", {"msg": "Rate limit reached — saving partial notes…", "pct": 85})
        if current_state:
            notes  = current_state.get("all_notes", [])
            topics = current_state.get("topics", [])
            doc = ""
            for i, note in enumerate(notes):
                heading = topics[i] if i < len(topics) else f"Topic {i + 1}"
                doc += f"# {heading}\n\n{note}\n\n"
            doc += "\n⚠ Stopped early due to API rate limit.\n"
            current_state["final_document"] = doc

    except Exception as exc:
        push("error", {"msg": str(exc)})
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"]  = str(exc)
        return

    if current_state is None:
        push("error", {"msg": "Generation failed — no state returned."})
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"]  = "No state returned from graph."
        return

    doc = current_state.get("final_document", "")
    if not doc:
        # Assemble from partial notes
        notes  = current_state.get("all_notes", [])
        topics = current_state.get("topics", [])
        doc = "\n\n".join(
            f"# {topics[i] if i < len(topics) else f'Topic {i+1}'}\n\n{note}"
            for i, note in enumerate(notes)
        )

    push("status", {"msg": "Compiling PDF…", "pct": 95})
    pdf_dest = os.path.join(TMP_DIR, f"noteforge_{job_id}.pdf")

    try:
        _build_pdf(doc, pdf_dest)
    except Exception as exc:
        push("error", {"msg": f"PDF build failed: {exc}"})
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"]  = str(exc)
        return

    with _jobs_lock:
        _jobs[job_id]["status"]   = "done"
        _jobs[job_id]["doc"]      = doc
        _jobs[job_id]["pdf_path"] = pdf_dest

    push("done", {"msg": "Notes ready!", "pct": 100})


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/validate-key", methods=["POST"])
def validate_key():
    data    = request.get_json(force=True)
    api_key = (data.get("api_key") or "").strip()

    if not api_key or not api_key.startswith("gsk_"):
        return jsonify({"valid": False, "error": "Key must start with gsk_"})

    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            groq_api_key=api_key,
            max_tokens=5,
        )
        llm.invoke("Hi")
        return jsonify({"valid": True})
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "auth" in msg.lower() or "invalid" in msg.lower():
            return jsonify({"valid": False, "error": "Invalid API key — please check and try again."})
        if "429" in msg or "rate" in msg.lower():
            # Key is valid but rate-limited
            return jsonify({"valid": True})
        return jsonify({"valid": False, "error": f"Could not verify key: {msg[:120]}"})


@app.route("/generate", methods=["POST"])
def generate():
    api_key  = (request.form.get("api_key") or "").strip()
    syllabus = (request.form.get("syllabus") or "").strip()
    pdf_file = request.files.get("pdf_file")

    # ── Validation ────────────────────────────────────────────────────────────
    errors = {}
    if not api_key:
        errors["api_key"] = "Groq API key is required."
    elif not api_key.startswith("gsk_"):
        errors["api_key"] = "Invalid key format — must start with gsk_."
    if not syllabus:
        errors["syllabus"] = "Syllabus cannot be empty."
    if errors:
        return jsonify({"error": errors}), 400

    # ── Save uploaded PDF ─────────────────────────────────────────────────────
    upload_path = None
    if pdf_file and pdf_file.filename:
        if not pdf_file.filename.lower().endswith(".pdf"):
            return jsonify({"error": {"pdf_file": "Only PDF files are accepted."}}), 400
        upload_path = os.path.join(TMP_DIR, f"noteforge_up_{uuid.uuid4().hex}.pdf")
        pdf_file.save(upload_path)

    # ── Create job ────────────────────────────────────────────────────────────
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "queue":    queue.Queue(),
            "status":   "running",
            "doc":      "",
            "pdf_path": "",
            "error":    "",
        }

    os.environ["GROQ_API_KEY"] = api_key

    t = threading.Thread(
        target=_run_generation,
        args=(job_id, syllabus, api_key, upload_path),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/stream/<job_id>")
def stream(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return Response("data: {\"event\":\"error\",\"msg\":\"Job not found\"}\n\n",
                        mimetype="text/event-stream")

    def generate_sse():
        q: queue.Queue = job["queue"]
        while True:
            try:
                item = q.get(timeout=30)
                yield f"data: {json.dumps(item)}\n\n"
                if item["event"] in ("done", "error"):
                    break
            except queue.Empty:
                # Heartbeat to keep connection alive
                yield "data: {\"event\":\"heartbeat\"}\n\n"
                with _jobs_lock:
                    if _jobs.get(job_id, {}).get("status") in ("done", "error"):
                        break

    return Response(
        stream_with_context(generate_sse()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/result/<job_id>")
def result(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    status = job["status"]
    if status == "running":
        return jsonify({"status": "running"})
    if status == "error":
        return jsonify({"status": "error", "error": job["error"]}), 500

    return jsonify({"status": "done", "doc": job["doc"]})


@app.route("/download/<job_id>")
def download(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return "Not ready", 404

    pdf_path = job["pdf_path"]
    if not os.path.exists(pdf_path):
        return "PDF not found", 404

    with open(pdf_path, "rb") as fh:
        data = fh.read()

    return Response(
        data,
        mimetype="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="NoteForge_Notes.pdf"',
            "Content-Length": str(len(data)),
        },
    )


# ── Cleanup old jobs (simple GC) ─────────────────────────────────────────────
def _cleanup_jobs():
    """Remove jobs older than 30 minutes to avoid memory leaks."""
    while True:
        time.sleep(600)
        cutoff = time.time() - 1800
        with _jobs_lock:
            stale = [jid for jid, j in _jobs.items()
                     if j.get("created_at", cutoff) < cutoff]
            for jid in stale:
                pdf = _jobs[jid].get("pdf_path", "")
                if pdf and os.path.exists(pdf):
                    try:
                        os.remove(pdf)
                    except OSError:
                        pass
                del _jobs[jid]


threading.Thread(target=_cleanup_jobs, daemon=True).start()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
