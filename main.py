"""
NoteForge-AI — Streamlit entry point
=====================================
Session isolation:
  - API key, vectorstore, and generation state live entirely in st.session_state.
  - Temp files are PID-scoped so concurrent users do not collide.
  - A fresh graph instance is built per generation call.
"""

import os
import tempfile

import streamlit as st
from fpdf import FPDF
from groq import RateLimitError

from graph.graph_builder import build_graph
from rag.vectordb import create_vectorstore

# ── Constants ─────────────────────────────────────────────────────────────────
_PID = os.getpid()
TMP_DIR    = tempfile.gettempdir()
NOTES_PDF  = os.path.join(TMP_DIR, f"noteforge_notes_{_PID}.pdf")
UPLOAD_PDF = os.path.join(TMP_DIR, f"noteforge_upload_{_PID}.pdf")

GROQ_CONSOLE_URL = "https://console.groq.com/keys"
GROQ_VIDEO_URL   = "https://youtu.be/nt1PJu47nTk?si=n0nlmzKDwcEU18ar"

SECTION_LABELS = frozenset({
    "Definition", "Intuition", "Detailed Explanation",
    "Example", "Key Points", "Connection",
})
MARK_LABELS = frozenset({"2 Marks", "5 Marks", "10 Marks"})


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NoteForge AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

:root {
  --bg:         #0c0e14;
  --surface:    #13161f;
  --surface-2:  #191d2a;
  --border:     #242836;
  --accent:     #5b8def;
  --accent-2:   #a78bfa;
  --accent-dim: #1e3260;
  --green:      #22c55e;
  --amber:      #f59e0b;
  --text:       #e4e8f4;
  --muted:      #606880;
  --r:          12px;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background: var(--bg) !important;
  color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
  max-width: 820px !important;
  padding: 1.5rem 1.25rem 4rem !important;
  margin: 0 auto;
}

/* ── Hero ── */
.nf-hero {
  text-align: center;
  padding: 2.8rem 1rem 2rem;
}
.nf-logo {
  font-family: 'Syne', sans-serif;
  font-size: clamp(2.1rem, 7vw, 3.4rem);
  font-weight: 800;
  letter-spacing: -1.5px;
  background: linear-gradient(125deg, #5b8def 0%, #a78bfa 60%, #5b8def 100%);
  background-size: 200% auto;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: shimmer 4s linear infinite;
  line-height: 1.1;
  margin-bottom: .45rem;
}
@keyframes shimmer {
  0%   { background-position: 0% center; }
  100% { background-position: 200% center; }
}
.nf-tagline {
  color: var(--muted);
  font-size: clamp(.82rem, 2.2vw, 1rem);
  font-weight: 300;
  letter-spacing: .25px;
}

/* ── Card ── */
.nf-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 1.4rem 1.5rem 1.6rem;
  margin-bottom: 1rem;
}
.nf-card-title {
  font-family: 'Syne', sans-serif;
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: 1.8px;
  text-transform: uppercase;
  color: var(--accent);
  margin: 0 0 1rem;
}

/* ── Step pill row ── */
.nf-pill-row { display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: .85rem; }
.nf-pill {
  display: inline-flex; align-items: center; gap: .35rem;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: .32rem .9rem;
  font-size: .82rem;
  color: var(--text);
  text-decoration: none;
  transition: border-color .2s, color .2s;
  white-space: nowrap;
}
.nf-pill:hover { border-color: var(--accent); color: var(--accent); }

/* ── Key status ── */
.nf-badge {
  display: inline-flex; align-items: center; gap: .4rem;
  padding: .28rem .75rem;
  border-radius: 999px;
  font-size: .78rem;
  font-weight: 500;
  margin-top: .5rem;
}
.nf-badge-ok  { background:#14532d33; border:1px solid #22c55e44; color:var(--green); }
.nf-badge-err { background:#78350f33; border:1px solid #f59e0b44; color:var(--amber); }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
  background: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: .9rem !important;
  padding: .65rem .85rem !important;
  transition: border-color .2s, box-shadow .2s !important;
  caret-color: var(--accent);
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px #5b8def18 !important;
  outline: none !important;
}
.stTextInput button svg { color: var(--muted) !important; fill: var(--muted) !important; }

[data-testid="stFileUploader"] section {
  background: var(--surface-2) !important;
  border: 1.5px dashed var(--border) !important;
  border-radius: 8px !important;
}
[data-testid="stFileUploader"] label { color: var(--muted) !important; }
[data-testid="stFileUploadDropzone"] { color: var(--muted) !important; }

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 9px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: .92rem !important;
  letter-spacing: .3px !important;
  padding: .72rem 1.5rem !important;
  width: 100% !important;
  transition: opacity .2s, transform .12s !important;
  cursor: pointer !important;
  box-shadow: 0 4px 20px #5b8def25 !important;
}
.stButton > button:hover  { opacity: .86 !important; }
.stButton > button:active { transform: scale(.985) !important; }
.stButton > button:disabled { opacity: .4 !important; cursor: not-allowed !important; }

.stDownloadButton > button {
  background: var(--surface-2) !important;
  border: 1.5px solid var(--accent) !important;
  color: var(--accent) !important;
  border-radius: 9px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: .88rem !important;
  padding: .65rem 1.4rem !important;
  width: 100% !important;
  transition: background .2s !important;
}
.stDownloadButton > button:hover { background: var(--accent-dim) !important; }

/* ── Progress ── */
.stProgress > div > div {
  background: var(--surface-2) !important;
  border-radius: 999px !important;
  height: 5px !important;
}
.stProgress > div > div > div > div {
  background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
  border-radius: 999px !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
  background: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  font-size: .88rem !important;
}

/* ── Labels ── */
.stTextArea label, .stTextInput label,
[data-testid="stFileUploader"] > label {
  color: var(--muted) !important;
  font-size: .8rem !important;
  letter-spacing: .3px !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }

/* ── Preview typography ── */
.nf-note-topic {
  font-family: 'Syne', sans-serif;
  font-size: 1.2rem;
  font-weight: 700;
  color: var(--accent);
  margin: 1.6rem 0 .2rem;
}
.nf-section-label {
  font-family: 'Syne', sans-serif;
  font-size: .7rem;
  font-weight: 700;
  letter-spacing: 1.6px;
  text-transform: uppercase;
  color: var(--muted);
  margin: 1rem 0 .25rem;
  border-left: 2px solid var(--accent);
  padding-left: .5rem;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Mobile ── */
@media (max-width: 600px) {
  .block-container { padding: 1rem .75rem 3rem !important; }
  .nf-card         { padding: 1.1rem 1rem 1.3rem; }
  .nf-logo         { letter-spacing: -1px; }
}
</style>
""", unsafe_allow_html=True)


# ── PDF builder ───────────────────────────────────────────────────────────────
def _build_pdf(text: str, dest: str) -> None:
    """Render *text* to a Unicode-safe PDF at *dest*."""
    pdf = FPDF()
    pdf.add_page()

    font_dir = os.path.join(os.path.dirname(__file__), "fonts")
    pdf.add_font("DejaVu",  "",  os.path.join(font_dir, "DejaVuSans.ttf"),      uni=True)
    pdf.add_font("DejaVu",  "B", os.path.join(font_dir, "DejaVuSans-Bold.ttf"), uni=True)
    pdf.set_auto_page_break(auto=True, margin=12)

    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            pdf.ln(3)
        elif line.startswith("# "):
            pdf.set_font("DejaVu", "B", 16)
            pdf.cell(0, 10, line[2:], ln=True)
        elif line in SECTION_LABELS or line in MARK_LABELS:
            pdf.set_font("DejaVu", "B", 13)
            pdf.multi_cell(0, 8, line)
        else:
            pdf.set_font("DejaVu", "", 12)
            pdf.multi_cell(0, 7, line)

    pdf.output(dest)


# ── Session state ─────────────────────────────────────────────────────────────
def _init() -> None:
    for k, v in {
        "api_ready":   False,
        "api_key":     "",
        "vectorstore": None,
        "generated":   False,
        "final_doc":   "",
    }.items():
        st.session_state.setdefault(k, v)

_init()


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="nf-hero">
  <div class="nf-logo">NoteForge AI</div>
  <div class="nf-tagline">Multi-agent study notes — from syllabus to polished PDF in minutes</div>
</div>
""", unsafe_allow_html=True)


# ── Step 1 — API key ──────────────────────────────────────────────────────────
st.markdown('<div class="nf-card">', unsafe_allow_html=True)
st.markdown('<div class="nf-card-title">Step 1 — Groq API Key</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="nf-pill-row">
  <a class="nf-pill" href="{GROQ_CONSOLE_URL}" target="_blank">🔑 Get free API key</a>
  <a class="nf-pill" href="{GROQ_VIDEO_URL}"   target="_blank">▶ Watch setup guide</a>
</div>
""", unsafe_allow_html=True)

raw_key = st.text_input(
    "API key",
    type="password",
    placeholder="Paste your Groq API key here (gsk_…)",
    label_visibility="collapsed",
)

if raw_key:
    st.session_state.update(api_key=raw_key, api_ready=True)
    os.environ["GROQ_API_KEY"] = raw_key
    st.markdown('<span class="nf-badge nf-badge-ok">✓ API key ready</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="nf-badge nf-badge-err">⚠ API key required</span>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ── Steps 2 & 3 — gated behind valid key ──────────────────────────────────────
if st.session_state["api_ready"]:

    # ── Step 2 — Inputs ───────────────────────────────────────────────────────
    st.markdown('<div class="nf-card">', unsafe_allow_html=True)
    st.markdown('<div class="nf-card-title">Step 2 — Syllabus & Source</div>', unsafe_allow_html=True)

    syllabus = st.text_area(
        "Syllabus",
        height=190,
        placeholder=(
            "Enter topics one per line. Use colon for subtopics, comma to separate them.\n\n"
            "Example:\n"
            "Fuzzy Logic: Membership Functions, Fuzzy Sets, Defuzzification\n"
            "Neural Networks: Perceptron, Backpropagation, Activation Functions\n"
            "Clustering: K-Means, DBSCAN, Hierarchical"
        ),
        label_visibility="collapsed",
    )

    pdf_file = st.file_uploader(
        "Reference book PDF (optional) — notes will be grounded in your book; "
        "otherwise the web is searched automatically.",
        type=["pdf"],
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Step 3 — Generate ─────────────────────────────────────────────────────
    st.markdown('<div class="nf-card">', unsafe_allow_html=True)
    st.markdown('<div class="nf-card-title">Step 3 — Generate</div>', unsafe_allow_html=True)

    status_slot   = st.empty()
    progress_slot = st.progress(0)

    if st.button("⚡ Generate Notes", use_container_width=True):
        if not syllabus.strip():
            st.error("Please enter your syllabus before generating.")
            st.stop()

        # ── Process PDF (session-local path) ──────────────────────────────
        vectorstore = None
        if pdf_file:
            with st.spinner("Indexing PDF — this may take a moment…"):
                with open(UPLOAD_PDF, "wb") as fh:
                    fh.write(pdf_file.read())
                vectorstore = create_vectorstore(UPLOAD_PDF)
            st.session_state["vectorstore"] = vectorstore
        else:
            vectorstore = st.session_state.get("vectorstore")

        # ── Build a fresh graph instance per run (no shared state) ────────
        graph = build_graph()

        run_state = {
            "syllabus":            syllabus,
            "has_book":            bool(vectorstore),
            "vectorstore":         vectorstore,
            "api_key":             st.session_state["api_key"],
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

        status_slot.info("Initialising agents…")
        current_state = None
        rate_limited  = False

        try:
            for step in graph.stream(run_state):
                node_name     = next(iter(step))
                current_state = step[node_name]

                topic   = current_state.get("current_topic", "")
                idx     = current_state.get("current_topic_index", 0)
                total   = len(current_state.get("topics", [])) or 1
                pct     = int(((idx + 1) / total) * 100)

                progress_slot.progress(pct)
                label = node_name.replace("_", " ").title()
                status_slot.info(
                    f"**{label}** — {topic}  ·  {pct}%"
                )

        except RateLimitError:
            rate_limited = True
            status_slot.warning(
                "Rate limit reached. Partial notes have been saved — "
                "wait a moment before trying again."
            )

        if current_state is None:
            st.error("Generation failed. Verify your API key and try again.")
            st.stop()

        # ── Assemble document ─────────────────────────────────────────────
        if rate_limited:
            notes  = current_state.get("all_notes",  [])
            topics = current_state.get("topics",      [])
            doc    = ""
            for i, note in enumerate(notes):
                heading = topics[i] if i < len(topics) else f"Topic {i + 1}"
                doc    += f"# {heading}\n\n{note}\n\n"
            doc += "\n⚠ Stopped early due to API rate limit.\n"
        else:
            doc = current_state.get("final_document", "")

        st.session_state.update(generated=True, final_doc=doc)

        with st.spinner("Compiling PDF…"):
            _build_pdf(doc, NOTES_PDF)

        progress_slot.progress(100)
        status_slot.success("Notes ready! ✓")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Output ────────────────────────────────────────────────────────────────
    if st.session_state["generated"] and st.session_state["final_doc"]:
        final_doc = st.session_state["final_doc"]

        if os.path.exists(NOTES_PDF):
            with open(NOTES_PDF, "rb") as fh:
                st.download_button(
                    "⬇ Download PDF",
                    fh,
                    file_name="NoteForge_Notes.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

        st.markdown("---")
        st.markdown('<div class="nf-card-title">Preview</div>', unsafe_allow_html=True)

        for raw in final_doc.split("\n"):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("# "):
                st.markdown(f'<div class="nf-note-topic">{line[2:]}</div>',
                            unsafe_allow_html=True)
                st.divider()
            elif line in SECTION_LABELS or line in MARK_LABELS:
                st.markdown(f'<div class="nf-section-label">{line}</div>',
                            unsafe_allow_html=True)
            elif line.startswith("- "):
                st.markdown(line)
            else:
                st.write(line)