# NoteForge-AI

NoteForge-AI is a multi-agent AI system that converts a syllabus into structured, exam-ready notes with unit-level questions and PDF export.<br>

---

## 🧠 Overview

The system uses an agent-based workflow to:

```text
- Break syllabus into topics
- Retrieve relevant content (PDF or web)
- Generate structured notes
- Refine output using a critic loop
- Produce final notes with important questions
```



## ✨ Features

```text
- Multi-agent pipeline (LangGraph)
- Works with:
    - Uploaded PDF (RAG)
    - Web fallback (Tavily)
- Self-improving notes using critic loop
- Structured, exam-oriented output
- Unit-level important questions
- PDF generation
- Caching for faster runs
```

---

## 🛠️ Installation

```bash
git clone https://github.com/Lingesh-7/NoteForge-AI.git
cd NoteForge-AI
pip install -r requirements.txt
```

---

## 🔐 Environment Setup

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_api_key
TAVILY_API_KEY=your_api_key
```

> ⚠️ **Important:** Never commit `.env` to GitHub. Ensure it is added to `.gitignore`.

---

## ▶️ Usage

```bash
streamlit run app.py
```

---

## 🔄 Steps

```text
1. Enter the syllabus
2. Upload reference PDF (optional)
3. Click Generate Notes
4. Download the structured PDF
```

---

## 📁 Project Structure

```bash
.
├── app.py
├── graph/
│   ├── nodes.py
│   ├── graph_builder.py
│   └── state.py
├── prompts/
├── rag/
├── utils/
├── fonts/
├── assets/
│   └── graph.png
├── requirements.txt
└── README.md
```

---

## ⚠️ Limitations

```text
- API rate limits may interrupt generation
- Large inputs increase latency
- Output quality depends on source quality
```

---

## 🚀 Roadmap

```text
[ ] Better PDF styling & formatting
[ ] Semantic caching (vector reuse)
[ ] Export to DOCX / Markdown
[ ] Fast vs High-Quality mode
[ ] Evaluation metrics for generated notes
```

---

## 🤝 Contributing

```text
Contributions, improvements, and ideas are welcome.
Feel free to open issues or submit pull requests.
```

---

## 📜 License

```text
MIT License
```

---

## 💡 Motivation

```text
Students spend hours converting scattered syllabus content into structured notes.
NoteForge-AI automates that entire workflow intelligently.
```

---

## ⭐ Support

```text
If you find this project useful, consider giving it a star ⭐
```
