from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def create_vectorstore(pdf_path: str):
    # ── Load PDF ──────────────────────────────────────────────────────────────
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    if not documents:
        raise ValueError(
            "Could not extract any text from the PDF. "
            "It may be a scanned/image-only PDF. "
            "Please use a text-based PDF or omit it to use web search."
        )

    # ── Split into chunks ─────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    split_docs = splitter.split_documents(documents)

    if not split_docs:
        raise ValueError(
            "PDF was loaded but produced no text chunks. "
            "The file may be empty or contain only images."
        )

    # ── Embeddings ────────────────────────────────────────────────────────────
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ── Build vectorstore ─────────────────────────────────────────────────────
    return FAISS.from_documents(split_docs, embeddings)