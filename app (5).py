import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="Research Paper Q&A", layout="wide")

st.title("Research Paper Q&A")
st.caption("Ask questions about my MSc research papers — answers are retrieved directly from the source documents.")

# Use the docs/ folder if it exists, otherwise index PDFs in the project root
DOCS_DIR = "docs" if os.path.isdir("docs") else "."

@st.cache_resource
def build_index():
    """Load PDFs, split into chunks, embed, and build a FAISS vector store."""
    documents = []

    if not os.path.exists(DOCS_DIR):
        return None

    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        return None

    for filename in pdf_files:
        loader = PyPDFLoader(os.path.join(DOCS_DIR, filename))
        pages = loader.load()
        for page in pages:
            page.metadata["source_file"] = filename
        documents.extend(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
    chunks = splitter.split_documents(documents)

    # Normalised vectors make FAISS distance rank results the same way as cosine similarity
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore, len(pdf_files), len(chunks)


with st.spinner("Loading and indexing documents..."):
    result = build_index()

if result is None:
    st.error("No PDFs found. Please add PDF files to the 'docs' folder and restart.")
    st.stop()

vectorstore, num_files, num_chunks = result

st.success(f"Indexed {num_files} document(s) into {num_chunks} searchable chunks.")

st.divider()

query = st.text_input(
    "Ask a question about the research:",
    placeholder="e.g. What is Differential Outcomes Training?"
)

top_k = st.slider("Number of passages to retrieve", 1, 8, 4)

if query:
    with st.spinner("Searching..."):
        results = vectorstore.similarity_search_with_score(query, k=top_k)

    st.subheader("Retrieved passages")

    for i, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source_file", "unknown")
        # PyPDF counts pages from 0, so show the printed page label (or page + 1)
        page = doc.metadata.get("page_label") or (doc.metadata.get("page", -1) + 1)

        with st.expander(f"Result {i} — {source} (page {page}) — distance {score:.3f} (lower = closer match)"):
            st.write(doc.page_content)

    st.divider()
    st.caption(
        "This is a retrieval system: it finds the most semantically relevant "
        "passages from the source documents rather than generating new text. "
        "Every answer is traceable to a specific page."
    )