# Research Paper Q&A — A RAG System

A retrieval-augmented generation (RAG) system that answers questions about academic papers by retrieving relevant passages directly from source PDFs.

Built as a self-directed project extending my MSc work in NLP and information retrieval.

## What It Does

Ask a natural language question and the system returns the most semantically relevant passages from the indexed documents, with the source file and page number for every result.

Unlike a keyword search, it matches on meaning — asking about "differential outcomes training" surfaces relevant discussion even where the exact phrase does not appear.

## How It Works

1. **Document loading** — PDFs parsed page by page using PyPDF
2. **Chunking** — split into 800-character chunks with 120-character overlap to preserve context across boundaries
3. **Embedding** — each chunk converted to a vector using `sentence-transformers/all-MiniLM-L6-v2`
4. **Indexing** — vectors stored in a FAISS index for fast similarity search
5. **Retrieval** — query embedded and matched against the index by cosine similarity

## Tech Stack

Python · LangChain · FAISS · Sentence-Transformers · Streamlit · PyPDF

## Running It

pip install streamlit langchain langchain-community langchain-text-splitters langchain-huggingface sentence-transformers faiss-cpu pypdf

streamlit run app.py

Place PDFs in the `docs/` folder before running.

## Design Decisions

**Retrieval only, no generation.** The system returns source passages rather than generating an answer. For academic material this is deliberate — every result is traceable to a page, and there is no risk of hallucinated content.

**Chunk size of 800 with overlap.** Academic writing carries arguments across sentences. Smaller chunks fragmented reasoning; the overlap prevents losing context at boundaries.

**MiniLM embeddings.** Runs locally with no API cost or latency, and performs well on this scale of corpus.

## Possible Extensions

- Add an LLM layer to synthesise retrieved passages into direct answers
- Hybrid search combining semantic and keyword (BM25) retrieval
- Re-ranking retrieved results with a cross-encoder

## Author

Dhanyashree Madappa Nethravathi — MSc Artificial Intelligence & Robotics, University of Hertfordshire