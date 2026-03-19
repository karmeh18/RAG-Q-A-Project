# RAG Q&A Pipeline v2

A production-style **Retrieval-Augmented Generation** system upgraded with
Hybrid Retrieval (FAISS + BM25), Reranking, and structured prompting.

---

## Project Structure

```
rag_project/
│
├── data/                        ← put your PDFs / TXTs / DOCXs here
├── db/                          ← FAISS vector index saved here (auto-created)
│
├── rag_pipeline.py              ← 🔥 CORE LOGIC (everything connected)
├── config.py                    ← all settings (model, chunk size, top-k, etc.)
├── utils.py                     ← helper functions
│
├── vectorstores/
│   └── bm25_store.py            ← BM25 keyword retrieval (Step 2)
│
├── retriever/
│   └── hybrid_retriever.py      ← FAISS + BM25 combined (Step 3)
│
├── reranker/
│   └── reranker.py              ← cosine similarity reranking (Step 4)
│
├── requirements.txt
└── README.md
```

---

## Upgraded Pipeline Flow

```
Your Documents (PDF / TXT / DOCX)
        │
        ▼  load_documents()
Raw LangChain Documents
        │
        ▼  chunk_documents()
        │  RecursiveCharacterTextSplitter (size=600, overlap=150)
Overlapping Chunks
        │
   ┌────┴─────┐
   ▼          ▼
 FAISS      BM25Store
 semantic   keyword
   │          │
   └────┬─────┘
        ▼  HybridRetriever.retrieve()
Merged + Deduplicated Candidates
        │
        ▼  rerank()
Top-3 Most Relevant Chunks
        │
        ▼  generate_answer()
Grounded Answer ✓
```

---

## What Was Upgraded (All 9 Steps)

| Step | Change | Why |
|---|---|---|
| 1 | Hybrid retrieval replaces single FAISS | Better recall — catches both semantic and keyword matches |
| 2 | BM25Store added | Exact keyword matching for technical terms, acronyms, names |
| 3 | HybridRetriever merges FAISS + BM25 | Deduplicates and combines both candidate sets |
| 4 | Reranker added | Filters ~16 candidates down to top-3 before LLM call |
| 5 | Structured prompt with source citations | Prevents hallucination, adds "I don't know" fallback |
| 6 | RecursiveCharacterTextSplitter, size=600, overlap=150 | Natural boundary splitting, more context overlap |
| 7 | Embedding model upgrade hook in config.py | Easy swap to stronger model without code changes |
| 8 | rank-bm25 added to requirements.txt | BM25 dependency |
| 9 | --test flag runs all 3 query types | Validates exact, conceptual, and mixed retrieval |

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
cp .env.example .env
# Edit .env → OPENAI_API_KEY=sk-...

# 3. Add your documents to data/
# Supported: .pdf  .txt  .docx

# 4. Run
python rag_pipeline.py --query "What are the main topics?"
```

---

## Usage

```bash
# Single question
python rag_pipeline.py --query "What does the document say about X?"

# Show retrieved chunks (debug mode)
python rag_pipeline.py --query "..." --verbose

# Interactive session
python rag_pipeline.py --interactive

# Run Step 9 test cases (exact / conceptual / mixed)
python rag_pipeline.py --test

# Force rebuild the FAISS index (after adding new documents)
python rag_pipeline.py --rebuild --query "..."
```

---

## Use in Your Own Code

```python
from rag_pipeline import setup_pipeline, run_rag_pipeline

# Build once at startup
vector_store, chunks, embedding_model = setup_pipeline()

# Query as many times as you want
answer = run_rag_pipeline(
    question="What is the main finding?",
    vector_store=vector_store,
    chunks=chunks,
    embedding_model=embedding_model,
    verbose=True,
)
print(answer)
```

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `LLM_MODEL` | `gpt-3.5-turbo` | OpenAI model — swap to `gpt-4o` for higher quality |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace model — swap to `all-mpnet-base-v2` for better accuracy |
| `CHUNK_SIZE` | `600` | Max characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |
| `TOP_K` | `8` | Chunks fetched from each retriever (FAISS + BM25) |
| `TOP_K_RERANK` | `3` | Final chunks sent to LLM after reranking |

---

## Step 9 — Test Query Types

After running `python rag_pipeline.py --test` you should see:

| Test Type | Example Query | What It Validates |
|---|---|---|
| Exact keyword | "What does the document say about FAISS?" | BM25 finds the exact term |
| Conceptual | "Explain the main idea of these documents" | FAISS finds semantic meaning |
| Mixed | "What indexing method is used for similarity search?" | Hybrid outperforms either alone |

Expected outcomes: better recall, more precise answers, fewer hallucinations.

---

## How to Explain This in an Interview

> "I upgraded a basic RAG pipeline with three key improvements.
> First, I added hybrid retrieval combining FAISS semantic search with BM25
> keyword search — FAISS finds conceptually related content while BM25 catches
> exact technical terms and acronyms, and together they produce higher recall
> than either alone. Second, I added a reranker that scores all candidates by
> cosine similarity to the query and keeps only the top-3 before calling the LLM
> — this reduces noise, cuts token cost, and improves answer precision.
> Third, I upgraded the prompt to a structured template with source citations
> and an explicit I-don't-know fallback, which significantly reduces hallucination.
> The entire pipeline is modular — config.py controls all settings and you can
> swap the embedding model or LLM without touching pipeline code."
