# RAG Q&A Pipeline

A clean, production-style **Retrieval-Augmented Generation** system
that answers questions over your own PDF / TXT / DOCX documents using
FAISS, Hugging Face embeddings, and OpenAI GPT.

---

## Project Structure

```
rag_project/
│
├── data/               ← put your PDFs / TXTs / DOCXs here
├── db/                 ← FAISS vector index saved here (auto-created)
│
├── rag_pipeline.py     ← 🔥 CORE LOGIC (everything connected)
├── config.py           ← all settings (model, chunk size, top-k, etc.)
├── utils.py            ← helper functions (formatting, prompts, display)
│
├── requirements.txt
└── README.md
```

---

## How RAG Works

```
Your Documents (PDF / TXT / DOCX)
        │
        ▼  load_documents()
Raw text pages
        │
        ▼  chunk_documents()
Overlapping chunks ──────────────────► FAISS Index (saved to db/)
        │                                      │
        ▼  HuggingFace embedding model         ▼  similarity_search()
  384-dim vectors                    Top-K relevant chunks
                                               │
                                               ▼  build_prompt()
                                     Context + Question
                                               │
                                               ▼  OpenAI GPT
                                     Grounded Answer ✓
```

**Why RAG over plain GPT?**
Standard LLMs hallucinate on private or recent documents because they
have no access to that content. RAG retrieves the *actual* relevant
text at query time and injects it as context — so the LLM answers from
your documents, not from imagination.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key
```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 3. Add your documents
Drop any `.pdf`, `.txt`, or `.docx` files into the `data/` folder.

### 4. Run

**Single question:**
```bash
python rag_pipeline.py --query "What are the main topics in these documents?"
```

**Interactive mode (chat session):**
```bash
python rag_pipeline.py --interactive
```

**Change prompt strategy:**
```bash
python rag_pipeline.py --query "Explain the deployment stages" --strategy chain_of_thought
```

**Force rebuild the vector index:**
```bash
python rag_pipeline.py --query "..." --rebuild
```

**Show retrieved chunks (debug):**
```bash
python rag_pipeline.py --query "..." --verbose
```

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `LLM_MODEL` | `gpt-3.5-turbo` | OpenAI model (`gpt-4o` for higher quality) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `500` | Max characters per document chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks |
| `TOP_K` | `4` | Number of chunks retrieved per query |
| `PROMPT_STRATEGY` | `zero_shot` | Prompt style (see below) |

---

## Prompt Strategies

| Strategy | Best For | Trade-off |
|---|---|---|
| `zero_shot` | Clear, factual questions | Fast, low token cost |
| `few_shot` | Structured or formatted output | Slightly more tokens |
| `chain_of_thought` | Complex, multi-part questions | Most accurate, most tokens |

Change the default in `config.py` or pass `--strategy` at runtime.

---

## Use in Your Own Code

```python
from rag_pipeline import setup_pipeline, run_rag_pipeline

# Build once
vector_store = setup_pipeline()

# Query as many times as you want
answer = run_rag_pipeline(
    question="What is the main finding?",
    vector_store=vector_store,
    strategy="chain_of_thought",
    verbose=True,
)
print(answer)
```

---

## How to Explain This in an Interview

> "I built a RAG pipeline to solve hallucination on private documents.
> Instead of relying on the LLM's training memory, I retrieve the actual
> relevant text at query time by embedding both the documents and the query
> using a Hugging Face sentence-transformer, then doing nearest-neighbour
> search with FAISS. The retrieved chunks are injected as context into a
> structured prompt before calling GPT. I implemented three prompting
> strategies — zero-shot, few-shot, and chain-of-thought — and the entire
> pipeline is modular: config.py controls all settings, utils.py handles
> shared helpers, and rag_pipeline.py is the single core file that connects
> every stage from document ingestion to answer generation."
