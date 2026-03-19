"""
rag_pipeline.py
---------------
🔥 CORE LOGIC — everything connected in one file.

UPGRADED PIPELINE (v2) — all 9 improvements applied:
  ✅ Step 1 — Hybrid retrieval (FAISS + BM25) replaces single FAISS retriever
  ✅ Step 2 — BM25Store for keyword-based search
  ✅ Step 3 — HybridRetriever merges semantic + keyword candidates
  ✅ Step 4 — Reranker scores and filters to top-K before sending to LLM
  ✅ Step 5 — Structured prompt with source citations and "I don't know" fallback
  ✅ Step 6 — RecursiveCharacterTextSplitter with chunk_size=600, overlap=150
  ✅ Step 7 — Embedding model upgrade hook (swap in config.py)
  ✅ Step 8 — rank-bm25 added to requirements.txt
  ✅ Step 9 — Test cases via --test flag

FULL UPGRADED FLOW:
───────────────────
  Your Documents (PDF / TXT / DOCX)
          │
          ▼  load_documents()
  Raw LangChain Documents
          │
          ▼  chunk_documents()  ← RecursiveCharacterTextSplitter (size=600, overlap=150)
  Overlapping Chunks
          │
    ┌─────┴──────┐
    ▼            ▼
  FAISS        BM25Store
  (semantic)   (keyword)
    │            │
    └─────┬──────┘
          ▼  HybridRetriever.retrieve()
  Merged + Deduplicated Candidates (~16 chunks)
          │
          ▼  rerank()  ← cosine similarity scoring
  Top-3 Most Relevant Chunks
          │
          ▼  generate_answer()  ← structured prompt with source citations
  Grounded Answer ✓
"""

import os
from typing import List, Tuple, Optional

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.schema import Document

import config
from utils import list_documents, print_retrieved_chunks, print_answer
from vectorstores.bm25_store import BM25Store
from retriever.hybrid_retriever import HybridRetriever
from reranker.reranker import rerank


# ── LOAD DOCUMENTS ────────────────────────────────────────────────────────────

def load_documents(data_dir: str = config.DATA_DIR) -> List[Document]:
    """
    Load all .pdf, .txt, .docx files from data_dir into LangChain Documents.
    Tags each document with its source filename in metadata.
    """
    loaders_map = {
        ".pdf":  PyPDFLoader,
        ".txt":  TextLoader,
        ".docx": Docx2txtLoader,
    }

    files = list_documents(data_dir)
    if not files:
        raise ValueError(
            f"No documents found in '{data_dir}'. "
            "Add .pdf, .txt, or .docx files and try again."
        )

    all_docs = []
    for filepath in files:
        ext    = os.path.splitext(filepath)[1].lower()
        loader = loaders_map[ext](filepath)
        docs   = loader.load()
        for doc in docs:
            doc.metadata["source"] = os.path.basename(filepath)
        print(f"  Loaded: {os.path.basename(filepath)}  ({len(docs)} page(s))")
        all_docs.extend(docs)

    print(f"\nTotal: {len(all_docs)} page(s) from {len(files)} file(s)")
    return all_docs


# ── STEP 6: CHUNK DOCUMENTS ───────────────────────────────────────────────────

def chunk_documents(
    documents: List[Document],
    chunk_size: int    = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split documents into overlapping chunks.

    UPGRADE (Step 6):
        Before: CharacterTextSplitter — cuts at fixed character counts,
                can break sentences mid-way.
        After:  RecursiveCharacterTextSplitter — splits at natural boundaries:
                paragraph (\n\n) → line (\n) → word ( ) → character
                chunk_size=600 (was 500), overlap=150 (was 50)

    Larger overlap means more shared context between adjacent chunks,
    so sentences near chunk boundaries are never completely cut off.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    print(f"Chunked into {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ── STEP 7: EMBEDDING MODEL ───────────────────────────────────────────────────

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load the Hugging Face sentence-transformer embedding model.

    UPGRADE NOTE (Step 7):
        Current default: all-MiniLM-L6-v2 (fast, 384-dim, ~80MB, free)
        Better accuracy: all-mpnet-base-v2 (768-dim, slower but stronger)

        To switch, update EMBEDDING_MODEL in config.py or .env — no other
        changes needed. Delete db/ first so the index is rebuilt with the
        new model's vectors.

        OpenAI alternative (requires API key + costs per token):
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model="text-embedding-3-large")
    """
    print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── FAISS INDEX ───────────────────────────────────────────────────────────────

def build_vector_store(
    chunks: List[Document],
    embedding_model: HuggingFaceEmbeddings,
    db_dir: str = config.DB_DIR,
) -> FAISS:
    """Embed all chunks, build FAISS index, and save to db/."""
    print(f"Embedding {len(chunks)} chunks and building FAISS index...")
    vector_store = FAISS.from_documents(chunks, embedding_model)
    os.makedirs(db_dir, exist_ok=True)
    vector_store.save_local(db_dir)
    print(f"FAISS index saved to '{db_dir}/'")
    return vector_store


def load_vector_store(
    embedding_model: HuggingFaceEmbeddings,
    db_dir: str = config.DB_DIR,
) -> FAISS:
    """Load a previously saved FAISS index from db/."""
    if not os.path.exists(db_dir):
        raise FileNotFoundError(
            f"No FAISS index at '{db_dir}'. Run setup_pipeline() first."
        )
    print(f"Loading FAISS index from '{db_dir}/'...")
    return FAISS.load_local(
        db_dir, embedding_model, allow_dangerous_deserialization=True
    )


def get_or_build_vector_store(
    chunks: List[Document],
    embedding_model: HuggingFaceEmbeddings,
    db_dir: str         = config.DB_DIR,
    force_rebuild: bool = False,
) -> FAISS:
    """Load existing FAISS index if present, otherwise build from scratch."""
    index_file = os.path.join(db_dir, "index.faiss")
    if not force_rebuild and os.path.exists(index_file):
        return load_vector_store(embedding_model, db_dir)
    return build_vector_store(chunks, embedding_model, db_dir)


# ── STEP 5: GENERATE ANSWER ───────────────────────────────────────────────────

def generate_answer(query: str, docs: List[Document]) -> str:
    """
    Build a structured, source-cited prompt and call OpenAI GPT.

    UPGRADE (Step 5):
        Before: llm.invoke(query)
                — raw query with no document context at all → hallucinations

        After:  Structured prompt that:
                - Tags each context block with its source file and chunk ID
                - Instructs the LLM to answer ONLY from the context
                - Provides an explicit "I don't know" fallback
                - Prevents the LLM from using its own prior knowledge

    This single change has the biggest impact on answer accuracy.
    """
    context = "\n\n".join([
        f"[Source: {doc.metadata.get('source', 'unknown')} | "
        f"Chunk: {doc.metadata.get('chunk_id', '?')}]\n{doc.page_content}"
        for doc in docs
    ])

    prompt = f"""Answer ONLY using the context below.
If the answer is not found in the context, say "I don't know based on the provided documents."
Do not make up information or use prior knowledge outside the context.

Context:
{context}

Question: {query}

Answer:"""

    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY.startswith("sk-your"):
        return (
            "[No API key configured — showing prompt that would be sent to GPT]\n\n"
            + prompt
        )

    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        openai_api_key=config.OPENAI_API_KEY,
    )

    return llm.invoke(prompt).content.strip()


# ── STEP 1: MAIN PIPELINE (full upgraded flow) ────────────────────────────────

def run_rag_pipeline(
    question: str,
    vector_store: FAISS,
    chunks: List[Document],
    embedding_model: HuggingFaceEmbeddings,
    top_k: int        = config.TOP_K,
    top_k_rerank: int = config.TOP_K_RERANK,
    verbose: bool     = False,
) -> str:
    """
    Run the full upgraded RAG pipeline for a single question.

    Steps:
        1. BM25Store built over all chunks (keyword index)
        2. HybridRetriever fetches top-K from FAISS + top-K from BM25
        3. Candidates deduplicated → reranker scores by cosine similarity
        4. Top-K reranked chunks injected into structured prompt → GPT answer

    Args:
        question:        User's natural language question.
        vector_store:    Loaded FAISS index.
        chunks:          All document chunks (for BM25).
        embedding_model: Loaded embedding model (for reranking).
        top_k:           Chunks fetched from each retriever (FAISS + BM25).
        top_k_rerank:    Final chunks passed to LLM after reranking.
        verbose:         Print intermediate results if True.

    Returns:
        Generated answer string.
    """
    # Step 2: BM25 keyword index over all chunks
    bm25_store = BM25Store(chunks)

    # Step 3: Hybrid retrieval — FAISS (semantic) + BM25 (keyword)
    hybrid_retriever = HybridRetriever(vector_store, bm25_store)
    candidates = hybrid_retriever.retrieve(question, k=top_k)

    if verbose:
        print(f"\nHybrid retrieval → {len(candidates)} candidate(s)")
        print_retrieved_chunks([(doc, 0.0) for doc in candidates])

    # Step 4: Rerank by cosine similarity — keep only the best chunks
    top_docs = rerank(question, candidates, embedding_model, top_k=top_k_rerank)

    if verbose:
        print(f"\nAfter reranking → top {top_k_rerank} chunk(s) sent to LLM:")
        for i, doc in enumerate(top_docs):
            src = doc.metadata.get("source", "unknown")
            print(f"  [{i+1}] {src}: {doc.page_content[:150]}...")

    # Step 5: Generate answer using structured prompt
    return generate_answer(question, top_docs)


# ── SETUP FUNCTION ────────────────────────────────────────────────────────────

def setup_pipeline(
    data_dir: str       = config.DATA_DIR,
    db_dir: str         = config.DB_DIR,
    force_rebuild: bool = False,
):
    """
    One-call setup: load docs → chunk → embed → FAISS index.

    Returns:
        Tuple of (vector_store, chunks, embedding_model)
        Pass all three into run_rag_pipeline() on every query.
    """
    print("\n" + "=" * 60)
    print("RAG Pipeline v2 — Setup")
    print("=" * 60)

    print("\n[1/3] Loading and chunking documents...")
    docs   = load_documents(data_dir)
    chunks = chunk_documents(docs)

    print("\n[2/3] Loading embedding model...")
    embedding_model = get_embedding_model()

    print("\n[3/3] Building / loading FAISS vector store...")
    vector_store = get_or_build_vector_store(
        chunks, embedding_model, db_dir, force_rebuild
    )

    print("\nSetup complete. Pipeline v2 ready.\n")
    return vector_store, chunks, embedding_model


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Pipeline v2")
    parser.add_argument("--query",       type=str, default=None,
                        help="Single question to ask")
    parser.add_argument("--rebuild",     action="store_true",
                        help="Force rebuild the FAISS index")
    parser.add_argument("--verbose",     action="store_true",
                        help="Show retrieved and reranked chunks")
    parser.add_argument("--interactive", action="store_true",
                        help="Start interactive Q&A session")
    parser.add_argument("--test",        action="store_true",
                        help="Run Step 9 test cases")
    args = parser.parse_args()

    vs, chunks, emb = setup_pipeline(force_rebuild=args.rebuild)

    def ask(q: str):
        ans = run_rag_pipeline(q, vs, chunks, emb, verbose=args.verbose)
        print_answer(q, ans, config.PROMPT_STRATEGY)

    if args.test:
        # ── STEP 9: Test Cases ────────────────────────────────────────────────
        # Run three types of query to validate retrieval quality:
        #
        #   Type 1 — Exact keyword
        #       Tests BM25 strength: looks for a specific term mentioned in the doc.
        #       Expected: precise chunk containing that exact term.
        #
        #   Type 2 — Conceptual
        #       Tests FAISS/semantic strength: no exact keyword, needs meaning match.
        #       Expected: relevant chunk even if wording differs.
        #
        #   Type 3 — Mixed keyword + concept
        #       Tests hybrid strength: combines specific term with broader concept.
        #       Expected: better results than either retriever alone.
        #
        # You should see: better recall, more precise answers, fewer hallucinations.
        print("\n" + "=" * 60)
        print("STEP 9 — Test Cases (Exact / Conceptual / Mixed)")
        print("=" * 60)

        print("\n[Test 1 — Exact keyword query]")
        ask("What does the document say about FAISS?")

        print("\n[Test 2 — Conceptual query]")
        ask("Explain the main idea of these documents")

        print("\n[Test 3 — Mixed keyword + concept]")
        ask("What indexing method is used for similarity search in this project?")

    elif args.interactive:
        print("\nInteractive mode — type 'exit' to quit\n")
        while True:
            q = input("You: ").strip()
            if q.lower() in ("exit", "quit", "q"):
                break
            if q:
                ask(q)

    elif args.query:
        ask(args.query)

    else:
        ask("What are the main topics covered in these documents?")
