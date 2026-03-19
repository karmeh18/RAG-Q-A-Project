"""
rag_pipeline.py
---------------
🔥 CORE LOGIC — everything connected in one file.

This is the heart of the project. It wires together:
  1. Document loading  — read PDFs / TXTs / DOCXs from data/
  2. Chunking          — split documents into overlapping segments
  3. Embedding         — convert chunks into vectors (Hugging Face)
  4. Vector store      — save/load vectors in FAISS (data/db/)
  5. Retrieval         — find the most relevant chunks for a query
  6. Prompt building   — inject retrieved context into the LLM prompt
  7. Generation        — call OpenAI GPT and return a grounded answer

HOW RAG WORKS (in plain English):
──────────────────────────────────
  Normal LLM:  User asks question → LLM answers from training memory
               Problem: LLM hallucinates on private/new documents

  RAG:         User asks question
               → embed question into a vector
               → find the most similar document chunks (FAISS search)
               → inject those chunks as context into the prompt
               → LLM answers using the real document content
               Result: grounded, accurate, cite-able answers

FLOW DIAGRAM:
─────────────
  Your Documents (PDF/TXT/DOCX)
         │
         ▼ load_documents()
  Raw LangChain Documents
         │
         ▼ chunk_documents()
  Overlapping Chunks  ──────────────────► FAISS Index (saved to db/)
         │                                      │
         ▼ embed (HuggingFace)                  ▼ similarity_search()
  384-dim Vectors                    Top-K Relevant Chunks
                                               │
                                               ▼ format_context()
                                      Context String
                                               │
                                               ▼ build_prompt()
                                      Prompt = Context + Question
                                               │
                                               ▼ OpenAI GPT
                                      Grounded Answer ✓
"""

import os
from typing import List, Tuple, Optional

# LangChain — document loading
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

# LangChain — text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain — embeddings and vector store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LangChain — LLM
from langchain_openai import ChatOpenAI
from langchain.schema import Document

# Local modules
import config
from utils import (
    list_documents,
    format_context,
    print_retrieved_chunks,
    get_prompt_template,
    print_answer,
)


# ── STEP 1: LOAD DOCUMENTS ────────────────────────────────────────────────────

def load_documents(data_dir: str = config.DATA_DIR) -> List[Document]:
    """
    Load all supported documents (.pdf, .txt, .docx) from data_dir.

    Each file is loaded by the appropriate LangChain loader.
    The source filename is stored in each document's metadata so we
    can trace which document each chunk came from.

    Args:
        data_dir: Path to folder containing your documents.

    Returns:
        List of raw LangChain Document objects.
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
        ext = os.path.splitext(filepath)[1].lower()
        loader = loaders_map[ext](filepath)
        docs   = loader.load()

        # Tag source filename into metadata for traceability
        for doc in docs:
            doc.metadata["source"] = os.path.basename(filepath)

        print(f"  Loaded: {os.path.basename(filepath)}  ({len(docs)} page(s))")
        all_docs.extend(docs)

    print(f"\nTotal: {len(all_docs)} document page(s) loaded from {len(files)} file(s)")
    return all_docs


# ── STEP 2: CHUNK DOCUMENTS ───────────────────────────────────────────────────

def chunk_documents(
    documents: List[Document],
    chunk_size: int    = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split documents into overlapping chunks for embedding.

    WHY CHUNK?
        Embedding models have token limits (~512 tokens).
        Long documents must be split. Overlap ensures sentences
        at chunk boundaries are not cut off and lost.

    WHY RecursiveCharacterTextSplitter?
        It tries to split at natural text boundaries first:
        paragraph breaks (\n\n) → line breaks (\n) → spaces → characters
        This preserves meaning better than splitting at fixed character counts.

    Args:
        documents:     Raw LangChain Documents.
        chunk_size:    Max characters per chunk (default from config).
        chunk_overlap: Overlap between adjacent chunks (default from config).

    Returns:
        List of chunked Documents, each with chunk_id in metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Tag each chunk with a sequential ID for tracing
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    print(f"Chunked into {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ── STEP 3 & 4: EMBED + BUILD FAISS INDEX ────────────────────────────────────

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load the Hugging Face sentence-transformer embedding model.

    The model converts text → dense vector (384 dimensions for MiniLM).
    Downloads ~80MB on first run, cached locally after that.
    Runs entirely on your machine — no API calls, no cost.

    Returns:
        HuggingFaceEmbeddings instance.
    """
    print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(
    chunks: List[Document],
    embedding_model: HuggingFaceEmbeddings,
    db_dir: str = config.DB_DIR,
) -> FAISS:
    """
    Embed all chunks and build a FAISS vector store. Save to db/.

    WHAT HAPPENS INTERNALLY:
        For each chunk:
            text → embedding model → 384-dim float vector
        All vectors are stacked into a matrix.
        FAISS builds an IndexFlatL2 index over that matrix.
        (IndexFlatL2 = exact Euclidean distance search)

    The index is saved to db/ so we don't re-embed on every run.
    Two files are written:
        db/index.faiss  — the raw vector index
        db/index.pkl    — chunk metadata (source, chunk_id, text)

    Args:
        chunks:          Chunked LangChain Documents.
        embedding_model: Loaded HuggingFaceEmbeddings.
        db_dir:          Directory to save the FAISS index.

    Returns:
        FAISS vector store (in-memory + persisted to disk).
    """
    print(f"Embedding {len(chunks)} chunks and building FAISS index...")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )

    os.makedirs(db_dir, exist_ok=True)
    vector_store.save_local(db_dir)
    print(f"Vector store saved to '{db_dir}/'")

    return vector_store


def load_vector_store(
    embedding_model: HuggingFaceEmbeddings,
    db_dir: str = config.DB_DIR,
) -> FAISS:
    """
    Load an existing FAISS index from db/.

    The embedding model passed here MUST be the same model that was
    used when the index was built — otherwise vectors are incompatible.

    Args:
        embedding_model: Same model used to build the index.
        db_dir:          Directory where index.faiss and index.pkl live.

    Returns:
        FAISS vector store ready for similarity search.
    """
    if not os.path.exists(db_dir):
        raise FileNotFoundError(
            f"No vector store found at '{db_dir}'. "
            "Run build_vector_store() first."
        )
    print(f"Loading existing vector store from '{db_dir}/'...")
    return FAISS.load_local(
        db_dir,
        embedding_model,
        allow_dangerous_deserialization=True,
    )


def get_or_build_vector_store(
    chunks: List[Document],
    embedding_model: HuggingFaceEmbeddings,
    db_dir: str    = config.DB_DIR,
    force_rebuild: bool = False,
) -> FAISS:
    """
    Smart loader: load existing index if available, else build it.

    On first run  → embeds all chunks, builds index, saves to db/
    On later runs → loads from db/ instantly (skips re-embedding)
    force_rebuild → always re-embeds and overwrites the saved index

    Args:
        chunks:        Chunks to embed (only used when building).
        embedding_model: Embedding model.
        db_dir:        Index save/load directory.
        force_rebuild: Force rebuild even if index exists.

    Returns:
        FAISS vector store.
    """
    index_file = os.path.join(db_dir, "index.faiss")
    if not force_rebuild and os.path.exists(index_file):
        return load_vector_store(embedding_model, db_dir)
    return build_vector_store(chunks, embedding_model, db_dir)


# ── STEP 5: RETRIEVE ─────────────────────────────────────────────────────────

def retrieve(
    query: str,
    vector_store: FAISS,
    top_k: int = config.TOP_K,
) -> List[Tuple[Document, float]]:
    """
    Find the top-K document chunks most semantically similar to the query.

    HOW IT WORKS:
        1. Embed the query string → 384-dim vector (same model as chunks)
        2. FAISS computes L2 distance between query vector and ALL chunk vectors
        3. Return the K chunks with smallest distance (= most similar meaning)

    L2 distance (lower = more similar):
        0.0  = identical meaning
        0.5  = closely related
        1.0+ = different topics

    Args:
        query:        User's natural language question.
        vector_store: Loaded FAISS index.
        top_k:        Number of chunks to return.

    Returns:
        List of (Document, score) tuples sorted by relevance.
    """
    return vector_store.similarity_search_with_score(query, k=top_k)


# ── STEP 6 & 7: BUILD PROMPT + GENERATE ANSWER ───────────────────────────────

def build_prompt(
    question: str,
    context: str,
    strategy: str = config.PROMPT_STRATEGY,
) -> str:
    """
    Inject the retrieved context and user question into the prompt template.

    Three strategies (set in config.py):
        zero_shot       — direct Q&A, fastest
        few_shot        — guided by examples, more consistent format
        chain_of_thought — step-by-step reasoning, best for complex queries

    Args:
        question: User's question string.
        context:  Formatted string of retrieved document chunks.
        strategy: Prompt strategy name.

    Returns:
        Complete prompt string ready to send to the LLM.
    """
    template = get_prompt_template(strategy)
    return template.format(context=context, question=question)


def generate_answer(prompt: str) -> str:
    """
    Send the prompt to OpenAI GPT and return the answer.

    Uses ChatOpenAI (chat completion API):
        - temperature=0 → deterministic, factual output
        - model set in config.py (default gpt-3.5-turbo)

    If no valid API key is set, returns the prompt itself so you can
    see what would be sent to the LLM.

    Args:
        prompt: Complete prompt string (context + question).

    Returns:
        LLM-generated answer string.
    """
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY.startswith("sk-your"):
        return (
            "[No API key configured]\n"
            "Add your OPENAI_API_KEY to the .env file.\n\n"
            "The following prompt would be sent to GPT:\n\n"
            + prompt
        )

    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        openai_api_key=config.OPENAI_API_KEY,
    )

    response = llm.invoke(prompt)
    return response.content.strip()


# ── MAIN PIPELINE FUNCTION ────────────────────────────────────────────────────

def run_rag_pipeline(
    question: str,
    vector_store: FAISS,
    strategy: str   = config.PROMPT_STRATEGY,
    top_k: int      = config.TOP_K,
    verbose: bool   = False,
) -> str:
    """
    Run the full RAG pipeline for a single question.

    This is the one function that ties everything together:
        retrieve → format context → build prompt → generate answer

    Args:
        question:     User's natural language question.
        vector_store: Loaded and populated FAISS index.
        strategy:     Prompt strategy ("zero_shot" / "few_shot" / "chain_of_thought").
        top_k:        Number of chunks to retrieve.
        verbose:      If True, print retrieved chunks before the answer.

    Returns:
        Generated answer string.
    """
    # Step 5: Retrieve relevant chunks
    retrieved = retrieve(question, vector_store, top_k)

    if verbose:
        print_retrieved_chunks(retrieved)

    # Step 6: Format context and build prompt
    context = format_context(retrieved, include_metadata=True)
    prompt  = build_prompt(question, context, strategy)

    # Step 7: Generate answer
    answer = generate_answer(prompt)

    return answer


# ── SETUP FUNCTION ────────────────────────────────────────────────────────────

def setup_pipeline(
    data_dir: str      = config.DATA_DIR,
    db_dir: str        = config.DB_DIR,
    force_rebuild: bool = False,
) -> FAISS:
    """
    One-call setup: load documents → chunk → embed → index.

    Call this once at the start of your session.
    Returns a ready-to-query FAISS vector store.

    Args:
        data_dir:      Folder with your documents.
        db_dir:        Where to save/load the FAISS index.
        force_rebuild: Rebuild the index even if db/ already exists.

    Returns:
        FAISS vector store ready for run_rag_pipeline().
    """
    print("\n" + "=" * 60)
    print("RAG Pipeline — Setup")
    print("=" * 60)

    # Load and chunk
    print("\n[1/3] Loading documents...")
    docs   = load_documents(data_dir)
    chunks = chunk_documents(docs)

    # Embed and index
    print("\n[2/3] Loading embedding model...")
    embeddings = get_embedding_model()

    print("\n[3/3] Building / loading vector store...")
    vector_store = get_or_build_vector_store(
        chunks, embeddings, db_dir, force_rebuild
    )

    print("\nSetup complete. Ready to answer questions.\n")
    return vector_store


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick demo — runs when you execute:  python rag_pipeline.py

    Add your documents to data/ first, then run this to test the pipeline.
    """
    import argparse

    parser = argparse.ArgumentParser(description="RAG Q&A Pipeline")
    parser.add_argument("--query",    type=str, default=None,
                        help="Question to ask")
    parser.add_argument("--strategy", type=str, default=config.PROMPT_STRATEGY,
                        choices=["zero_shot", "few_shot", "chain_of_thought"],
                        help="Prompt strategy")
    parser.add_argument("--rebuild",  action="store_true",
                        help="Force rebuild the vector store")
    parser.add_argument("--verbose",  action="store_true",
                        help="Show retrieved chunks")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive Q&A session")
    args = parser.parse_args()

    # Setup
    vs = setup_pipeline(force_rebuild=args.rebuild)

    if args.interactive:
        print("Interactive mode — type 'exit' to quit\n")
        while True:
            q = input("You: ").strip()
            if q.lower() in ("exit", "quit", "q"):
                break
            if q:
                ans = run_rag_pipeline(q, vs, args.strategy, verbose=args.verbose)
                print_answer(q, ans, args.strategy)

    elif args.query:
        ans = run_rag_pipeline(args.query, vs, args.strategy, verbose=args.verbose)
        print_answer(args.query, ans, args.strategy)

    else:
        # Default demo questions
        demo_questions = [
            "What is FAISS and what is it used for?",
            "How do airlines use AI for predictive maintenance?",
            "What are the stages of an ML deployment pipeline?",
        ]
        for q in demo_questions:
            ans = run_rag_pipeline(q, vs, args.strategy, verbose=args.verbose)
            print_answer(q, ans, args.strategy)
