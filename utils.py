"""
utils.py
--------
Helper functions used across the RAG pipeline.
Keeps rag_pipeline.py clean by moving repetitive or utility logic here.
"""

import os
from typing import List, Tuple
from langchain.schema import Document


# ── Document helpers ──────────────────────────────────────────────────────────

def list_documents(data_dir: str) -> List[str]:
    """
    List all supported document files in the data directory.

    Supported: .pdf, .txt, .docx

    Args:
        data_dir: Path to the folder containing documents.

    Returns:
        List of full file paths.
    """
    supported = {".pdf", ".txt", ".docx"}
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory '{data_dir}' not found. "
            "Create it and add your documents."
        )
    files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if os.path.splitext(f)[1].lower() in supported
    ]
    return sorted(files)


def print_documents(data_dir: str) -> None:
    """Print all documents found in the data directory."""
    files = list_documents(data_dir)
    if not files:
        print(f"No documents found in '{data_dir}'.")
        print("Add .pdf, .txt, or .docx files to get started.")
        return
    print(f"\nDocuments in '{data_dir}':")
    for f in files:
        size_kb = os.path.getsize(f) // 1024
        print(f"  {os.path.basename(f)}  ({size_kb} KB)")


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def format_context(
    retrieved: List[Tuple[Document, float]],
    include_metadata: bool = True,
) -> str:
    """
    Format retrieved (Document, score) pairs into a single context string
    ready to be injected into the LLM prompt.

    Args:
        retrieved:        List of (Document, similarity_score) tuples.
        include_metadata: Whether to prepend source info to each chunk.

    Returns:
        Multi-chunk context string.
    """
    parts = []
    for i, (doc, score) in enumerate(retrieved):
        source   = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_id", i)
        if include_metadata:
            header = f"[Source: {source} | Chunk: {chunk_id} | Score: {score:.4f}]"
            parts.append(f"{header}\n{doc.page_content}")
        else:
            parts.append(doc.page_content)
    return "\n\n---\n\n".join(parts)


def print_retrieved_chunks(retrieved: List[Tuple[Document, float]]) -> None:
    """Pretty-print retrieved chunks for debugging."""
    print(f"\nRetrieved {len(retrieved)} chunk(s):")
    for i, (doc, score) in enumerate(retrieved):
        source = doc.metadata.get("source", "unknown")
        print(f"\n  [{i+1}] Source: {source}  |  Score: {score:.4f}")
        print(f"       {doc.page_content[:200]}...")


# ── Prompt templates ──────────────────────────────────────────────────────────

PROMPTS = {

    # Direct Q&A — fast, works well for clear factual queries
    "zero_shot": (
        "You are a helpful assistant. Answer the question using ONLY the "
        "provided context. If the answer is not in the context, say: "
        "'I don't have enough information to answer this.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),

    # Provide example Q&A pairs to guide the output format
    "few_shot": (
        "You are a precise assistant. Answer using only the context provided.\n\n"
        "--- EXAMPLE ---\n"
        "Context: 'FAISS supports exact and approximate nearest-neighbour search "
        "over dense float vectors.'\n"
        "Question: What does FAISS do?\n"
        "Answer: FAISS performs nearest-neighbour search over dense float vectors, "
        "supporting both exact and approximate modes.\n"
        "--- END EXAMPLE ---\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),

    # Ask the model to reason step-by-step before answering
    # Best for complex or multi-part questions
    "chain_of_thought": (
        "You are an expert analyst. Answer the question using only the context.\n\n"
        "Follow these steps:\n"
        "1. Identify relevant information from the context.\n"
        "2. Reason through the answer step by step.\n"
        "3. Give a concise final answer.\n"
        "4. Note which part of the context you used.\n\n"
        "If the answer is not in the context, say so clearly.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Step-by-step reasoning:"
    ),
}


def get_prompt_template(strategy: str) -> str:
    """
    Return the prompt template string for a given strategy.

    Args:
        strategy: One of "zero_shot", "few_shot", "chain_of_thought".

    Returns:
        Prompt template string with {context} and {question} placeholders.
    """
    if strategy not in PROMPTS:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from: {list(PROMPTS.keys())}"
        )
    return PROMPTS[strategy]


# ── Display helpers ───────────────────────────────────────────────────────────

def print_answer(question: str, answer: str, strategy: str) -> None:
    """Print a formatted Q&A result."""
    print("\n" + "=" * 60)
    print(f"Q: {question}")
    print(f"Strategy: {strategy}")
    print("-" * 60)
    print(f"A: {answer}")
    print("=" * 60)
