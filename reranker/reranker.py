"""
reranker/reranker.py
--------------------
Reranking — re-scores candidate documents against the query and
returns only the top-K most relevant ones.

WHY RERANK?
    The hybrid retriever returns up to ~16 candidate chunks.
    Not all of them are equally good — some are marginally related.
    Sending all of them to the LLM:
        ✗ Wastes tokens (costs more money)
        ✗ Adds noise that confuses the LLM
        ✗ May dilute the actual answer

    Reranking precisely scores every candidate and keeps only the
    top-K — so the LLM receives a tight, high-quality context.

HOW THIS RERANKER WORKS (Cosine Similarity):
    1. Embed the query                 → query_vector  (384 dims)
    2. Embed each candidate chunk      → doc_vector    (384 dims)
    3. Compute cosine similarity:
           score = dot_product(query_vec, doc_vec)
           (vectors are already normalised → dot product = cosine similarity)
    4. Sort by score descending
    5. Return top_k documents

    Cosine similarity of 1.0 = identical direction (perfectly relevant)
    Cosine similarity of 0.0 = perpendicular (unrelated)
    Cosine similarity of -1.0 = opposite (contradictory)

PRODUCTION ALTERNATIVE:
    For even better reranking, use a Cross-Encoder model:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        scores = model.predict([(query, doc.page_content) for doc in docs])
    Cross-encoders jointly encode query + document together,
    giving more accurate scores but at higher latency cost.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List


def rerank(
    query: str,
    docs: List[Document],
    embeddings: HuggingFaceEmbeddings,
    top_k: int = 3,
) -> List[Document]:
    """
    Re-score candidate documents by cosine similarity to the query
    and return only the top_k most relevant.

    Args:
        query:      User's natural language question.
        docs:       Candidate documents from HybridRetriever.
        embeddings: The same embedding model used to build the FAISS index.
        top_k:      Number of top documents to return to the LLM.

    Returns:
        Top-K documents ranked by cosine similarity to the query.
    """
    if not docs:
        return []

    # Embed the query into a vector
    query_vec = embeddings.embed_query(query)

    scored = []
    for doc in docs:
        # Embed the document chunk
        doc_vec = embeddings.embed_query(doc.page_content)

        # Cosine similarity = dot product of normalised vectors
        score = sum(q * d for q, d in zip(query_vec, doc_vec))
        scored.append((score, doc))

    # Sort by score descending and return top_k
    scored.sort(reverse=True, key=lambda x: x[0])

    return [doc for _, doc in scored[:top_k]]
