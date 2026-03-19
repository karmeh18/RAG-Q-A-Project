"""
retriever/hybrid_retriever.py
------------------------------
Hybrid Retrieval — combines FAISS (semantic) + BM25 (keyword).

WHY HYBRID?
    Semantic-only (FAISS):
        ✓ Finds conceptually related content even with different words
        ✗ Misses exact terms, acronyms, model names, version numbers

    Keyword-only (BM25):
        ✓ Pinpoints exact matches — technical terms, proper nouns
        ✗ Misses paraphrases, synonyms, conceptual queries

    Hybrid:
        ✓ Gets the best of both worlds
        ✓ Higher recall — finds more truly relevant chunks
        ✓ Standard in production search systems (Elasticsearch, Weaviate)

HOW IT WORKS:
    1. Run FAISS similarity search   → top-K semantic results
    2. Run BM25 keyword search       → top-K keyword results
    3. Merge both result lists
    4. Deduplicate by page content   → unique set of candidates
    5. Pass to reranker for final scoring

    Deduplication ensures chunks that rank highly in BOTH methods
    (the best results) appear only once — not twice.
"""

from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from vectorstores.bm25_store import BM25Store
from typing import List


class HybridRetriever:
    """
    Combines FAISS semantic search and BM25 keyword search
    into a single merged candidate set.

    Args:
        faiss_store: Built FAISS vector store.
        bm25_store:  Built BM25Store over the same document chunks.
    """

    def __init__(self, faiss_store: FAISS, bm25_store: BM25Store):
        self.faiss = faiss_store
        self.bm25  = bm25_store

    def retrieve(self, query: str, k: int = 8) -> List[Document]:
        """
        Retrieve a merged, deduplicated set of candidate documents.

        Fetches k results from each method → up to 2k candidates →
        deduplication → typically 8–12 unique candidates passed to reranker.

        Args:
            query: User's natural language question.
            k:     Number of results to fetch from each retriever.

        Returns:
            Deduplicated list of candidate LangChain Documents.
        """
        # Semantic retrieval via FAISS (meaning-based)
        semantic_docs = self.faiss.similarity_search(query, k=k)

        # Keyword retrieval via BM25 (term-frequency based)
        keyword_docs  = self.bm25.search(query, k=k)

        # Merge and deduplicate by page content
        # Dict preserves the first occurrence and drops duplicates
        unique_docs = {}
        for doc in semantic_docs + keyword_docs:
            unique_docs[doc.page_content] = doc

        return list(unique_docs.values())
