"""
vectorstores/bm25_store.py
--------------------------
BM25 (Best Match 25) — keyword-based retrieval.

WHY BM25?
    FAISS retrieves by semantic MEANING (embedding similarity).
    BM25 retrieves by exact KEYWORD MATCH (term frequency + inverse document frequency).

    They complement each other:
    - FAISS finds "automobile" when you search "car" (same meaning, different word)
    - BM25 finds exact technical terms, model names, version numbers, acronyms

    Together as Hybrid Retrieval → better recall than either alone.

HOW BM25 WORKS:
    For each document chunk, it scores how relevant it is to the query based on:
    1. TF  (Term Frequency)       — how often the query word appears in the chunk
    2. IDF (Inverse Doc Frequency)— how rare the word is across ALL chunks
    3. Document length normalisation — shorter docs aren't penalised for having fewer words

    BM25Okapi is the standard variant used in search engines like Elasticsearch.
"""

from rank_bm25 import BM25Okapi
from langchain.schema import Document
from typing import List


class BM25Store:
    """
    Keyword-based document retrieval using BM25Okapi algorithm.

    Args:
        documents: List of LangChain Document chunks to index.
    """

    def __init__(self, documents: List[Document]):
        self.docs = documents

        # Tokenise each document by splitting on whitespace
        # In production you'd use a proper tokeniser (NLTK, spaCy)
        # but whitespace split works well for most English text
        self.tokenized_docs = [
            doc.page_content.lower().split()
            for doc in documents
        ]

        # Build the BM25 index over all tokenised chunks
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve the top-K most keyword-relevant documents for a query.

        Args:
            query: User's search query string.
            k:     Number of top results to return.

        Returns:
            List of top-K LangChain Documents ranked by BM25 score.
        """
        tokenized_query = query.lower().split()

        # Get BM25 relevance score for every document in the index
        scores = self.bm25.get_scores(tokenized_query)

        # Pair each doc with its score, sort descending, return top-K
        ranked = sorted(
            zip(self.docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, _ in ranked[:k]]
