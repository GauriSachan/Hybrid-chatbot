"""
reranker.py -- adds cross-encoder reranking (the useful part of the teammate's retriever.py)
on top of the team's real Part 2 store (kb_store / hybrid_retriever), instead of running a
second, separate ChromaDB store.

Why not use retriever.py's RAGRetriever directly:
  - It stores everything in one ChromaDB collection with no "kb" vs "user_docs" separation and
    no session_id filtering, so one user's uploaded file could be retrieved for another user.
  - kb_store.py already solves that (two FAISS indexes + session_key filtering) and is what
    ingestion.py and hybrid_retriever.py write into / read from. Switching stores would mean
    re-ingesting everything through a second pipeline.
The cross-encoder reranking step itself is a genuine improvement (pairwise relevance, not just
cosine similarity), so it's kept -- as a wrapper around HybridRetriever, not a replacement store.
"""
from __future__ import annotations

from typing import Callable, Optional

from hybrid_retriever import HybridRetriever
from response_generator import Chunk

CrossEncoderFn = Callable[[list[tuple[str, str]]], list[float]]   # [(query, chunk_text)] -> scores


class RerankingRetriever:
    """Drop-in replacement for HybridRetriever: fetches more candidates, reranks with a
    cross-encoder, returns the top_k. Same signature, same Chunk output, same session isolation
    (delegated to HybridRetriever/kb_store -- reranking never bypasses it)."""

    def __init__(self, base: Optional[HybridRetriever] = None, rerank_fn: Optional[CrossEncoderFn] = None,
                 candidate_multiplier: int = 4):
        self.base = base or HybridRetriever()
        self.rerank_fn = rerank_fn
        self.candidate_multiplier = candidate_multiplier

    def search(self, query: str, top_k: int = 5, sources: Optional[list[str]] = None,
              session_id: Optional[str] = None) -> list[Chunk]:
        if self.rerank_fn is None:
            return self.base.search(query, top_k=top_k, sources=sources, session_id=session_id)

        candidates = self.base.search(query, top_k=top_k * self.candidate_multiplier,
                                      sources=sources, session_id=session_id)
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        scores = self.rerank_fn(pairs)
        # Cross-encoder scores are unbounded logits, not 0..1 -- squash with a logistic so
        # ResponseGenerator's min_score filtering still means something.
        import math
        squashed = [1.0 / (1.0 + math.exp(-s)) for s in scores]
        ranked = sorted(zip(candidates, squashed), key=lambda cs: cs[1], reverse=True)
        out = []
        for c, s in ranked[:top_k]:
            c.score = s
            out.append(c)
        return out


def sentence_transformers_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoderFn:
    """pip install sentence-transformers (CrossEncoder is part of it, same package retriever.py used)."""
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_name)
    return lambda pairs: [float(s) for s in model.predict(pairs)]
