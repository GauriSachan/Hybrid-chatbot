"""
hybrid_retriever.py  --  thin reference implementation of the Retriever protocol in pipeline.py

This is Part 2's job; it is included so Part 1 can be tested end to end and so Part 2 has a
working example. Part 2 can replace/extend it (reranking, weights) as long as search() keeps
this signature and returns Chunk objects with a 0..1 score.

IMPORTANT: Chunk.score must be a 0..1 similarity because ResponseGenerator drops chunks below
min_score=0.30. RRF scores are ~0.01-0.03, so they are used ONLY for ranking; the score that is
returned is the cosine similarity between the query and the chunk.
"""
from __future__ import annotations

from typing import Optional

from kb_store import KnowledgeStore, embed_texts, get_store
from response_generator import Chunk


class HybridRetriever:
    def __init__(self, store: Optional[KnowledgeStore] = None, rrf_k: int = 60, candidates: int = 20):
        self.store = store or get_store()
        self.rrf_k = rrf_k
        self.candidates = candidates

    def search(self, query: str, top_k: int = 5, sources: Optional[list[str]] = None,
               session_id: Optional[str] = None) -> list[Chunk]:
        origin = (sources or ["kb"])[0]
        if origin == "user_docs" and not session_id:
            return []                                   # never search uploads without a session
        qvec = embed_texts([query])[0]
        dense = self.store.dense_search(origin, qvec, self.candidates, session_id)
        sparse = self.store.bm25_search(origin, query, self.candidates, session_id)

        fused: dict[int, float] = {}                    # Reciprocal Rank Fusion
        for ranking in (dense, sparse):
            for rank, (cid, _) in enumerate(ranking, start=1):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        top_ids = sorted(fused, key=fused.get, reverse=True)[:top_k]
        if not top_ids:
            return []

        cos = self.store.cosine_scores(origin, top_ids, qvec)
        rows = self.store.get_chunks(top_ids)
        return [Chunk(text=rows[i]["text"], source=rows[i]["source"], score=max(0.0, min(1.0, cos[i])),
                      url=rows[i]["url"] or None, page=rows[i]["page"], origin=origin)
                for i in top_ids if i in rows]
