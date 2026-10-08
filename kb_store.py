"""
kb_store.py  --  Part 1: storage layer (SQLite + FAISS + BM25)

One chunk collection, indexed twice (as in the Phase-I report):
    SQLite  -> chunk text + metadata (source of truth)
    FAISS   -> dense vectors (MiniLM, cosine via inner product on normalised vectors)
    BM25    -> sparse index, built lazily in memory from SQLite

Two logical collections, kept in separate FAISS indexes:
    "kb"        persistent knowledge base
    "user_docs" per-session uploads (always filtered by session_id)

Other parts should only use:
    get_store(), embed_texts(), embed_fn         (embedding shared by Part 2 and Part 3)
    store.dense_search / bm25_search / get_chunks / cosine_scores   (Part 2)
    ingestion.py                                 (writes into the store)
"""
from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Callable, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

DATA_DIR = Path(os.getenv("KB_DATA_DIR", "./kb_data"))
EMBED_MODEL = os.getenv("KB_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
ORIGINS = ("kb", "user_docs")

# --------------------------------------------------------------------------- #
# Embeddings (single source of truth -- Part 2 and Part 3 must import these)
# --------------------------------------------------------------------------- #
_embedder: Optional[Callable[[list[str]], np.ndarray]] = None
_dim = 384  # all-MiniLM-L6-v2


def set_embedder(fn: Callable[[list[str]], np.ndarray], dim: int) -> None:
    """Swap the embedding model (or inject a fake one for tests)."""
    global _embedder, _dim
    _embedder, _dim = fn, dim


def embed_texts(texts: list[str]) -> np.ndarray:
    """-> float32 array (n, dim), L2-normalised."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL)
        _embedder = lambda t: model.encode(t, normalize_embeddings=True, batch_size=64,
                                           convert_to_numpy=True)
    return np.ascontiguousarray(_embedder(texts), dtype=np.float32)


def embed_fn(texts: list[str]) -> list[list[float]]:
    """Signature expected by QueryRouter(embed_fn=...)."""
    return embed_texts(texts).tolist()


# --------------------------------------------------------------------------- #
# BM25 tokenisation
# --------------------------------------------------------------------------- #
_STOP = set("a an and are as at be by for from has have how in is it of on or that the this to was "
            "were what when where which who why will with do does did i you my me".split())


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", text.lower()) if w not in _STOP]


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
class KnowledgeStore:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.dir / "kb.sqlite", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._init_db()
        self.index = {o: self._load_index(o) for o in ORIGINS}
        self._bm25: dict = {}

    # ------------------------------------------------------------ setup ----
    def _init_db(self):
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS documents(
            doc_id TEXT PRIMARY KEY, origin TEXT NOT NULL, session_key TEXT NOT NULL DEFAULT '',
            source TEXT, url TEXT, sha256 TEXT, n_chunks INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS chunks(
            id INTEGER PRIMARY KEY AUTOINCREMENT,        -- also the FAISS id
            chunk_id TEXT, doc_id TEXT NOT NULL, origin TEXT NOT NULL,
            session_key TEXT NOT NULL DEFAULT '',        -- '' for kb, session id for user_docs
            source TEXT, url TEXT, page INTEGER, ord INTEGER, text TEXT NOT NULL, hash TEXT NOT NULL);
        CREATE UNIQUE INDEX IF NOT EXISTS ux_chunk ON chunks(origin, session_key, hash);
        CREATE INDEX IF NOT EXISTS ix_doc ON chunks(doc_id);
        CREATE INDEX IF NOT EXISTS ix_sess ON chunks(origin, session_key);
        """)
        self.db.commit()

    def _index_path(self, origin: str) -> Path:
        return self.dir / f"{origin}.faiss"

    def _load_index(self, origin: str):
        p = self._index_path(origin)
        if p.exists():
            return faiss.read_index(str(p))
        return faiss.IndexIDMap2(faiss.IndexFlatIP(_dim))

    def _save_index(self, origin: str):
        faiss.write_index(self.index[origin], str(self._index_path(origin)))

    # ----------------------------------------------------------- writes ----
    def has_document(self, sha256: str, origin: str, session_id: Optional[str]) -> Optional[str]:
        r = self.db.execute("SELECT doc_id FROM documents WHERE sha256=? AND origin=? AND session_key=?",
                            (sha256, origin, session_id or "")).fetchone()
        return r["doc_id"] if r else None

    def add_document(self, doc_id: str, origin: str, source: str, chunks: list[dict],
                     url: Optional[str] = None, session_id: Optional[str] = None,
                     sha256: str = "") -> int:
        """chunks: [{"text": str, "page": int|None}, ...]   Returns number of NEW chunks stored."""
        assert origin in ORIGINS
        if origin == "user_docs" and not session_id:
            raise ValueError("user_docs require a session_id")
        skey = session_id or ""
        with self.lock:
            hashes = [hashlib.sha1(re.sub(r"\s+", " ", c["text"]).strip().lower().encode()).hexdigest()
                      for c in chunks]
            existing = {r["hash"] for r in self.db.execute(
                "SELECT hash FROM chunks WHERE origin=? AND session_key=?", (origin, skey))}
            seen, keep = set(), []
            for i, h in enumerate(hashes):               # drop duplicates (in store and in this doc)
                if h in existing or h in seen:
                    continue
                seen.add(h)
                keep.append(i)

            self.db.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
            self.db.execute("INSERT INTO documents(doc_id,origin,session_key,source,url,sha256,n_chunks) "
                            "VALUES(?,?,?,?,?,?,?)", (doc_id, origin, skey, source, url, sha256, len(keep)))
            if not keep:
                self.db.commit()
                return 0

            vecs = embed_texts([chunks[i]["text"] for i in keep])
            ids = []
            for n, i in enumerate(keep):
                cur = self.db.execute(
                    "INSERT INTO chunks(chunk_id,doc_id,origin,session_key,source,url,page,ord,text,hash) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (f"{doc_id}-{n}", doc_id, origin, skey, source, url, chunks[i].get("page"), n,
                     chunks[i]["text"], hashes[i]))
                ids.append(cur.lastrowid)
            self.index[origin].add_with_ids(vecs, np.asarray(ids, dtype=np.int64))
            self.db.commit()
            self._save_index(origin)
            self._bm25.clear()
            return len(ids)

    def delete_document(self, doc_id: str) -> int:
        with self.lock:
            rows = self.db.execute("SELECT id, origin FROM chunks WHERE doc_id=?", (doc_id,)).fetchall()
            self._remove(rows)
            self.db.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
            self.db.commit()
            return len(rows)

    def delete_session(self, session_id: str) -> int:
        """Remove everything a session uploaded (call when the chat session ends)."""
        with self.lock:
            rows = self.db.execute("SELECT id, origin FROM chunks WHERE origin='user_docs' AND session_key=?",
                                   (session_id,)).fetchall()
            self._remove(rows)
            self.db.execute("DELETE FROM documents WHERE origin='user_docs' AND session_key=?", (session_id,))
            self.db.commit()
            return len(rows)

    def _remove(self, rows):
        by_origin: dict[str, list[int]] = {}
        for r in rows:
            by_origin.setdefault(r["origin"], []).append(r["id"])
        for origin, ids in by_origin.items():
            self.index[origin].remove_ids(np.asarray(ids, dtype=np.int64))
            self.db.executemany("DELETE FROM chunks WHERE id=?", [(i,) for i in ids])
            self._save_index(origin)
        self._bm25.clear()

    # ------------------------------------------------------------ reads ----
    def get_chunks(self, ids: list[int]) -> dict[int, sqlite3.Row]:
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        return {r["id"]: r for r in self.db.execute(f"SELECT * FROM chunks WHERE id IN ({q})", ids)}

    def list_documents(self, origin: str, session_id: Optional[str] = None) -> list[dict]:
        rows = self.db.execute("SELECT doc_id,source,url,n_chunks FROM documents WHERE origin=? AND session_key=?",
                               (origin, session_id or "")).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        return {o: {"chunks": self.index[o].ntotal,
                    "documents": self.db.execute("SELECT COUNT(*) FROM documents WHERE origin=?", (o,)).fetchone()[0]}
                for o in ORIGINS}

    def _session_ids(self, origin: str, session_id: Optional[str]) -> set[int]:
        return {r["id"] for r in self.db.execute(
            "SELECT id FROM chunks WHERE origin=? AND session_key=?", (origin, session_id or ""))}

    # ----------------------------------------------------------- search ----
    def dense_search(self, origin: str, qvec: np.ndarray, k: int,
                     session_id: Optional[str] = None) -> list[tuple[int, float]]:
        """-> [(chunk_id, cosine)] best first."""
        idx = self.index[origin]
        n = idx.ntotal
        if n == 0 or (origin == "user_docs" and not session_id):
            return []
        allowed = self._session_ids(origin, session_id) if origin == "user_docs" else None
        try_k = min(n, k) if allowed is None else min(n, max(k * 10, 50))
        while True:
            D, I = idx.search(qvec.reshape(1, -1).astype(np.float32), try_k)
            hits = [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i != -1]
            if allowed is not None:
                hits = [h for h in hits if h[0] in allowed]
            if len(hits) >= k or try_k >= n:
                return hits[:k]
            try_k = min(n, try_k * 4)

    def _get_bm25(self, origin: str, session_id: Optional[str]):
        key = (origin, session_id or "")
        if key not in self._bm25:
            rows = self.db.execute("SELECT id, text FROM chunks WHERE origin=? AND session_key=? ORDER BY id",
                                   key).fetchall()
            if not rows:
                self._bm25[key] = (None, [])
            else:
                self._bm25[key] = (BM25Okapi([tokenize(r["text"]) or ["_"] for r in rows]),
                                   [r["id"] for r in rows])
        return self._bm25[key]

    def bm25_search(self, origin: str, query: str, k: int,
                    session_id: Optional[str] = None) -> list[tuple[int, float]]:
        """-> [(chunk_id, bm25_score)] best first (only positive scores)."""
        if origin == "user_docs" and not session_id:
            return []
        toks = tokenize(query)
        bm25, ids = self._get_bm25(origin, session_id)
        if bm25 is None or not toks:
            return []
        scores = bm25.get_scores(toks)
        order = np.argsort(-scores)[:k]
        return [(ids[i], float(scores[i])) for i in order if scores[i] > 0]

    def cosine_scores(self, origin: str, ids: list[int], qvec: np.ndarray) -> dict[int, float]:
        """Exact cosine between the query and given chunks (used to give BM25-only hits a 0..1 score)."""
        out = {}
        for i in ids:
            out[i] = float(np.dot(self.index[origin].reconstruct(int(i)), qvec))
        return out


_store: Optional[KnowledgeStore] = None


def get_store() -> KnowledgeStore:
    global _store
    if _store is None:
        _store = KnowledgeStore()
    return _store
