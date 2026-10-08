import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import tiktoken

class RAGRetriever:
    """
    A unified, streamlined retriever that handles:
    1. Query embedding & Similarity Search (via ChromaDB)
    2. Metadata Filtering
    3. Cross-Encoder Reranking
    4. Token-limited Context Selection
    """
    def __init__(self, db_path="./chroma_db", collection_name="knowledge_base"):
        # 1. Initialize Vector Database
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # 2. Initialize Models
        # Used for initial dense retrieval
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        # Used for highly accurate reranking of the top N results
        self.reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        # Used for token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def retrieve(self, query: str, top_k: int = 5, filters: dict = None, max_tokens: int = 2000):
        # Step 1: Query Embedding
        query_emb = self.embedding_model.encode(query, normalize_embeddings=True).tolist()
        
        # Step 2: Similarity Search & Filtering
        where_clause = self._build_where_clause(filters)
        
        # Fetch 4x candidates for the reranker to evaluate
        candidate_k = top_k * 4
        n_results = min(candidate_k, self.collection.count())
        
        if n_results == 0:
            return []
            
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=n_results,
            where=where_clause
        )
        
        # Format initial results
        candidates = []
        for i, cid in enumerate(results["ids"][0]):
            candidates.append({
                "chunk_id": cid,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] or {},
                "score": 1.0 - float(results["distances"][0][i]) # Convert distance to similarity
            })
            
        # Step 3: Reranking (Cross-Encoder)
        if candidates:
            pairs = [[query, c["text"]] for c in candidates]
            rerank_scores = self.reranker_model.predict(pairs)
            for c, score in zip(candidates, rerank_scores):
                c["score"] = float(score)
            
            # Sort by the new reranked score
            candidates.sort(key=lambda x: x["score"], reverse=True)
            
        # Step 4: Relevant-context selection (Token limit trimming)
        final_results = []
        current_tokens = 0
        
        for c in candidates:
            tokens = len(self.tokenizer.encode(c["text"]))
            
            # Stop if we exceed the token budget
            if current_tokens + tokens > max_tokens:
                # Truncate the text if it's the very first chunk and it's too big
                if not final_results:
                    c["text"] = self.tokenizer.decode(self.tokenizer.encode(c["text"])[:max_tokens])
                    final_results.append(c)
                break
                
            final_results.append(c)
            current_tokens += tokens
            
            # Stop if we hit our requested top_k
            if len(final_results) >= top_k:
                break
                
        return final_results

    def _build_where_clause(self, filters):
        """Convert a simple dict filter (e.g. {'source': 'book.pdf'}) into ChromaDB syntax."""
        if not filters:
            return None
        clauses = []
        for k, v in filters.items():
            if isinstance(v, list):
                clauses.append({k: {"$in": v}})
            else:
                clauses.append({k: v})
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}
