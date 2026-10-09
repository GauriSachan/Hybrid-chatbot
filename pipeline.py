"""
pipeline.py  --  glue: Part 3 (router) -> Part 2 (retrieval) -> Part 4 (generation)

Part 5's chat API/UI should only ever call  Chatbot.chat(...).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from query_router import QueryRouter, Route
from response_generator import BotResponse, Chunk, ChatFn, ResponseGenerator


class Retriever(Protocol):
    """Interface Part 2 must implement."""
    def search(self, query: str, top_k: int = 5, sources: Optional[list[str]] = None,
               session_id: Optional[str] = None) -> list[Chunk]:
        """sources: ["kb"] | ["user_docs"]. user_docs must be filtered by session_id."""


WebSearchFn = Callable[..., list[Chunk]]      # web_search(query, num_results=5) -> [Chunk(origin="web")]


@dataclass
class Session:
    id: str
    history: list[dict] = field(default_factory=list)   # [{"role","content"}]
    user_docs: list[str] = field(default_factory=list)  # filenames uploaded this session

    @property
    def has_user_docs(self) -> bool:
        return bool(self.user_docs)


class Chatbot:
    def __init__(self, chat_fn: ChatFn, retriever: Retriever,
                 web_search_fn: Optional[WebSearchFn] = None,
                 embed_fn=None, verify: bool = False):
        self.retriever = retriever
        self.web_search_fn = web_search_fn
        self.router = QueryRouter(
            embed_fn=embed_fn,
            llm_fn=lambda prompt: chat_fn([{"role": "user", "content": prompt}], 0.0),  # router's LLM stage
            web_search_enabled=web_search_fn is not None,
        )
        self.generator = ResponseGenerator(chat_fn, verify=verify)

    def chat(self, user_msg: str, session: Session) -> BotResponse:
        # Part 3: decide
        d = self.router.route(user_msg, history=session.history,
                              has_user_docs=session.has_user_docs)

        # Part 2 / web: fetch context according to the decision
        chunks = None
        if d.route in (Route.RAG_USER_DOCS, Route.RAG_KB):
            chunks = self.retriever.search(d.query, session_id=session.id, **d.retrieval_params)
        elif d.route is Route.RAG_DECOMPOSE:
            chunks = {s: self.retriever.search(s, session_id=session.id, **d.retrieval_params)
                      for s in d.sub_queries}
        elif d.route is Route.WEB_SEARCH and self.web_search_fn:
            chunks = self.web_search_fn(d.query, **d.retrieval_params)

        # Part 4: generate
        resp = self.generator.generate(d, chunks, session.history)

        # Remember the turn (store the user's original words)
        session.history += [{"role": "user", "content": user_msg},
                            {"role": "assistant", "content": resp.answer}]
        return resp


# --------------------------------------------------------------------------- #
# Factory: wires the REAL Part 1 + Part 2 into the chatbot
# --------------------------------------------------------------------------- #
def build_chatbot(provider: str = "anthropic", chat_fn: Optional[ChatFn] = None,
                  web_search_fn: Optional[WebSearchFn] = None, verify: bool = False,
                  rerank: bool = True) -> Chatbot:
    """Part 5 (API/UI) should create the bot with this, once at startup.

    rerank=True (default): wraps retrieval with the cross-encoder reranker (reranker.py).
    If the model can't be loaded (no internet, package missing, etc.) this prints a warning
    and falls back to plain hybrid retrieval rather than crashing startup."""
    from hybrid_retriever import HybridRetriever      # Part 2
    from kb_store import embed_fn, get_store          # Part 1 (shared embedding model)
    if chat_fn is None:
        from response_generator import anthropic_chat_fn, openai_chat_fn, groq_chat_fn, gemini_chat_fn
        chat_fn = {"anthropic": anthropic_chat_fn, "openai": openai_chat_fn,
                   "groq": groq_chat_fn, "gemini": gemini_chat_fn}[provider]()

    retriever = HybridRetriever(get_store())
    if rerank:
        try:
            from reranker import RerankingRetriever, sentence_transformers_cross_encoder
            retriever = RerankingRetriever(retriever, rerank_fn=sentence_transformers_cross_encoder())
        except Exception as e:
            print(f"[build_chatbot] Reranker unavailable ({e}); using plain hybrid retrieval.")

    return Chatbot(chat_fn, retriever, web_search_fn=web_search_fn, embed_fn=embed_fn, verify=verify)


# --------------------------------------------------------------------------- #
# Demo with fakes -- runs with no API key.  Swap in the real pieces later.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    class FakeRetriever:
        def search(self, query, top_k=5, sources=None, session_id=None):
            if sources == ["user_docs"]:
                return [Chunk("The project deadline is 30 November 2026.", "plan.pdf", 0.82, page=2, origin="user_docs")]
            if "eiffel" in query.lower():
                return [Chunk("The Eiffel Tower was completed in 1889.", "Eiffel Tower", 0.9, url="https://en.wikipedia.org/wiki/Eiffel_Tower"),
                        Chunk("Irrelevant text about cooking.", "Cooking", 0.1)]
            return []   # nothing found

    def fake_llm(messages, temperature=0.3):
        last = messages[-1]["content"]
        if "<context>" in last:
            return "According to the sources, the answer is stated [1]. Extra claim [7]."
        return "Sure -- here is a direct answer from general knowledge."

    bot = Chatbot(fake_llm, FakeRetriever(), web_search_fn=None)
    s = Session("demo")
    for msg, files in [("hello!", []), ("explain how black holes form", []),
                       ("when was the Eiffel Tower built?", []),
                       ("who was the first person on Mars?", []),            # KB has nothing
                       ("what does my PDF say about the deadline?", []),     # nothing uploaded
                       ("what does my PDF say about the deadline?", ["plan.pdf"])]:
        s.user_docs = files
        r = bot.chat(msg, s)
        print(f"\nUSER : {msg}   (uploads={files})\nROUTE: {r.route}\nBOT  : {r.answer}"
              f"\nSRC  : {[x['source'] for x in r.sources]}  warnings={r.warnings}")
