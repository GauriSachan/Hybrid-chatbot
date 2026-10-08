"""
test_integration.py -- end-to-end wiring test of Parts 1-4 with NO model downloads / API keys.

Uses a hashing bag-of-words embedder and a fake LLM so it runs anywhere:
    python test_integration.py
For a real-quality test, delete the set_embedder() line (uses MiniLM) and pass a real chat_fn.
"""
import os, tempfile, zlib
from pathlib import Path

os.environ["KB_DATA_DIR"] = tempfile.mkdtemp()           # must be set BEFORE importing kb_store

import numpy as np
import kb_store
from kb_store import tokenize


def hash_embed(texts):
    out = np.zeros((len(texts), 384), dtype=np.float32)
    for r, t in enumerate(texts):
        for w in tokenize(t):
            out[r, zlib.crc32(w.encode()) % 384] += 1.0
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)


kb_store.set_embedder(hash_embed, 384)                   # stand-in for MiniLM

import json
from ingestion import build_kb, delete_session_docs, ingest_user_doc
from hybrid_retriever import HybridRetriever
from pipeline import Chatbot, Session

tmp = Path(tempfile.mkdtemp())
kb = tmp / "kb"; kb.mkdir()
(kb / "Eiffel_Tower.txt").write_text(
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. Construction of the Eiffel Tower "
    "began in 1887 and the tower was completed in 1889 for the World's Fair. It is 330 metres tall.")
(kb / "Marie_Curie.txt").write_text(
    "Marie Curie was a Polish-French physicist and chemist. She was the first woman to win a Nobel Prize "
    "in 1903 and the only person to win Nobel Prizes in two different sciences.")
(tmp / "manifest.json").write_text(json.dumps(
    {"Eiffel_Tower.txt": {"title": "Eiffel Tower", "url": "https://en.wikipedia.org/wiki/Eiffel_Tower"},
     "Marie_Curie.txt": {"title": "Marie Curie", "url": "https://en.wikipedia.org/wiki/Marie_Curie"}}))
build_kb(kb, tmp / "manifest.json")

upload = tmp / "plan.txt"
upload.write_text("Project Falcon status report. The project deadline is 30 November 2026. "
                  "The total budget approved for Project Falcon is 45,000 dollars.")


def fake_llm(messages, temperature=0.3):                 # echoes the first retrieved passage + cites it
    last = messages[-1]["content"]
    if "<context>" in last:
        body = last.split("]", 1)[1].split("\n", 1)[1].split("\n\n")[0]
        return f"From the sources: {body[:90]}... [1]"
    return "Sure, here is a direct answer from general knowledge."


bot = Chatbot(fake_llm, HybridRetriever(), embed_fn=kb_store.embed_fn)   # no web search -> disabled
s1, s2 = Session("s1"), Session("s2")
s1.user_docs.append(ingest_user_doc(upload, "s1", "plan.txt"))

checks = []
def ask(sess, msg, expect_route, expect_src=None):
    r = bot.chat(msg, sess)
    ok = r.route == expect_route and (expect_src is None or expect_src in [x["source"] for x in r.sources])
    checks.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  [{sess.id}] {msg!r:55} -> {r.route:14} src={[x['source'] for x in r.sources]}")
    return r

ask(s1, "hello!", "direct_llm")
ask(s1, "explain how black holes form", "direct_llm")
ask(s1, "when was the Eiffel Tower built?", "rag_kb", "Eiffel Tower")
ask(s1, "who was the first woman to win a Nobel Prize?", "rag_kb", "Marie Curie")
ask(s1, "what does my document say about the deadline?", "rag_user_docs", "plan.txt")
ask(s2, "what does my document say about the deadline?", "clarify")        # s2 uploaded nothing
r = ask(s1, "what's the latest news today?", "rag_kb")                       # web disabled -> KB

# Privacy: another session must never see s1's uploads
leak = HybridRetriever().search("project deadline", sources=["user_docs"], session_id="s2")
checks.append(leak == []); print(f"{'PASS' if leak == [] else 'FAIL'}  session isolation (s2 sees {len(leak)} chunks)")

delete_session_docs("s1")
gone = HybridRetriever().search("project deadline", sources=["user_docs"], session_id="s1")
checks.append(gone == []); print(f"{'PASS' if gone == [] else 'FAIL'}  uploads deleted on session end")

print(f"\n{sum(checks)}/{len(checks)} checks passed")
