# 🧠 Hybrid RAG Chatbot

**An intelligent, citation-grounded chatbot that knows when to search, what to search, and when to honestly say "I don't know."**

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688)
![Streamlit](https://img.shields.io/badge/frontend-Streamlit-FF4B4B)
![License](https://img.shields.io/badge/license-educational%20project-lightgrey)

---

## ✨ What this is

A general-purpose chatbot that doesn't just retrieve-for-everything or answer-from-memory-for-everything. A **query router** looks at each question and decides, per question, whether to:

| 🎯 Route | When |
|---|---|
| 💬 **Answer directly** | Chit-chat, math, coding, general explanations |
| 📚 **Search the knowledge base** | Factual questions about the ingested document corpus |
| 📎 **Search your uploaded files** | Questions about documents *you* uploaded this session |
| 🌐 **Search the web** | Current events, prices, anything time-sensitive |
| 🔀 **Decompose and search** | Comparisons / multi-part questions |
| ❓ **Ask for clarification** | Genuinely ambiguous follow-ups |
| 🚫 **Refuse** | Prompt injection / out-of-scope requests |

Every answer that draws on retrieved content is **cited inline** (`[1]`, `[2]`…), and the system is built to say *"I couldn't find this"* rather than invent an answer — including a one-shot **query-reformulation retry** that tries a reworded search before giving up.

---

## 🏗️ Architecture

```
                         ┌─────────────────────┐
  User question  ─────▶  │   Part 3: Router     │
                         │ rules → embeddings    │
                         │      → LLM fallback   │
                         └──────────┬───────────┘
                                    │
            ┌───────────────┬──────┼──────┬───────────────┐
            ▼               ▼      ▼      ▼               ▼
      Direct LLM      KB search  File search  Web search  Decompose
                            │          │
                            ▼          ▼
                 ┌─────────────────────────────┐
                 │  Part 2: Hybrid Retrieval     │
                 │  BM25 + Dense (FAISS) + RRF   │
                 │  + optional cross-encoder      │
                 └──────────────┬────────────────┘
                                 ▼
                 ┌─────────────────────────────┐
                 │ Part 4: Response Generator    │
                 │ grounded prompt · citations    │
                 │ hallucination guards · retry   │
                 └──────────────┬────────────────┘
                                 ▼
                      Cited, grounded answer
```

**Part 1 — Data & Knowledge Base:** `ingestion.py` + `kb_store.py` — chunking, embedding, SQLite + FAISS + BM25 storage, with separate, session-isolated collections for the shared KB vs. per-user uploads.

**Part 2 — Retrieval:** `hybrid_retriever.py` (+ optional `reranker.py`) — dense + sparse search fused with Reciprocal Rank Fusion, with an optional cross-encoder reranking pass.

**Part 3 — Query Router:** `query_router.py` — a 3-stage cascade (cheap rules → embedding similarity → LLM classification) that decides the route for every question.

**Part 4 — Generation:** `response_generator.py` + `pipeline.py` — grounded prompting, inline citations, invalid-citation stripping, an explicit no-evidence refusal, and a query-reformulation retry for weak-evidence answers. Works with **Gemini, Groq, GitHub Models, OpenAI, or Anthropic** — pick any one.

**Part 5 — App & Evaluation:** `server.py` (FastAPI backend) + `app.py` (Streamlit UI) + `evaluate.py` (automated routing/groundedness/citation evaluation).

---

## 🚀 Quick start

### 1. Install dependencies

```bash
pip install fastapi uvicorn python-multipart streamlit
pip install sentence-transformers faiss-cpu rank-bm25
pip install pypdf python-docx beautifulsoup4
pip install openai anthropic
```

### 2. Get a free API key (no credit card needed)

Pick **one**:

| Provider | Get a key at | Free tier |
|---|---|---|
| 🟢 **Gemini** *(default)* | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | ✅ No card, daily request cap |
| 🟡 **Groq** | [console.groq.com](https://console.groq.com) | ✅ No card, fast inference |
| ⚪ **GitHub Models** | [github.com/settings/personal-access-tokens](https://github.com/settings/personal-access-tokens) | ✅ Uses your GitHub account |
| 🔵 Anthropic / OpenAI | respective consoles | 💳 Paid |

### 3. Set your credentials

```powershell
# PowerShell
$env:GEMINI_API_KEY = "your-key-here"
$env:GEMINI_MODEL   = "gemini-3.8-flash"      # check current model names in the console
$env:CHATBOT_PROVIDER = "gemini"
```
```bash
# bash / macOS / Linux
export GEMINI_API_KEY="your-key-here"
export GEMINI_MODEL="gemini-3.8-flash"
export CHATBOT_PROVIDER="gemini"
```

### 4. Build the knowledge base

```bash
python make_kb_docs.py                              # pulls Wikipedia articles
python fetch_web_pages.py --urls-file sources.txt    # optional: add other sites
python ingestion.py build-kb ./kb_docs --manifest manifest.json
python ingestion.py stats                            # confirm it worked
```

### 5. Run it

```bash
# Terminal 1 — backend (leave running)
uvicorn server:app --port 8000

# Terminal 2 — frontend
streamlit run app.py
```

Open the Streamlit URL, set `MOCK_MODE = False` in `app.py` if you haven't already, and start chatting. 🎉

---

## 📁 Project structure

```
├── ingestion.py            # Part 1: load → clean → chunk → embed → store
├── kb_store.py              # Part 1: SQLite + FAISS + BM25 storage layer
├── make_kb_docs.py          # Part 1: Wikipedia article downloader
├── fetch_web_pages.py       # Part 1: ingest from any website / PDF
│
├── hybrid_retriever.py      # Part 2: BM25 + dense retrieval, RRF fusion
├── reranker.py               # Part 2: optional cross-encoder reranking
│
├── query_router.py          # Part 3: rules → embeddings → LLM routing cascade
│
├── response_generator.py    # Part 4: grounded generation, citations, hallucination guards
├── pipeline.py               # Part 4: orchestrates router → retriever → generator
│
├── server.py                 # Part 5: FastAPI backend (/chat, /upload, /session/*)
├── app.py                    # Part 5: Streamlit chat UI
├── chat_cli.py                # Part 5: terminal chatbot (no UI, for quick testing)
├── evaluate.py                # Part 5: automated evaluation harness
│
├── qa.json / qa_eval.json    # Test question sets
└── manifest.json / kb_docs/  # Knowledge base source documents
```

---

## 🛡️ Hallucination control, by design

- Retrieved chunks below a similarity threshold are **filtered out**, not used.
- The generation prompt restricts the model to **only** the retrieved context.
- Citations are **validated** — any `[n]` pointing to a non-existent source is stripped.
- A weak first search triggers **one automatic retry** with a reformulated query before giving up.
- If nothing relevant is found, the bot says so explicitly — it does not guess.

✅ **Verified:** questions with real-world answers absent from the knowledge base (e.g. *"What is the capital of Liechtenstein?"*) correctly produce an honest "I couldn't find this" rather than a fabricated, confidently-cited answer.

---

## 📊 Results

| Metric | Result |
|---|---|
| Knowledge base size | **168 documents, 18,599 chunks** |
| Retrieval hit@5 (known-answer test) | **3/3 = 100%** |
| Routing accuracy | **12/12 = 100%** *(improved from 83.3% after a rule-coverage fix)* |
| Invalid citations observed | **0** |
| Hallucination spot-check | **Passed** — honest refusal on out-of-KB factual questions |

Run your own evaluation anytime:
```bash
python evaluate.py qa_eval.json --no-rerank --delay 13 --out results.json
```

---

## ⚙️ Configuration notes

- **Speed vs. quality:** the cross-encoder reranker (`rerank=True` by default) improves retrieval quality but adds latency. Disable it in `server.py` for faster responses: `build_chatbot(provider="gemini", rerank=False)`.
- **Free-tier limits:** Gemini's free tier caps requests per day (watch for `RESOURCE_EXHAUSTED` errors). The built-in retry logic handles transient `503`/`429` errors automatically, but a daily quota needs either a wait or a provider/model switch.
- **Web search:** not yet wired to a real search API — `web_search`-routed questions currently fall back to the static knowledge base. A real web-search function can be plugged into `Chatbot(web_search_fn=...)`.

---

## 👥 Team

| Part | Module | Member |
|---|---|---|
| 1 | Data & Knowledge Base | Palak Gupta |
| 2 | RAG Retrieval System | Aaditya Jain |
| 3 | Intelligent Query Router | Gauri Sachan |
| 4 | LLM + Response Generation | Naman Awasthi |
| 5 | Chatbot App + Evaluation | Neeti Totla |

---

<p align="center"><i>Built as a final-year / Review-2 project — adaptive routing + hybrid retrieval, with hallucination control as a first-class design goal.</i></p>
