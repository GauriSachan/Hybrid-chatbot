

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
import json
import re
import time
import uuid
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

# =============================================================================
# SECTION 2: CONFIGURATION  (the only part you normally need to edit)
# =============================================================================

# >>> PUT YOUR TEAM'S BACKEND URL HERE <<<
API_URL = "http://localhost:8000/chat"

# >>> MOCK_MODE <<<
#   True  -> does NOT call the backend; returns built-in sample answers.
#   False -> sends a real POST request to API_URL.
MOCK_MODE = False

UPLOAD_TIMEOUT_SECONDS = 120         # Ingesting a file (chunk + embed) can be slow
REQUEST_TIMEOUT_SECONDS = 30        # How long to wait for the backend
MOCK_DELAY_SECONDS = 0.4            # Fake delay in mock mode (looks realistic)
SEND_HISTORY_TO_BACKEND = False     # True -> also send {"history": [...]} in the request

# Shown to the user when the backend finds nothing relevant.
FALLBACK_MESSAGE = (
    "I'm sorry, I couldn't find a relevant answer in the knowledge base. "
    "Please try rephrasing your question or ask about a different topic."
)

# If the backend's answer contains one of these phrases (or is empty),
# the app treats it as "no relevant answer found" and shows the fallback.
NO_ANSWER_PHRASES = [
    "i don't know",
    "i do not know",
    "no relevant",
    "cannot find",
    "can't find",
    "unable to find",
    "not found in the",
]

# Evaluation thresholds
ACCURACY_KEYWORD_THRESHOLD = 0.5   # share of expected keywords that must appear in an answer
GROUNDEDNESS_THRESHOLD = 0.6       # score at or above this counts as "grounded"

# Default test questions (editable in the Evaluation page as JSON).
#   expected_keywords : words/phrases a correct answer should contain
#   expect_fallback   : True if the chatbot SHOULD say "no answer found"
DEFAULT_TEST_CASES = [
    {"question": "What is RAG?",
     "expected_keywords": ["retrieval-augmented generation", "language model"],
     "expect_fallback": False},
    {"question": "What are embeddings?",
     "expected_keywords": ["vector", "meaning"],
     "expect_fallback": False},
    {"question": "Why is chunking used in RAG?",
     "expected_keywords": ["smaller", "passages"],
     "expect_fallback": False},
    {"question": "What is a vector database?",
     "expected_keywords": ["stores", "similarity"],
     "expect_fallback": False},
    {"question": "How does RAG reduce hallucinations?",
     "expected_keywords": ["retriev", "grounded"],
     "expect_fallback": False},
    {"question": "What is the capital of Mars?",
     "expected_keywords": [],
     "expect_fallback": True},
]

SAMPLE_QUESTIONS = [
    "What is RAG?",
    "What are embeddings?",
    "How does RAG reduce hallucinations?",
    "What is the capital of Mars?",
]

# =============================================================================
# SECTION 3: MOCK BACKEND  (used only when MOCK_MODE = True)
# =============================================================================
# Each entry: if ANY trigger word appears in the question, that answer is used.
# Specific topics come BEFORE the general "RAG" entry on purpose.
MOCK_KNOWLEDGE = [
    {
        "triggers": ["hallucination", "hallucinations"],
        "answer": ("RAG reduces hallucinations by retrieving relevant passages and "
                   "instructing the model to answer only from that retrieved context, "
                   "so answers stay grounded and can be cited."),
        "sources": ["rag_overview.pdf, page 12"],
        "contexts": ["RAG reduces hallucinations by retrieving relevant passages and "
                     "instructing the model to answer only from the retrieved context. "
                     "Answers stay grounded in the source documents and can be cited."],
    },
    {
        "triggers": ["chunking", "chunk", "chunks"],
        "answer": ("Chunking splits long documents into smaller passages so each piece "
                   "can be embedded and retrieved accurately."),
        "sources": ["data_pipeline.pdf, page 3"],
        "contexts": ["Chunking splits long documents into smaller passages so each piece "
                     "can be embedded and retrieved accurately by the retrieval system."],
    },
    {
        "triggers": ["vector database", "vector db", "vector store"],
        "answer": ("A vector database stores embeddings and quickly finds the most "
                   "similar ones using similarity search."),
        "sources": ["vector_db_guide.pdf, page 2"],
        "contexts": ["A vector database stores embeddings and quickly finds the most "
                     "similar ones using similarity search."],
    },
    {
        "triggers": ["embedding", "embeddings"],
        "answer": ("Embeddings are numerical vectors that capture the meaning of text, "
                   "so similar texts end up close together in vector space."),
        "sources": ["embeddings_basics.pdf, page 1"],
        "contexts": ["Embeddings are numerical vectors that capture the meaning of text. "
                     "Similar texts end up close together in vector space."],
    },
    {
        "triggers": ["rag", "retrieval-augmented generation", "retrieval augmented generation"],
        "answer": ("RAG stands for Retrieval-Augmented Generation. It retrieves relevant "
                   "document passages and gives them to a language model so the answer "
                   "is grounded in your own data."),
        "sources": ["rag_overview.pdf, page 5"],
        "contexts": ["RAG stands for Retrieval-Augmented Generation. It retrieves relevant "
                     "document passages and gives them to a language model so that the "
                     "answer is grounded in your own data."],
    },
]


def contains_word(text: str, phrase: str) -> bool:
    """True if `phrase` appears in `text` as a whole word/phrase (case-insensitive)."""
    return re.search(r"\b" + re.escape(phrase.lower()) + r"\b", text.lower()) is not None


def mock_backend_response(question: str) -> dict:
    """Pretend to be the backend. Returns the same JSON shape as the real API."""
    time.sleep(MOCK_DELAY_SECONDS)

    # Lets you test the error-handling UI: ask "simulate error".
    if "simulate error" in question.lower():
        raise requests.exceptions.ConnectionError("Simulated connection failure (mock mode).")

    for entry in MOCK_KNOWLEDGE:
        if any(contains_word(question, t) for t in entry["triggers"]):
            return {"answer": entry["answer"],
                    "sources": entry["sources"],
                    "contexts": entry["contexts"]}

    # Nothing matched -> empty answer, which triggers the fallback response.
    return {"answer": "", "sources": []}


# =============================================================================
# SECTION 4: BACKEND API INTEGRATION
# =============================================================================
def call_real_backend(question: str, history: list | None = None, session_id: str | None = None) -> dict:
    """
    Sends POST {API_URL} with JSON {"question": "...", "session_id": "..."} and returns the
    parsed JSON. Expected response: {"answer": "...", "sources": ["file.pdf, page 5"]}
    (An optional "contexts": ["retrieved text", ...] improves groundedness scoring.)

    session_id lets the backend keep conversation history and uploaded-file access private to
    this browser session -- required so the chatbot's RAG-over-uploads feature works safely.
    """
    payload = {"question": question, "session_id": session_id}
    if SEND_HISTORY_TO_BACKEND and history:
        payload["history"] = history

    response = requests.post(
        API_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()   # Raises HTTPError for 4xx / 5xx
    return response.json()        # Raises ValueError if body is not valid JSON


def end_backend_session(session_id: str) -> None:
    """Tell the backend to delete this session's history and uploaded files. Never raises."""
    if MOCK_MODE or not session_id:
        return
    try:
        base = API_URL.rsplit("/chat", 1)[0]
        requests.post(f"{base}/session/{session_id}/end", timeout=5)
    except Exception:
        pass


def upload_to_backend(uploaded_file, session_id: str) -> tuple:
    """POST a file to the backend's /upload. Returns (ok, message). Never raises."""
    try:
        base = API_URL.rsplit("/chat", 1)[0]
        response = requests.post(
            f"{base}/upload",
            data={"session_id": session_id},
            files={"file": (uploaded_file.name, uploaded_file.getvalue(),
                            uploaded_file.type or "application/octet-stream")},
            timeout=UPLOAD_TIMEOUT_SECONDS,
        )
        if response.status_code == 400:
            return False, response.json().get("detail", "The backend rejected this file.")
        response.raise_for_status()
        return True, response.json().get("filename", uploaded_file.name)
    except requests.exceptions.Timeout:
        return False, f"Upload timed out after {UPLOAD_TIMEOUT_SECONDS} seconds."
    except requests.exceptions.ConnectionError:
        return False, f"Could not connect to the backend at {API_URL}."
    except Exception as exc:
        return False, f"Upload failed: {exc}"


def to_source_list(raw) -> list:
    """Turns whatever the backend sent as 'sources' into a clean list of strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, (list, tuple)):
        output = []
        for item in raw:
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                name = item.get("source") or item.get("title") or item.get("file") or item.get("document")
                page = item.get("page")
                if name and page is not None:
                    text = f"{name}, page {page}"
                elif name:
                    text = str(name)
                else:
                    text = json.dumps(item, ensure_ascii=False)
            else:
                text = str(item)
            if text.strip():
                output.append(text.strip())
        return output
    return [str(raw)]


def to_context_list(raw) -> list:
    """Turns the optional 'contexts' field into a list of strings."""
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw]
    output = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, str):
                output.append(item)
            elif isinstance(item, dict):
                output.append(str(item.get("text") or item.get("content") or ""))
    return [c for c in output if c.strip()]


def parse_backend_response(data) -> dict:
    """Validates the backend JSON and converts it to our internal format."""
    if not isinstance(data, dict):
        raise ValueError("Backend response must be a JSON object.")
    answer = data.get("answer", "")
    if answer is None:
        answer = ""
    return {
        "answer": str(answer).strip(),
        "sources": to_source_list(data.get("sources")),
        "contexts": to_context_list(data.get("contexts")),
    }


def is_no_answer(answer: str) -> bool:
    """True if the answer is empty or looks like 'I don't know'."""
    if not answer.strip():
        return True
    lowered = answer.lower()
    return any(phrase in lowered for phrase in NO_ANSWER_PHRASES)


def ask_backend(question: str, history: list | None = None, session_id: str | None = None) -> dict:
    """
    THE function the whole app uses to talk to the chatbot backend.
    It never raises: all problems are returned in result["error"].

    Returns a dict with:
        answer, sources, contexts, response_time (seconds), error, is_fallback
    """
    result = {"answer": "", "sources": [], "contexts": [],
              "response_time": 0.0, "error": None, "is_fallback": False}
    start = time.perf_counter()

    try:
        raw = mock_backend_response(question) if MOCK_MODE else call_real_backend(question, history, session_id)
        result.update(parse_backend_response(raw))
    except requests.exceptions.Timeout:
        result["error"] = (f"The backend did not respond within {REQUEST_TIMEOUT_SECONDS} seconds. "
                           "Please try again.")
    except requests.exceptions.ConnectionError:
        result["error"] = (f"Could not connect to the backend at {API_URL}. "
                           "Check that the server is running, or set MOCK_MODE = True.")
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "unknown"
        result["error"] = f"The backend returned an error (HTTP {code})."
    except ValueError as exc:   # Bad JSON or wrong structure
        result["error"] = f"The backend sent an invalid response: {exc}"
    except Exception as exc:    # Last-resort safety net
        result["error"] = f"Unexpected error: {exc}"

    result["response_time"] = time.perf_counter() - start

    # Fallback: the request worked, but there is no usable answer.
    if result["error"] is None and is_no_answer(result["answer"]):
        result["answer"] = FALLBACK_MESSAGE
        result["sources"] = []
        result["contexts"] = []
        result["is_fallback"] = True

    return result


# =============================================================================
# SECTION 5: EVALUATION LOGIC
# =============================================================================
STOPWORDS = {
    "the", "and", "for", "are", "that", "this", "with", "from", "was", "were", "has",
    "have", "had", "can", "will", "its", "your", "you", "our", "their", "into", "than",
    "then", "them", "they", "but", "not", "all", "any", "each", "such", "also", "which",
    "when", "what", "who", "how", "why", "use", "used", "using", "does", "did", "been",
}


def tokenize(text: str) -> set:
    """Lowercase, keep word-like tokens, drop short words and stopwords."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def keyword_match_ratio(answer: str, expected_keywords: list) -> float:
    """Share of expected keywords found in the answer (1.0 if none are expected)."""
    if not expected_keywords:
        return 1.0
    lowered = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in lowered)
    return found / len(expected_keywords)


def evaluate_groundedness(answer: str, sources: list, contexts: list) -> dict:
    """
    Basic groundedness / faithfulness structure.
    Question answered: "Is the answer supported by the retrieved evidence?"

    Method A - "context_overlap" (used when the backend returns 'contexts'):
        score = share of the answer's meaningful words that appear in the contexts.
    Method B - "citation_check" (used when only 'sources' are available):
        score = 1.0 if at least one source is cited, else 0.0.

    This is a simple heuristic. To upgrade it later, change ONLY this function
    (for example, to an LLM-as-judge or an NLI model) and keep the return format.
    """
    if contexts:
        answer_tokens = tokenize(answer)
        context_tokens = tokenize(" ".join(contexts))
        score = (len(answer_tokens & context_tokens) / len(answer_tokens)) if answer_tokens else 0.0
        method = "context_overlap"
    else:
        score = 1.0 if sources else 0.0
        method = "citation_check"

    return {"score": round(score, 2),
            "grounded": score >= GROUNDEDNESS_THRESHOLD,
            "method": method}


def parse_test_cases(text: str) -> list:
    """Reads the JSON typed in the UI and validates it. Raises ValueError if invalid."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}")
    if not isinstance(data, list) or not data:
        raise ValueError("Test cases must be a non-empty JSON list.")

    cases = []
    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict) or not str(item.get("question", "")).strip():
            raise ValueError(f"Test case #{i} needs a non-empty 'question'.")
        keywords = item.get("expected_keywords", [])
        if not isinstance(keywords, list):
            raise ValueError(f"Test case #{i}: 'expected_keywords' must be a list.")
        cases.append({
            "question": str(item["question"]).strip(),
            "expected_keywords": [str(k) for k in keywords],
            "expect_fallback": bool(item.get("expect_fallback", False)),
        })
    return cases


def run_evaluation(test_cases: list, progress_bar=None) -> list:
    """Sends every test question to the backend and scores each response."""
    rows = []
    for i, case in enumerate(test_cases, start=1):
        result = ask_backend(case["question"], session_id=f"eval-{uuid.uuid4()}")  # fresh session per test: no shared history
        ratio = keyword_match_ratio(result["answer"], case["expected_keywords"])
        groundedness, grounded_label = None, None

        if result["error"]:
            status, correct, ratio = "Error", False, None
        elif result["is_fallback"]:
            status, ratio = "Fallback", None
            correct = case["expect_fallback"]          # Correct only if a fallback was expected
        else:
            status = "Answered"
            correct = (not case["expect_fallback"]) and ratio >= ACCURACY_KEYWORD_THRESHOLD
            g = evaluate_groundedness(result["answer"], result["sources"], result["contexts"])
            groundedness, grounded_label = g["score"], "Yes" if g["grounded"] else "No"

        rows.append({
            "#": i,
            "Question": case["question"],
            "Answer": result["error"] or result["answer"],
            "Sources": "; ".join(result["sources"]),
            "Status": status,
            "Keyword Match": None if ratio is None else round(ratio, 2),
            "Correct": "Yes" if correct else "No",
            "Groundedness": groundedness,
            "Grounded?": grounded_label,
            "Response Time (s)": round(result["response_time"], 3),
        })
        if progress_bar is not None:
            progress_bar.progress(i / len(test_cases), text=f"Testing question {i} of {len(test_cases)}...")
    return rows


def compute_metrics(rows: list) -> dict:
    """Turns result rows into the summary numbers shown on the dashboard."""
    total = len(rows)
    correct = sum(1 for r in rows if r["Correct"] == "Yes")
    times = [r["Response Time (s)"] for r in rows if r["Status"] != "Error"]
    scores = [r["Groundedness"] for r in rows if r["Groundedness"] is not None]
    answered = [r for r in rows if r["Status"] == "Answered"]
    grounded = sum(1 for r in answered if r["Grounded?"] == "Yes")
    with_sources = sum(1 for r in answered if r["Sources"])

    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total * 100) if total else 0.0,
        "avg_time": (sum(times) / len(times)) if times else 0.0,
        "max_time": max(times) if times else 0.0,
        "avg_groundedness": (sum(scores) / len(scores)) if scores else 0.0,
        "grounded_pct": (grounded / len(answered) * 100) if answered else 0.0,
        "citation_pct": (with_sources / len(answered) * 100) if answered else 0.0,
        "fallbacks": sum(1 for r in rows if r["Status"] == "Fallback"),
        "errors": sum(1 for r in rows if r["Status"] == "Error"),
    }


# =============================================================================
# SECTION 6: CHAT HELPERS (conversation history)
# =============================================================================
def build_backend_history(messages: list) -> list:
    """Converts stored messages to a simple [{"role","content"}] list for the backend."""
    return [{"role": m["role"], "content": m["content"]}
            for m in messages if not m.get("error")]


def render_message_body(msg: dict):
    """Draws the inside of one chat bubble: text, fallback notice, sources, timing."""
    if msg.get("error"):
        st.error(msg["error"])
        return

    st.markdown(msg["content"])

    if msg["role"] == "assistant":
        if msg.get("is_fallback"):
            st.info("No relevant answer was found in the knowledge base.")
        elif msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})", expanded=True):
                for source in msg["sources"]:
                    st.markdown(f"- {source}")
        if msg.get("response_time") is not None:
            st.caption(f"⏱ {msg['response_time']:.2f}s · {msg.get('timestamp', '')}")


def render_message(msg: dict):
    with st.chat_message(msg["role"]):
        render_message_body(msg)


def set_pending_question(question: str):
    """Button callback: remembers a clicked sample question."""
    st.session_state.pending_question = question


def reset_test_cases():
    """Button callback: restores the default test questions."""
    st.session_state.test_cases_text = json.dumps(DEFAULT_TEST_CASES, indent=2)


# =============================================================================
# SECTION 7: CHAT PAGE
# =============================================================================
def render_chat_page():
    st.title("🤖 RAG Knowledge Assistant")
    st.caption("Ask a question and get an answer with source citations.")

    # Read input first, so the new question is saved before history is drawn.
    prompt = st.chat_input("Type your question here...")
    pending = st.session_state.pop("pending_question", None)
    if pending:
        prompt = pending

    history_before = build_backend_history(st.session_state.messages)

    if prompt:
        st.session_state.messages.append({
            "role": "user", "content": prompt,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        })

    # Welcome screen with clickable sample questions
    if not st.session_state.messages:
        st.info("👋 Welcome! Type a question below, or try one of these:")
        columns = st.columns(len(SAMPLE_QUESTIONS))
        for col, question in zip(columns, SAMPLE_QUESTIONS):
            col.button(question, key=f"sample_{question}",
                       on_click=set_pending_question, args=(question,))

    # Draw the whole conversation so far
    for msg in st.session_state.messages:
        render_message(msg)

    # Get the answer for the new question
    if prompt:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask_backend(prompt, history_before, st.session_state.session_id)

            assistant_msg = {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "is_fallback": result["is_fallback"],
                "error": result["error"],
                "response_time": result["response_time"],
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
            st.session_state.messages.append(assistant_msg)
            render_message_body(assistant_msg)


# =============================================================================
# SECTION 8: EVALUATION PAGE (dashboard)
# =============================================================================
def render_evaluation_page():
    st.title("📊 Evaluation Dashboard")
    st.caption("Run test questions through the chatbot and measure accuracy, speed and groundedness.")

    mode_text = "MOCK MODE (sample data)" if MOCK_MODE else f"LIVE backend: {API_URL}"
    st.info(f"Evaluation will use: **{mode_text}**")

    # ---- Test question editor ----
    st.subheader("1. Test questions")
    st.text_area(
        "Edit the test questions (JSON list). Fields: question, expected_keywords, expect_fallback.",
        key="test_cases_text", height=260,
    )
    col_run, col_reset, _ = st.columns([1, 1, 3])
    run_clicked = col_run.button("▶ Run evaluation", type="primary")
    col_reset.button("↺ Reset defaults", on_click=reset_test_cases)

    if run_clicked:
        try:
            test_cases = parse_test_cases(st.session_state.test_cases_text)
        except ValueError as exc:
            st.error(str(exc))
        else:
            progress = st.progress(0.0, text="Starting evaluation...")
            st.session_state.eval_rows = run_evaluation(test_cases, progress)
            progress.empty()
            st.success(f"Evaluation finished: {len(test_cases)} questions tested.")

    rows = st.session_state.eval_rows
    if not rows:
        st.write("Click **Run evaluation** to see results.")
        return

    metrics = compute_metrics(rows)

    # ---- Summary metrics ----
    st.subheader("2. Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.1f}%", f"{metrics['correct']}/{metrics['total']} correct", delta_color="off")
    c2.metric("Avg response time", f"{metrics['avg_time']:.2f}s")
    c3.metric("Avg groundedness", f"{metrics['avg_groundedness']:.2f}")
    c4.metric("Answers with sources", f"{metrics['citation_pct']:.0f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Grounded answers", f"{metrics['grounded_pct']:.0f}%")
    c6.metric("Slowest response", f"{metrics['max_time']:.2f}s")
    c7.metric("Fallback responses", metrics["fallbacks"])
    c8.metric("Errors", metrics["errors"])

    # ---- Detailed results ----
    st.subheader("3. Detailed results")
    df = pd.DataFrame(rows)
    st.dataframe(df)

    st.subheader("4. Response time per question")
    st.bar_chart(df.set_index("#")["Response Time (s)"])

    # ---- Downloads ----
    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button("⬇ Results (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="evaluation_results.csv", mime="text/csv")
    d2.download_button("⬇ Results (JSON)", json.dumps(rows, indent=2, ensure_ascii=False),
                       file_name="evaluation_results.json", mime="application/json")

    with st.expander("How are these metrics calculated?"):
        st.markdown(
            f"""
- **Accuracy** = correct answers ÷ total questions. A question is *correct* when the chatbot
  answers and at least **{int(ACCURACY_KEYWORD_THRESHOLD * 100)}%** of the expected keywords appear in
  the answer. If `expect_fallback` is true, it is correct only when the chatbot says it has no answer.
- **Average response time** = mean time (seconds) of all requests that did not fail.
- **Groundedness** (0 to 1) estimates whether the answer is supported by evidence. If the backend
  returns `contexts`, it is the share of answer words found in those contexts. Otherwise it is
  1 if a source is cited and 0 if not. Scores ≥ **{GROUNDEDNESS_THRESHOLD}** count as grounded.
- **Fallback / Error** rows are counted separately and have no groundedness score.
"""
        )


# =============================================================================
# SECTION 9: SIDEBAR + MAIN
# =============================================================================
def render_sidebar() -> str:
    """Draws the sidebar and returns the selected page name."""
    with st.sidebar:
        st.header("🤖 RAG Chatbot")
        page = st.radio("Navigation", ["💬 Chat", "📊 Evaluation"], label_visibility="collapsed")

        st.divider()
        st.subheader("Backend status")
        if MOCK_MODE:
            st.warning("MOCK MODE is ON (sample answers)")
        else:
            st.success("LIVE MODE")
            st.caption(f"Endpoint: `{API_URL}`")

        st.divider()
        st.subheader("Session")
        messages = st.session_state.messages
        questions = sum(1 for m in messages if m["role"] == "user")
        times = [m["response_time"] for m in messages
                 if m["role"] == "assistant" and not m.get("error") and m.get("response_time") is not None]
        st.write(f"Questions asked: **{questions}**")
        if times:
            st.write(f"Avg response time: **{sum(times) / len(times):.2f}s**")

        st.divider()
        st.subheader("Your documents")
        if MOCK_MODE:
            st.caption("Uploads are disabled in mock mode. Set MOCK_MODE = False to use them.")
        else:
            st.caption("Files are private to this session and deleted when you clear the conversation.")
            files = st.file_uploader("Upload a document to ask about",
                                     accept_multiple_files=True,
                                     key=f"uploader_{st.session_state.uploader_key}")
            for f in files or []:
                file_id = f"{f.name}:{f.size}"
                if file_id in st.session_state.uploaded_ids:
                    continue                               # already sent (Streamlit reruns often)
                with st.spinner(f"Processing {f.name}..."):
                    ok, message = upload_to_backend(f, st.session_state.session_id)
                if ok:
                    st.session_state.uploaded_ids.add(file_id)
                    st.session_state.uploaded_docs.append(message)
                else:
                    st.error(f"{f.name}: {message}")
            for name in st.session_state.uploaded_docs:
                st.markdown(f"✅ {name}")

        if st.button("🗑 Clear conversation"):
            end_backend_session(st.session_state.session_id)       # wipes server history + uploads
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.uploaded_docs = []
            st.session_state.uploaded_ids = set()
            st.session_state.uploader_key += 1             # forces the uploader widget to empty
            st.session_state.messages = []
            st.rerun()

        if messages:
            st.download_button("⬇ Download chat (JSON)",
                               json.dumps(messages, indent=2, ensure_ascii=False),
                               file_name="chat_history.json", mime="application/json")
    return page


def main():
    # set_page_config must be the first Streamlit command.
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem;}
        [data-testid="stMetric"] {
            background: rgba(128,128,128,0.08);
            border: 1px solid rgba(128,128,128,0.2);
            padding: 12px 16px; border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Session state = memory that survives between button clicks.
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []                # filenames the backend accepted
    if "uploaded_ids" not in st.session_state:
        st.session_state.uploaded_ids = set()              # name:size keys, to avoid re-uploading
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())     # one id per browser tab, for the backend
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "eval_rows" not in st.session_state:
        st.session_state.eval_rows = None
    if "test_cases_text" not in st.session_state:
        st.session_state.test_cases_text = json.dumps(DEFAULT_TEST_CASES, indent=2)

    page = render_sidebar()
    if page == "💬 Chat":
        render_chat_page()
    else:
        render_evaluation_page()


if __name__ == "__main__":
    main()