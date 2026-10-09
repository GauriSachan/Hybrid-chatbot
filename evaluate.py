"""
evaluate.py -- Part 5: automated evaluation for the whole pipeline (Parts 1-4).

Measures, from one labeled test file:
    1. Routing accuracy       did Part 3 pick the expected route?
    2. Answer correctness     do expected keywords appear in the answer? (skipped if not given)
    3. Groundedness           did BotResponse mark the answer as grounded in retrieved evidence?
    4. Citation validity      every [n] citation actually resolved to a real source?
    5. Fallback correctness   did it correctly say "not found" when expect_fallback is true?
    6. Latency                response time per question

Runs DIRECTLY against the pipeline (no server needed) so it's fast to run repeatedly while
tuning thresholds. test_server.py / the Streamlit Evaluation page cover the HTTP path.

qa_eval.json format (one item per test question):
    {
      "question": "...",
      "expected_route": "direct_llm|rag_user_docs|rag_kb|web_search|rag_decompose|clarify|out_of_scope",
      "expected_keywords": ["..."],   # optional: substrings the answer should contain
      "expect_fallback": false,       # optional: true if the bot should say "not found"
      "session_id": "..."             # optional: use a session with files already uploaded
    }

Usage:
    python evaluate.py qa_eval.json                       # live LLM (needs ANTHROPIC_API_KEY)
    python evaluate.py qa_eval.json --provider openai
    python evaluate.py qa_eval.json --fake                # no API key / no internet; checks
                                                            # routing + wiring only, not answer quality
    python evaluate.py qa_eval.json --out results.json
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from pipeline import Session, build_chatbot
from query_router import QueryRouter


def load_cases(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("qa file must be a non-empty JSON list")
    return data


def fake_chat_fn(messages, temperature=0.3):
    """Offline stand-in: echoes the first context passage with a citation, or a generic
    direct answer. Lets you check ROUTING and WIRING without an API key; answer-quality
    metrics (keyword match) will be unreliable with this -- use a real chat_fn for those."""
    last = messages[-1]["content"]
    if "<context>" in last:
        try:
            body = last.split("]", 1)[1].split("\n", 1)[1].split("\n\n")[0]
        except IndexError:
            body = ""
        return f"{body[:150]} [1]" if body else "I could not find this in the sources."
    return "This is a general-knowledge answer (fake LLM)."


def run(cases: list[dict], provider: str, use_fake: bool, rerank: bool) -> list[dict]:
    chat_fn = fake_chat_fn if use_fake else None
    bot = build_chatbot(provider=provider, chat_fn=chat_fn, rerank=rerank)
    router = bot.router   # reused for a routing-only pass if ever needed

    rows = []
    for case in cases:
        session = Session(id=case.get("session_id", "eval"))
        start = time.perf_counter()
        resp = bot.chat(case["question"], session)
        elapsed = time.perf_counter() - start

        actual_route = resp.route
        expected_route = case.get("expected_route")
        route_ok = (expected_route is None) or (actual_route == expected_route)

        expect_fallback = case.get("expect_fallback", False)
        said_not_found = bool(re.search(r"couldn'?t find|could not find", resp.answer, re.I)) \
                         or actual_route == "clarify"
        fallback_ok = (not expect_fallback) or said_not_found

        kws = case.get("expected_keywords") or []
        kw_hits = sum(1 for k in kws if k.lower() in resp.answer.lower())
        kw_ratio = (kw_hits / len(kws)) if kws else None
        answer_ok = (kw_ratio is None) or (kw_ratio >= 0.5) or expect_fallback

        rows.append({
            "question": case["question"],
            "expected_route": expected_route,
            "actual_route": actual_route,
            "route_ok": route_ok,
            "expect_fallback": expect_fallback,
            "said_not_found": said_not_found,
            "fallback_ok": fallback_ok,
            "keyword_ratio": kw_ratio,
            "answer_ok": answer_ok,
            "grounded": resp.grounded,
            "warnings": resp.warnings,
            "n_sources": len(resp.sources),
            "latency_s": round(elapsed, 3),
            "answer_preview": resp.answer[:160],
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    route_acc = sum(r["route_ok"] for r in rows) / n
    fallback_acc = sum(r["fallback_ok"] for r in rows) / n
    scored = [r for r in rows if r["keyword_ratio"] is not None]
    answer_acc = (sum(r["answer_ok"] for r in scored) / len(scored)) if scored else None
    grounded_candidates = [r for r in rows if r["actual_route"] not in ("direct_llm", "clarify", "out_of_scope")]
    grounded_pct = (sum(r["grounded"] for r in grounded_candidates) / len(grounded_candidates)
                    if grounded_candidates else None)
    citation_warnings = sum("invalid_citation_removed" in r["warnings"] for r in rows)
    avg_latency = sum(r["latency_s"] for r in rows) / n
    return {
        "n_questions": n,
        "routing_accuracy": round(route_acc, 3),
        "fallback_accuracy": round(fallback_acc, 3),
        "answer_keyword_accuracy": round(answer_acc, 3) if answer_acc is not None else None,
        "groundedness_rate": round(grounded_pct, 3) if grounded_pct is not None else None,
        "answers_with_invalid_citations": citation_warnings,
        "avg_latency_s": round(avg_latency, 3),
    }


def print_report(rows: list[dict], summary: dict):
    print(f"\n{'Question':48} {'Route':14} {'OK?':4} {'Grnd':5} {'t(s)':6}")
    print("-" * 85)
    for r in rows:
        route_mark = "OK" if r["route_ok"] else f"!= {r['expected_route']}"
        print(f"{r['question'][:46]:48} {r['actual_route']:14} {route_mark:12} "
              f"{str(r['grounded']):5} {r['latency_s']:<6}")
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k:28}: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("qa_file")
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    ap.add_argument("--fake", action="store_true", help="no API key needed; routing/wiring check only")
    ap.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder reranker")
    ap.add_argument("--out", help="write full per-question results to this JSON file")
    a = ap.parse_args()

    cases = load_cases(a.qa_file)
    rows = run(cases, a.provider, a.fake, rerank=not a.no_rerank)
    summary = summarize(rows)
    print_report(rows, summary)

    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, "results": rows}, indent=2), encoding="utf-8")
        print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
