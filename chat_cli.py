"""
chat_cli.py -- end-to-end terminal chatbot using Parts 1-4 (no UI needed).

    python ingestion.py build-kb ./kb_docs --manifest manifest.json     # once
    export ANTHROPIC_API_KEY=...                                        # or OPENAI_API_KEY
    python chat_cli.py [--provider openai] [--verify] [--quiet]

Commands inside the chat:
    /upload <path>   ingest a PDF/DOCX/TXT/MD into this session
    /docs            list this session's uploads
    /quit            exit (deletes this session's uploads)
"""
import argparse
import uuid

from ingestion import delete_session_docs, ingest_user_doc
from pipeline import Session, build_chatbot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    ap.add_argument("--verify", action="store_true", help="extra LLM groundedness check")
    ap.add_argument("--quiet", action="store_true", help="hide routing/debug info")
    a = ap.parse_args()

    bot = build_chatbot(a.provider, verify=a.verify)
    session = Session(id=str(uuid.uuid4())[:8])
    print("Chatbot ready. /upload <path>, /docs, /quit\n")
    try:
        while True:
            msg = input("you> ").strip()
            if not msg:
                continue
            if msg == "/quit":
                break
            if msg == "/docs":
                print("uploads:", session.user_docs or "none"); continue
            if msg.startswith("/upload "):
                try:
                    session.user_docs.append(ingest_user_doc(msg[8:].strip().strip('"'), session.id))
                    print(f"[uploaded] {session.user_docs[-1]}")
                except Exception as e:
                    print(f"[upload failed] {e}")
                continue
            r = bot.chat(msg, session)
            print(f"\nbot> {r.answer}")
            if r.sources:
                print("     sources:", "; ".join(f"[{s['id']}] {s['source']}" + (f" p.{s['page']}" if s['page'] else "")
                                                for s in r.sources))
            if not a.quiet:
                d = r.decision
                print(f"     (route={d['route']} via {d['stage']}, conf={d['confidence']}, "
                      f"grounded={r.grounded}, warnings={r.warnings})")
            print()
    finally:
        delete_session_docs(session.id)       # privacy: remove this session's uploads


if __name__ == "__main__":
    main()
