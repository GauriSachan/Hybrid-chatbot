"""
fetch_web_pages.py -- pulls documents from ANY site (not just Wikipedia) into ./kb_docs/
and merges them into manifest.json, so  python ingestion.py build-kb ./kb_docs --manifest
manifest.json  picks them up exactly like the Wikipedia articles from make_kb_docs.py.

Handles two kinds of URLs:
    - HTML pages: strips nav/ads/scripts, keeps the readable article text.
    - Direct links to .pdf / .docx files: downloaded as-is; ingestion.py's own pdf/docx
      loader (pypdf / python-docx) extracts the text when you build the KB.

Usage:
    python fetch_web_pages.py --urls "https://example.com/article-1,https://example.com/a.pdf"
    python fetch_web_pages.py --urls-file sources.txt
    python fetch_web_pages.py --check https://example.com    # test ONE url, see what gets extracted

sources.txt format -- one URL per line, optional custom title after a pipe:
    https://www.investopedia.com/terms/i/inflation.asp | Inflation (Investopedia)
    https://www.geeksforgeeks.org/machine-learning/    | Machine Learning Basics
    https://arxiv.org/abs/1706.03762.pdf

Notes:
    - Respect each site's Terms of Service / robots.txt before scraping it for a project.
      Wikipedia, arXiv, government (.gov) sites, and most documentation sites (readthedocs,
      official framework docs) are generally fine. Many news sites and paywalled sites are not.
    - pip install beautifulsoup4
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (RAG-Chatbot-Student-Project/1.0; educational use)"}
_STRIP_TAGS = ["script", "style", "nav", "header", "footer", "aside", "form", "iframe",
              "noscript", "svg", "button", "input", "figure"]
_JUNK_CLASSES = re.compile(r"(nav|menu|sidebar|footer|header|ad|advert|cookie|banner|"
                           r"popup|social|share|comment|related|breadcrumb)", re.I)


def explain_error(e: Exception) -> str:
    import socket, ssl
    if isinstance(e, urllib.error.HTTPError):
        if e.code == 403:
            return "HTTP 403 - blocked (the site may reject scrapers; check its robots.txt)."
        if e.code == 404:
            return "HTTP 404 - page not found. Check the URL."
        if e.code == 429:
            return "HTTP 429 - rate limited. Wait and retry, or space out requests."
        return f"HTTP {e.code}."
    reason = getattr(e, "reason", e)
    if isinstance(reason, ssl.SSLError):
        return "SSL error - likely a proxy/firewall intercepting HTTPS."
    if isinstance(reason, socket.gaierror):
        return "DNS failure - no internet, or the domain is blocked/misspelled."
    if isinstance(reason, (socket.timeout, TimeoutError)):
        return "Timed out - slow or blocked connection."
    return f"{type(e).__name__}: {e}"


def fetch_url(url: str, retries: int = 3, timeout: int = 30):
    """-> (content_type: str, raw_bytes: bytes)"""
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
                return ctype, r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2 * attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt < retries:
                time.sleep(2 * attempt)
                continue
            raise


def extract_article(html: bytes) -> tuple[str, str]:
    """-> (title, clean_text). Keeps <main>/<article> if present, else <body>."""
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("title")
    og = soup.find("meta", property="og:title")
    title = (og.get("content") if og else None) or (title_tag.text if title_tag else "") or "Untitled"
    title = re.sub(r"\s+", " ", title).strip()

    for tag in soup(_STRIP_TAGS):
        tag.decompose()
    for tag in soup.find_all(class_=_JUNK_CLASSES):
        tag.decompose()
    for tag in soup.find_all(id=_JUNK_CLASSES):
        tag.decompose()

    main = soup.find("article") or soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
    text = main.get_text(separator="\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)
    lines = [l.strip() for l in text.split("\n")]
    text = "\n".join(l for l in lines if l)
    return title, text.strip()


def slugify(s: str) -> str:
    return re.sub(r"[^\w\-]+", "_", s).strip("_")[:80] or "untitled"


def process_one(url: str, out: Path, manifest: dict, custom_title: Optional[str] = None) -> dict:
    try:
        ctype, raw = fetch_url(url)
    except Exception as e:
        return {"url": url, "status": "error", "detail": explain_error(e)}

    if ctype == "application/pdf" or url.lower().endswith(".pdf"):
        title = custom_title or Path(urllib.parse.urlparse(url).path).stem.replace("_", " ")
        fname = slugify(title) + ".pdf"
        (out / fname).write_bytes(raw)
        manifest[fname] = {"title": title, "url": url}
        return {"url": url, "status": "ok", "file": fname, "chars": len(raw), "type": "pdf"}

    if "html" not in ctype and ctype != "":
        return {"url": url, "status": "skipped", "detail": f"unsupported content-type: {ctype}"}

    title, text = extract_article(raw)
    if custom_title:
        title = custom_title
    if len(text) < 200:
        return {"url": url, "status": "too_short",
                "detail": f"only {len(text)} chars extracted -- page may be JS-rendered or paywalled"}

    fname = slugify(title) + ".txt"
    (out / fname).write_text(text, encoding="utf-8")
    manifest[fname] = {"title": title, "url": url}
    return {"url": url, "status": "ok", "file": fname, "chars": len(text), "type": "html"}


def load_url_list(args) -> list[tuple[str, Optional[str]]]:
    pairs = []
    if args.urls_file:
        for line in Path(args.urls_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                url, title = line.split("|", 1)
                pairs.append((url.strip(), title.strip()))
            else:
                pairs.append((line, None))
    elif args.urls:
        pairs = [(u.strip(), None) for u in args.urls.split(",") if u.strip()]
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kb_docs")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--urls", help="comma-separated URLs")
    ap.add_argument("--urls-file", help="text file: one URL per line, optional '| Title'")
    ap.add_argument("--check", metavar="URL", help="fetch ONE url and print what would be extracted, "
                    "without saving anything -- use this first to sanity-check a new site")
    a = ap.parse_args()

    if a.check:
        ctype, raw = fetch_url(a.check)
        if "pdf" in ctype or a.check.lower().endswith(".pdf"):
            print(f"Content-Type: {ctype}\nWould save as a .pdf file, {len(raw):,} bytes -- "
                  f"text gets extracted later by ingestion.py's pdf loader.")
            return
        title, text = extract_article(raw)
        print(f"Title: {title}\nExtracted {len(text):,} chars\n\n--- First 500 chars ---\n{text[:500]}")
        return

    pairs = load_url_list(a)
    if not pairs:
        ap.error("give --urls, --urls-file, or --check")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mp = Path(a.manifest)
    manifest = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}

    results = []
    for url, title in pairs:
        r = process_one(url, out, manifest, custom_title=title)
        results.append(r)
        tag = {"ok": "ok     ", "skipped": "skipped", "too_short": "too_short", "error": "error  "}[r["status"]]
        extra = r.get("file") or r.get("detail", "")
        print(f"[{tag}] {url}  -> {extra}")
        time.sleep(0.5)   # be polite

    mp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    ok = sum(r["status"] == "ok" for r in results)
    print(f"\nDone: {ok}/{len(results)} pages saved to ./{out}  +  {mp}")
    print(f"Next:  python ingestion.py build-kb ./{out} --manifest {mp}")


if __name__ == "__main__":
    main()
