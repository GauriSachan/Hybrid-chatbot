"""
make_kb_docs.py -- creates ./kb_docs/*.txt and ./manifest.json from Wikipedia articles,
so that   python ingestion.py build-kb ./kb_docs --manifest manifest.json   works.

    python make_kb_docs.py                          # ~35 default general-knowledge topics
    python make_kb_docs.py --check                   # test the connection to Wikipedia first
    python make_kb_docs.py --sample                  # offline: 6 small built-in docs, no internet
    python make_kb_docs.py --topics "Tiger,Moon,Jupiter"
    python make_kb_docs.py --topics-file my_topics.txt     # one article title per line

Needs internet access to en.wikipedia.org. No extra packages (standard library only).
You can ALSO drop your own .txt / .md / .pdf / .docx files into kb_docs/ -- if a file is not
listed in manifest.json, its filename is used as the citation title.
"""
import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "RAG-Chatbot-Student-Project/1.0 (educational use)"}

DEFAULT_TOPICS = [
    "Eiffel Tower", "Marie Curie", "Albert Einstein", "Isaac Newton", "Photosynthesis",
    "Black hole", "World War II", "World War I", "Roman Empire", "French Revolution",
    "Industrial Revolution", "Great Wall of China", "Mount Everest", "Amazon rainforest",
    "Nile", "Apollo 11", "Mahatma Gandhi", "India", "Taj Mahal", "Python (programming language)",
    "Machine learning", "Artificial intelligence", "Internet", "DNA", "Climate change",
    "Solar System", "Mars", "Leonardo da Vinci", "William Shakespeare", "Napoleon",
    "Nelson Mandela", "Berlin Wall", "Australia", "Brazil", "Japan", "Nobel Prize",
]


# Small original sample articles for OFFLINE testing (--sample). Written for this project.
SAMPLE_DOCS = {
    "Eiffel_Tower": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built it. Construction began in 1887 and the tower was completed in 1889 for the World's Fair, which marked the centennial of the French Revolution. The tower is about 330 metres tall, roughly the height of an 81-storey building. It was the tallest man-made structure in the world until the Chrysler Building was completed in New York in 1930. Today it is one of the most visited paid monuments in the world, attracting millions of visitors every year.",
    "Marie_Curie": "Marie Curie was a Polish-born physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize, receiving the Nobel Prize in Physics in 1903 together with her husband Pierre Curie and Henri Becquerel. In 1911 she won a second Nobel Prize, in Chemistry, for discovering the elements polonium and radium. She remains the only person to have won Nobel Prizes in two different scientific fields. During World War I she developed mobile X-ray units, known as petites Curies, to help treat wounded soldiers. She died in 1934 from a condition linked to long exposure to radiation.",
    "Photosynthesis": "Photosynthesis is the process by which plants, algae and some bacteria convert light energy into chemical energy. In plants it takes place mainly in the chloroplasts of leaf cells, where the green pigment chlorophyll absorbs sunlight. Using this energy, the plant combines carbon dioxide from the air with water from the soil to produce glucose and oxygen. The overall reaction can be summarised as carbon dioxide plus water plus light energy producing glucose plus oxygen. The oxygen is released into the atmosphere, which is why photosynthesis is essential for most life on Earth. The glucose is used for growth or stored as starch.",
    "Python_programming_language": "Python is a high-level, general-purpose programming language created by Guido van Rossum and first released in 1991. It emphasises code readability and uses indentation to define blocks of code. Python supports multiple programming styles, including procedural, object-oriented and functional programming. It is widely used in web development, data science, machine learning, automation and scientific computing. The language has a large standard library and a huge ecosystem of third-party packages available through the Python Package Index, known as PyPI. Python 3.0 was released in 2008 and is not fully backward compatible with Python 2.",
    "Solar_System": "The Solar System is the gravitationally bound system of the Sun and the objects that orbit it. It formed about 4.6 billion years ago from the collapse of a giant molecular cloud. The Sun contains more than 99 percent of the total mass of the system. Eight planets orbit the Sun: Mercury, Venus, Earth and Mars are the rocky inner planets, while Jupiter, Saturn, Uranus and Neptune are the outer giant planets. Jupiter is the largest planet. Pluto was reclassified as a dwarf planet in 2006 by the International Astronomical Union. Beyond Neptune lies the Kuiper Belt, a region of icy bodies.",
    "Great_Wall_of_China": "The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect against nomadic invasions. Construction began as early as the 7th century BC, and several dynasties added to it over the centuries. The best-known sections were built during the Ming dynasty, between the 14th and 17th centuries. The wall including all its branches stretches for more than 20,000 kilometres. Contrary to a popular myth, it is not visible to the naked eye from the Moon. It was designated a UNESCO World Heritage Site in 1987.",
}

_CUT = re.compile(r"\n=+\s*(See also|References|External links|Further reading|Notes|Bibliography|Sources|Footnotes)\s*=+", re.I)


def _opener(proxy: str | None):
    """urllib already honours HTTP(S)_PROXY env vars; --proxy overrides them."""
    if proxy:
        return urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener()


_OPENER = _opener(None)


def explain_error(e: Exception) -> str:
    """Turn a network exception into a human-readable cause + fix."""
    import socket, ssl
    if isinstance(e, urllib.error.HTTPError):
        if e.code == 403:
            return "HTTP 403 - request blocked (firewall/proxy, or the host is not allowed on this network)."
        if e.code == 429:
            return "HTTP 429 - rate limited by Wikipedia; wait a minute and rerun (finished files are kept)."
        return f"HTTP {e.code} from Wikipedia."
    reason = getattr(e, "reason", e)
    if isinstance(reason, ssl.SSLError) or "CERTIFICATE" in str(reason).upper():
        return ("SSL certificate error - usually a company/school proxy or antivirus intercepting HTTPS. "
                "Try: pip install --upgrade certifi, a different network (e.g. phone hotspot), or --proxy.")
    if isinstance(reason, socket.gaierror):
        return "DNS failure - no internet, or wikipedia.org is blocked. Check your connection / VPN / hotspot."
    if isinstance(reason, (socket.timeout, TimeoutError)) or "timed out" in str(reason):
        return "Timed out - slow or blocked network. Rerun, or try --proxy / another network."
    if isinstance(reason, ConnectionRefusedError) or "refused" in str(reason).lower():
        return "Connection refused - a proxy/firewall is blocking it. Try --proxy or another network."
    return f"{type(e).__name__}: {e}"


def fetch(title: str, retries: int = 3):
    """-> (canonical_title, plain_text) or None if the article does not exist.
    Retries on temporary failures; raises on permanent ones."""
    q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1,
                                "redirects": 1, "titles": title, "format": "json"})
    req = urllib.request.Request(f"{API}?{q}", headers=HEADERS)
    for attempt in range(1, retries + 1):
        try:
            with _OPENER.open(req, timeout=30) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(float(e.headers.get("Retry-After", 2 * attempt)))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt < retries:
                time.sleep(2 * attempt)
                continue
            raise
    page = next(iter(data["query"]["pages"].values()))
    if "missing" in page or not page.get("extract"):
        return None
    return page["title"], page["extract"]


def check_connection() -> bool:
    print("Testing connection to en.wikipedia.org ...")
    try:
        res = fetch("Eiffel Tower", retries=1)
        print(f"OK - received '{res[0]}' ({len(res[1]):,} chars). You can run the full download." if res
              else "Reached Wikipedia but got no article (unexpected).")
        return bool(res)
    except Exception as e:
        print("FAILED:", explain_error(e))
        return False


def clean(text: str) -> str:
    m = _CUT.search(text)
    if m:                                             # drop reference/link sections
        text = text[:m.start()]
    text = re.sub(r"^=+\s*.+?\s*=+\s*$", "", text, flags=re.M)   # remove "== Heading ==" lines
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kb_docs")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--topics", help="comma-separated article titles")
    ap.add_argument("--topics-file", help="text file, one article title per line")
    ap.add_argument("--check", action="store_true", help="only test the connection to Wikipedia and exit")
    ap.add_argument("--proxy", help="e.g. http://user:pass@host:8080 (if your network needs one)")
    ap.add_argument("--sample", action="store_true",
                    help="OFFLINE: write 6 small built-in sample documents + manifest (no internet needed)")
    a = ap.parse_args()

    global _OPENER
    _OPENER = _opener(a.proxy)
    if a.check:
        raise SystemExit(0 if check_connection() else 1)

    if a.sample:
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for name, text in SAMPLE_DOCS.items():
            (out / f"{name}.txt").write_text(text, encoding="utf-8")
            manifest[f"{name}.txt"] = {"title": name.replace("_", " "), "url": ""}
            print(f"[ok     ] {name}.txt  ({len(text):,} chars)")
        Path(a.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nDone: {len(SAMPLE_DOCS)} sample docs in ./{out}  +  {a.manifest}")
        print(f"Next:  python ingestion.py build-kb ./{out} --manifest {a.manifest}")
        return

    if a.topics_file:
        topics = [l.strip() for l in Path(a.topics_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    elif a.topics:
        topics = [t.strip() for t in a.topics.split(",") if t.strip()]
    else:
        topics = DEFAULT_TOPICS

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mp = Path(a.manifest)
    manifest = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}   # keep earlier entries

    ok = 0
    for t in topics:
        try:
            res = fetch(t)
        except Exception as e:
            print(f"[error  ] {t}: {explain_error(e)}")
            if ok == 0 and t == topics[0]:
                print("\nCannot reach Wikipedia, so stopping early. Run  python make_kb_docs.py --check  "
                      "to diagnose, or use  --sample  for offline test documents.")
                raise SystemExit(1)
            continue
        if not res:
            print(f"[missing] {t}"); continue
        title, text = res
        text = clean(text)
        fname = re.sub(r"[^\w\-]+", "_", title).strip("_") + ".txt"
        (out / fname).write_text(text, encoding="utf-8")
        manifest[fname] = {"title": title,
                           "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))}
        print(f"[ok     ] {fname}  ({len(text):,} chars)")
        ok += 1
        time.sleep(0.3)                               # be polite to Wikipedia

    mp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone: {ok}/{len(topics)} articles in ./{out}  +  {mp}")
    print(f"Next:  python ingestion.py build-kb ./{out} --manifest {mp}")


if __name__ == "__main__":
    main()
