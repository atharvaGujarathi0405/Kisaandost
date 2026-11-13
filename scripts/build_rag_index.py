"""Download the public farm documents, extract their text and cut it into chunks for the bot's retrieval step (bot/rag.py).

  python -I scripts/build_rag_index.py            # download missing files, then write data/rag/chunks.jsonl

Self-contained on purpose (no repo imports): downloaded files are untrusted, so this runs in isolated mode (-I).
Files land in data/rag/raw/<id>/ (one folder each), which is gitignored: the documents belong to their publishers, we only
keep the extracted text locally and cite the source with every answer. TLS is always verified; a site that fails is skipped.
Chunks keep source id, title, publisher, url, page and the crops they mention. Nothing here is checked by a native speaker
or an agronomist: it is the publishers' text, cut up.
"""
import html.parser
import json
import re
import sys
from pathlib import Path

import pymupdf as fitz  # PyMuPDF
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data" / "rag" / "raw", ROOT / "data" / "rag" / "chunks.jsonl"
MAX_BYTES = 40_000_000
UA = {"User-Agent": "Mozilla/5.0 (KisaanDost research; contact via repository)"}
# id, title, publisher, url, kind
SOURCES = [
    ("pdkv-recom-2025", "Agricultural Recommendations 2025 (Joint Agresco, Vidarbha/PDKV)", "Dr. PDKV Akola", "https://www.pdkv.ac.in/Circulars/Recom-2025English.pdf", "pdf"),
    ("pdkv-recom-2024", "Agricultural Recommendations 2024 (Vidarbha/PDKV)", "Dr. PDKV Akola", "https://www.pdkv.ac.in/Circulars/Recom-2024English.pdf", "pdf"),
    ("pdkv-techinv-iv", "Technology Inventory IV: commodity-wise package of practices", "Dr. PDKV Akola", "https://www.pdkv.ac.in/pdf/technology-inventory-IV.pdf", "pdf"),
    ("pdkv-contingency", "Kharif contingency planning, Vidarbha", "Dr. PDKV Akola", "https://www.pdkv.ac.in/Circulars/cont-planningeng-14.pdf", "pdf"),
    ("cicr-ipm-guide", "Illustrated guide for integrated management of cotton pests, diseases and nematodes", "ICAR-CICR Nagpur", "https://cicr.org.in/wp-content/uploads/2026/08/CICR_Illustrative-guide-for-integrated-management-for-Cotton-Pests-Diseases-and-Nematodes.pdf", "pdf"),
    ("crida-pigeonpea", "Rainfed pigeonpea: district-wise guide (Madhya Pradesh, Maharashtra, Uttar Pradesh)", "ICAR-CRIDA", "https://www.icar-crida.res.in/assets/img/Books/2005-06/Pigeonpea.pdf", "pdf"),
    ("mahapocra-crt", "Climate resilient technologies booklet (Maharashtra)", "MahaPoCRA, Govt of Maharashtra", "https://mahapocra.gov.in/documents/CRT_Booklet.pdf", "pdf"),
    ("crida-amravati", "District agriculture contingency plan, Amravati", "ICAR-CRIDA / PDKV Akola", "https://icar-crida.res.in/CP/Maharastra/PDKV,%20Akola/Amravati.pdf", "pdf"),
    ("vikas-mandarin", "Nagpur mandarin: package of practices", "Vikaspedia (Govt of India)", "https://en.vikaspedia.in/viewcontent/agriculture/crop-production/package-of-practices/fruits-1/nagpur-mandarin?lgn=en", "html"),
    ("vikas-soybean", "Soybean cultivation: package of practices", "Vikaspedia (Govt of India)", "https://en.vikaspedia.in/viewcontent/agriculture/crop-production/package-of-practices/oilseeds/a-comprehensive-guide-to-soybean-cultivation?lgn=en", "html"),
    ("vikas-pigeonpea", "Pigeon pea (tur): package of practices", "Vikaspedia (Govt of India)", "https://en.vikaspedia.in/viewcontent/agriculture/crop-production/package-of-practices/pulses/pigeon-pea?lgn=en", "html"),
]
CROPS = {"cotton": r"cotton|कापूस|कपाशी|bollworm|bt\b", "soybean": r"soy ?bean|soya|सोयाबीन|glycine", "tur": r"pigeon ?pea|\btur\b|arhar|तूर|cajanus",
         "orange": r"mandarin|orange|citrus|संत्र|मोसंबी|nagpur"}
CHUNK, OVERLAP, MIN_CHUNK = 900, 150, 200
# PDF ligatures that come out as private characters (Ɵ = "ti", Ʃ/ƫ = "tt"); bullets and private-use glyphs become plain dashes
FIX = str.maketrans({"Ɵ": "ti", "Ʃ": "tt", "ƫ": "tt", "ﬁ": "fi", "ﬂ": "fl", "": "-", "": " ", "": "-", "": "-"})
# Old-font Marathi in the PDFs is extracted as Latin-Extended/IPA letters mixed into Devanagari ("ɟपक"): that text is unusable
GARBAGE = re.compile("[ƀ-ʯ-]")


class Text(html.parser.HTMLParser):
    """Visible text of an HTML page, one block per paragraph/list item/heading/table cell."""
    SKIP = {"script", "style", "nav", "header", "footer", "noscript", "form", "svg"}
    BLOCK = {"p", "li", "h1", "h2", "h3", "h4", "h5", "td", "th", "tr", "div", "br"}

    def __init__(self):
        super().__init__()
        self.out, self.skip = [""], 0

    def handle_starttag(self, tag, attrs):
        self.skip += tag in self.SKIP
        if tag in self.BLOCK:
            self.out.append("")

    def handle_endtag(self, tag):
        self.skip -= tag in self.SKIP and self.skip > 0

    def handle_data(self, data):
        if not self.skip:
            self.out[-1] += data


def download(sid, url, kind):
    d = RAW / sid
    f = d / ("doc.pdf" if kind == "pdf" else "page.html")
    if f.exists():
        return f
    try:
        r = requests.get(url, headers=UA, timeout=60, stream=True)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if (kind == "pdf") != ("pdf" in ct):
            raise ValueError(f"unexpected content-type {ct}")
        body = b""
        for part in r.iter_content(1 << 20):
            body += part
            if len(body) > MAX_BYTES:
                raise ValueError("file larger than 40 MB")
    except Exception as e:  # one unreachable source must not stop the rest
        print(f"  SKIP {sid}: {e}")
        return None
    d.mkdir(parents=True, exist_ok=True)
    f.write_bytes(body)
    print(f"  downloaded {sid}: {len(body) / 1e6:.1f} MB")
    return f


def pages(f, kind):
    """Yield (page_number, text)."""
    if kind == "pdf":
        with fitz.open(f) as doc:
            for i, p in enumerate(doc, 1):
                yield i, p.get_text()
    else:
        raw = f.read_text(encoding="utf-8", errors="replace")
        m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', raw, re.S)  # Vikaspedia keeps the article in this JSON
        if m:
            raw = json.loads(m.group(1))["props"]["pageProps"]["ssrPageContent"]["content"]
        t = Text()
        t.feed(raw)
        yield 1, "\n".join(s.strip() for s in t.out if len(s.strip()) > 25)


def cut(text):
    text = re.sub(r"\s+", " ", text).strip()
    while len(text) > CHUNK:
        end = max(text.rfind(". ", 0, CHUNK), text.rfind(" ", 0, CHUNK))
        end = end if end > CHUNK // 2 else CHUNK
        yield text[:end + 1].strip()
        text = text[max(end - OVERLAP, 1):].strip()
    yield text


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(OUT, "w", encoding="utf-8") as out:
        for sid, title, pub, url, kind in SOURCES:
            f = download(sid, url, kind)
            if not f:
                continue
            k = 0
            for page, text in pages(f, kind):
                for c in cut(text.translate(FIX)):
                    if len(GARBAGE.findall(c)) > 0.01 * len(c):
                        continue
                    letters = sum(ch.isalpha() for ch in c)
                    if len(c) < MIN_CHUNK or letters < 0.5 * len(c):  # drop page numbers, empty tables, scanned-image noise
                        continue
                    crops = [k_ for k_, rx in CROPS.items() if re.search(rx, c, re.I)]
                    out.write(json.dumps({"id": f"{sid}:{page}:{k}", "source": sid, "title": title, "publisher": pub, "url": url,
                                          "page": page, "crops": crops, "text": c}, ensure_ascii=False) + "\n")
                    k += 1
            n += k
            print(f"  {sid}: {k} chunks")
    print(f"{n} chunks -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
