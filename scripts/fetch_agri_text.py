"""Fetch Marathi Wikipedia agriculture articles (CC BY-SA 4.0) as source text for Varhadi translation.

Writes data/agri_text/sources.tsv (one row per sentence, `varhadi` column empty for a native speaker)
and data/agri_text/provenance.json (title, revid, url per article). Standard Marathi only: this is
the SOURCE side of a translation task, not Varhadi.
"""
import csv
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "agri_text"
API = "https://mr.wikipedia.org/w/api.php"
UA = {"User-Agent": "KisaanDost-research/0.1 (atharvagujarathi0405@gmail.com)"}
# title -> crop tag (crop_scope.json crops + general)
PAGES = {"कापूस": "cotton", "शेंदरी बोंडअळी": "cotton", "बोंड अळी": "cotton", "तुडतुडा": "cotton",
         "सोयाबीन": "soybean", "तूर": "tur", "नागपुरी संत्री": "orange",
         "ठिबक सिंचन": "general", "तुषार सिंचन": "general", "प्रधानमंत्री पीक विमा योजना": "general",
         "बीजप्रक्रिया": "general", "सेंद्रिय शेती": "general"}


def get(**p):
    url = API + "?" + urllib.parse.urlencode({"format": "json", **p})
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA)) as r:
        return json.load(r)


def sentences(text: str):
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("=="):  # drop blanks and section headings
            continue
        for s in re.split(r"(?<=[.?!।])\s+", line):
            if len(s) >= 25 and re.search(r"[ऀ-ॿ]", s):
                yield s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prov, rows = [], []
    for title, crop in PAGES.items():
        q = get(action="query", prop="extracts|revisions", explaintext=1, titles=title, rvprop="ids", redirects=1)
        page = next(iter(q["query"]["pages"].values()))
        if "extract" not in page:
            print("MISSING", title)
            continue
        prov.append({"title": page["title"], "revid": page["revisions"][0]["revid"], "crop": crop,
                     "url": "https://mr.wikipedia.org/wiki/" + urllib.parse.quote(page["title"].replace(" ", "_")),
                     "licence": "CC BY-SA 4.0 (Wikipedia contributors)"})
        for i, s in enumerate(sentences(page["extract"])):
            rows.append({"id": f"wiki_{len(prov):02d}_{i:03d}", "crop": crop, "article": page["title"],
                         "marathi": s, "varhadi": "", "verified": "false"})
    with open(OUT / "sources.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys(), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    (OUT / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(prov), "articles,", len(rows), "sentences")


if __name__ == "__main__":
    main()
