"""Grow the substitution knowledge base for farm text, with evidence, and list every farm swap for native review.

  python -m scripts.build_farm_kb                 # writes bot/farm_kb.json and data/agri_text/farm_substitution_candidates.csv
  python -m scripts.build_farm_kb --regenerate    # first re-run the unfiltered translator on the farm sentences (about 25 min on CPU)
  python -m scripts.build_farm_kb --merge-reviewed  # after a native speaker fills native_verdict=yes in the CSV

Why this exists: the bot's strict filter (bot/mr2vh_model.py) only accepts word changes seen in the scripture training pairs, and
farm words never occur there, so farm answers stay Marathi. Measured on held-out scripture, the changes the filter throws away match
the real Varhadi only 19% of the time (31-33% when the Varhadi word also occurs in the real text; 3.5% for invented words), so the filter
is NOT simply loosened. Instead:
  1. RULES: swaps from a short pool that I vetted for meaning (a developer's check, NOT a native speaker's) are applied as rules only
     when the real Varhadi text backs them with numbers (thresholds below). Dative -ले on farm nouns is accepted on the strength of the
     pattern itself (the best-attested Varhadi feature) plus the noun occurring in the farm sentences.
  2. REVIEW LIST: every swap the unfiltered translator proposes on the farm sentences, with its evidence and an empty native_verdict column.
     Rows marked yes become `allowed_extra` (the translator may then make that swap) via --merge-reviewed.
Everything here is UNVERIFIED by a native speaker.
"""
import argparse
import collections
import csv
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bot import mr2vh_model as M  # noqa: E402

TRANS = ROOT / "data" / "translation"
FARM_CSV = ROOT / "datasets" / "mr2vh_farming_dataset.csv"
PROPOSALS = ROOT / "data" / "agri_text" / "farm_model_proposals.jsonl"
REVIEW = ROOT / "data" / "agri_text" / "farm_substitution_candidates.csv"
KB = ROOT / "bot" / "farm_kb.json"
W = re.compile("[ऀ-ॿ]+")

# NOT in the pool on purpose: सर्व -> सगळे. Varhadi changes it with gender and number (सगळा / सगळी / सगळे / सगळ्या), so a context-free swap
# makes ungrammatical text ("सगळे माहिती"). Only swaps whose form does not depend on the neighbouring words belong here.
# Pool vetted by the developer for "same word, same meaning, Varhadi form". Lexical swaps must pass the numeric thresholds in `evidence`.
LEXICAL = [
    ("मग", "मंग"), ("आधी", "पयले"), ("पाहिजे", "पायजे"), ("पर्यंत", "परेंत"), ("आतापर्यंत", "आतापरेंत"),
    ("जोपर्यंत", "जोपरेंत"), ("स्वतः", "स्वता"), ("स्वतःच्या", "स्वताच्या"), ("काहीच", "काईच"), ("लोक", "लोकं"),
    ("दिले", "देले"), ("दिल्या", "देल्या"), ("पाहून", "पाऊन"), ("राहून", "राऊन"), ("पाहिले", "पायले"), ("पहिले", "पयले"),
    ("राहते", "रायते"), ("राहत", "रायत"), ("राहिली", "रायली"), ("राहिलो", "रायलो"), ("सांगितले", "सांगतले"), ("सांगितली", "सांगतली"),
    ("सांगितल्या", "सांगतल्या"), ("करायला", "कराले"), ("मिळते", "भेटते"), ("मिळेल", "भेटीन"), ("पाशी", "पासी"),
    ("त्याच्यापाशी", "त्याच्यापासी"), ("तुमच्यापाशी", "तुमच्यापासी"), ("डोळे", "डोये"),
]
# Nouns and pronouns that end bot answers or occur in the farm sentences, taken to the dative -ले (never verbs such as आला, गेला).
DATIVE = ["सहाय्यकाला", "खताला", "तज्ज्ञाला", "जिल्ह्याला", "ज्याला", "कोणाला", "शरीराला", "देवाला", "माणसाला", "आपल्याला",
          "पिकांना"]  # the last is dropped by the check below (plural, not attested on crops): kept here so the report shows why
MIN_TARGET, MIN_PREFERENCE = 8, 2          # real Varhadi uses y at least 8 times and at least twice as often as x


def read(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def corpora():
    rows = [r for f in ("train", "val", "test") for r in read(TRANS / f"{f}.jsonl")]
    vh = collections.Counter(M._core(w) for r in rows for w in r["target"].split())      # real Varhadi
    mr = collections.Counter(M._core(w) for r in rows for w in r["source"].split())      # machine Marathi
    al, occ = collections.Counter(), collections.Counter()
    for r in rows:
        a, b = r["source"].split(), r["target"].split()
        occ.update(M._core(x) for x in a)
        for op, i1, i2, j1, j2 in M._ops(a, b):
            if op in ("equal", "replace") and i2 - i1 == j2 - j1:
                al.update((M._core(x), M._core(y)) for x, y in zip(a[i1:i2], b[j1:j2]))
    return vh, mr, al, occ


def evidence(x, y, vh, mr, al, occ, farm):
    return {"varhadi_text_y": vh[y], "varhadi_text_x": vh[x], "marathi_side_x": mr[x], "aligned_pairs": al[(x, y)],
            "farm_sentences_x": farm[x]}


def build_rules(vh, mr, al, occ, farm):
    rules, ev, dropped = {}, {}, []
    have = set(json.loads((ROOT / "bot" / "varhadi_style.json").read_text(encoding="utf-8"))["enforce"]) |         set(json.loads((ROOT / "bot" / "varhadi_style.json").read_text(encoding="utf-8"))["enforce_by_rule"])
    pat_nouns = len({w for w in vh if w.endswith("ाले") and len(w) > 4})                 # distinct Varhadi dative nouns in the real text
    for x, y in LEXICAL:
        if x in have:
            dropped.append((x, y, "already one of the hand-checked word rules")); continue
        e = evidence(x, y, vh, mr, al, occ, farm)
        if e["varhadi_text_y"] < MIN_TARGET:
            dropped.append((x, y, f"Varhadi form seen only {e['varhadi_text_y']}x in the real text")); continue
        if e["varhadi_text_y"] < MIN_PREFERENCE * e["varhadi_text_x"]:
            dropped.append((x, y, f"real text uses {x} {e['varhadi_text_x']}x vs {y} {e['varhadi_text_y']}x")); continue
        if e["marathi_side_x"] + e["farm_sentences_x"] < 3:
            dropped.append((x, y, "the Marathi word is too rare to matter")); continue
        rules[x], ev[x] = y, e
    for x in DATIVE:
        y = x[:-2] + "ले"
        if x in have:
            dropped.append((x, y, "already one of the hand-checked word rules")); continue
        e = evidence(x, y, vh, mr, al, occ, farm); e["dative_nouns_in_real_text"] = pat_nouns
        if not x.endswith("ला") or pat_nouns < 50 or e["farm_sentences_x"] + e["marathi_side_x"] < 1:
            dropped.append((x, y, "not a singular dative noun seen in the farm or training text")); continue
        rules[x], ev[x] = y, e
    return rules, ev, dropped


def proposals(regenerate):
    if regenerate or not PROPOSALS.exists():
        from bot import server  # noqa: F401  (loads .env, not needed for the model)
        sents = sorted(((r["id"], r["marathi"]) for r in csv.DictReader(open(FARM_CSV, encoding="utf-8-sig"))
                        if len(r["marathi"].encode()) < 600), key=lambda s: len(s[1]))
        PROPOSALS.parent.mkdir(parents=True, exist_ok=True)
        with open(PROPOSALS, "w", encoding="utf-8") as f:
            for i in range(0, len(sents), 12):
                batch = sents[i:i + 12]
                for (id_, s), p in zip(batch, M._generate([s for _, s in batch])):
                    f.write(json.dumps({"id": id_, "source": s, "raw": p.strip()}, ensure_ascii=False) + "\n")
                print(i + len(batch), flush=True)
    return read(PROPOSALS)


def review_rows(props, vh, mr, al, occ, rules):
    allowed = M._allowed() or {"pairs": [], "suffixes": []}
    pairs, suff = {tuple(p) for p in allowed["pairs"]}, {tuple(p) for p in allowed["suffixes"]}
    c, ex = collections.Counter(), {}
    for r in props:
        if M.guard(r["source"], r["raw"], None) is None or not M.negation_ok(r["source"], r["raw"]):
            continue                                   # the bot would reject these sentences as a whole
        a, b = r["source"].split(), r["raw"].split()
        for op, i1, i2, j1, j2 in M._ops(a, b):
            if op == "replace" and i2 - i1 == j2 - j1:
                for x_, y_ in zip(a[i1:i2], b[j1:j2]):
                    x, y = M._core(x_), M._core(y_)
                    if x != y:
                        c[(x, y)] += 1; ex.setdefault((x, y), r["source"][:90])
    out = []
    for (x, y), n in c.most_common():
        accepted = (x, y) in pairs or (x[:-2] == y[:-2] and (x[-2:], y[-2:]) in suff)
        s = round(difflib.SequenceMatcher(None, x, y).ratio(), 2)
        if accepted:
            tier = "A accepted by the filter already"
        elif rules.get(x) == y:
            tier = "R now a rule"
        elif vh[y] == 0:
            tier = "C invented (Varhadi word not in the real text; 3.5% right on held-out scripture)"
        elif vh[y] >= 2 and s >= 0.6:
            tier = "B attested and similar (about 31% right on held-out scripture)"
        else:
            tier = "B- attested but different (19% right overall)"
        out.append({"marathi": x, "varhadi": y, "farm_count": n, "tier": tier, "varhadi_text_y": vh[y], "varhadi_text_x": vh[x],
                    "aligned_in_training": al[(x, y)], "spelling_similarity": s, "example_sentence": ex[(x, y)], "native_verdict": ""})
    return out


def merge_reviewed():
    kb = json.loads(KB.read_text(encoding="utf-8")) if KB.exists() else {"rules": {}, "evidence": {}}
    ok = [[r["marathi"], r["varhadi"]] for r in csv.DictReader(open(REVIEW, encoding="utf-8-sig"))
          if r["native_verdict"].strip().lower() in ("yes", "y", "ha", "हो")]
    kb["allowed_extra"] = ok
    KB.write_text(json.dumps(kb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(ok), "native-approved swaps now in allowed_extra")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true"); ap.add_argument("--merge-reviewed", action="store_true")
    a = ap.parse_args()
    if a.merge_reviewed:
        return merge_reviewed()
    vh, mr, al, occ = corpora()
    farm = collections.Counter(M._core(w) for r in csv.DictReader(open(FARM_CSV, encoding="utf-8-sig")) for w in r["marathi"].split())
    rules, ev, dropped = build_rules(vh, mr, al, occ, farm)
    old = json.loads(KB.read_text(encoding="utf-8")) if KB.exists() else {}
    kb = {"status": "UNVERIFIED by a native speaker. Each swap passed the numeric evidence in `evidence` (counts in the real Varhadi text and the "
                    "training pairs) and a developer's meaning check. Varhadi values are attested in the VAHNT text; dative -ले nouns rest on the pattern.",
          "rules": rules, "allowed_extra": old.get("allowed_extra", []), "evidence": ev}
    KB.write_text(json.dumps(kb, ensure_ascii=False, indent=1), encoding="utf-8")
    rows = review_rows(proposals(a.regenerate), vh, mr, al, occ, rules)
    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEW, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"rules kept {len(rules)} | dropped {len(dropped)}")
    for x, y, why in dropped:
        print(f"  dropped {x} -> {y}: {why}")
    print("review rows", len(rows), dict(collections.Counter(r["tier"].split()[0] for r in rows)))
    # how much of the farm sentences the rules touch
    n = sum(farm[x] for x in rules); tot = sum(farm.values())
    print(f"rules match {n} of {tot} Marathi words in the 573 farm sentences ({n / tot * 100:.1f}%)")


if __name__ == "__main__":
    main()
