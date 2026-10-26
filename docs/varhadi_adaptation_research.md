# Varhadi adaptation layer: research notes (2026-10-26)

Scope: the Marathi -> Varhadi layer of the bot (`bot/varhadi_style.json`, `varhadify()` in `bot/server.py`) and
source text for a Varhadi translation set. Nothing here is native-speaker verified.

## 1. What exists about Varhadi (and what does not)

| Source | What it gives | Use |
|---|---|---|
| [Wikipedia: Varhadi dialect](https://en.wikipedia.org/wiki/Varhadi_dialect) | dative `-le` (`malā` -> `male`), `jā'un rāhilō`, `abe` | Only `-le` and `hay` are corroborated by our corpus. The article is tagged "needs citations"; do not treat the rest as fact. |
| [Marathi Wikipedia: वऱ्हाडी भाषा](https://mr.wikipedia.org/wiki/वऱ्हाडी_भाषा) | 4 sub-varieties (खडसी, देसी, वरतील्ली, खाल्तील्ली); bullock cart is `बंडी` (देसी) vs `खासर` (खडसी) | Shows variation by sub-region: one Varhadi form per concept is a simplification. No grammar tables. |
| Balbharati class 9 poem "तिफन" (Varhadi farm poem) | `जोळीले`, `सांजीले` (dative -ले), `झोयी` = झोळी, `रगत` = रक्त, `पाळते` = पाडते | Independent confirmation of -ले and the ळ/य sound shift. Source: [exercise page](https://theshaykhacademy.com/textbook-solutions-for-maharashtra-board/standard-nine/chapter-13-tifan/). |
| VAHNT New Testament (already local) | ~80k tokens, 1,296 aligned segments | The only large Varhadi text we have. Scripture register, one translation team. All current rules come from it. |
| [HF `rimrafadii/5gram-mined`](https://huggingface.co/datasets/rimrafadii/5gram-mined) | 305k "Varhadi" sentences | **Do not use as Varhadi data.** ~93% is rule-generated from standard Marathi; the rest is web text. It would teach the model our own rule mistakes back to us. |
| Pune University 1967 thesis on Varhadi around Akola (Shodhganga handle 10603/154333, 483 pp. with a word index) | Best scholarly grammar/lexicon lead | Page is behind a bot check, so I could not open it. **A person should download it** and check the verb endings and the farm words in its index. |
| Sant Gadge Baba Amravati University Varhadi dictionary / idiom / proverb dictionaries (Wagh and Kale) | Print dictionary | Not online as far as I found. Ask the university; best source for farm vocabulary. |
| warhaddhan.in word list, maayboli.com/node/11937 | Word lists | Both unreachable today (DNS failure, 503). Retry later; useful for cross-checking only. |

No Varhadi agriculture corpus exists online (searched Hugging Face, data.gov.in, AIKosh, scholarly sites). Kisan Call
Centre transcripts are on data.gov.in / AIKosh, but I found no Maharashtra Marathi file and could not confirm a
licence. Farm Varhadi text will have to be produced by translation, which is why the sheet below exists.

## 2. Corpus facts that drove the rules

Absence in VAHNT is strong evidence of replacement. Standard word -> count in VAHNT, and the Varhadi form:

- `आहे` 0 -> `हाय` 130 · `नाही` 0 -> `नाई` 423 · `आणि` 0 -> `अन्` 1435 · `मला` 0 -> `मले` 205 · `तुम्ही` 0 -> `तुमी` 264
- `खूप` 10 -> `लय` 101 · `मुलगा` 0 -> `पोरगा` 157 · `लोक` 0 -> `लोकं` 147 · `पुन्हा` 0 -> `परत` 57 · `मात्र`/`परंतु` 0 -> `पण` 678
- `माझा/माझे/माझी` 0 -> `माह्या` 176, `माह्यी` 19, `माह्ये` 12 · `तुझा/तुझे/तुझी` 0 -> `तुह्या` 64
- Plural human oblique takes `-आय-`: `त्यायले` 220, `त्यायनं` 128, `त्यायच्या` 58, `लोकायले` 87, `लोकायच्या` 41, `लोकायसाठी` 12
- Dative `-ले` on nouns: `त्याले` 443, `तुमाले` 234, `येशूले` 89, `माणसाले` 25, `स्वताले` 25
- Agent `-नं`: `त्यानं` 207 vs `त्याने` 7

Caveats: (1) scripture avoids some words for register reasons, so a zero count is evidence only for very common
function words; (2) the ASR side of the mined pairs is not clean standard Marathi, so mined "standard" columns are
hypotheses; (3) one translation team's habits are not a speech community.

## 3. Findings about the layer

- Closed-class swaps (pronouns, copula, particles, a few verbs) are well supported. They can be made deterministic
  in code, which is what `varhadify()` does.
- Rules that need a noun/verb distinction are the risky ones: `-ला` -> `-ले` also hits verbs (`आला`, `झाला`) and adjectives
  (`चांगला`, `पहिला`). The current design uses an allowlist for nouns; the pattern route needs a stoplist and a
  false-positive audit (`scripts/eval_adaptation.py`, check 3).
- Plural oblique (`-ांना` -> `-ायले`, `-ांची` -> `-ायची`, `-ांनी` -> `-ायनं`) is the biggest rule family in the corpus after
  `-ले`, and generalises to farm nouns (`शेतकऱ्यांना` -> `शेतकऱ्यायले`). Evidence is human nouns only, so extending it
  to non-human nouns is a guess for a native speaker to confirm.
- Anything touching farm vocabulary has zero corpus evidence. Words I believe are Vidarbha farm speech but could not
  verify online, to ask a speaker (not used anywhere): `वावर` (field), `पऱ्हाटी` (cotton stalk), `तिफन` (seed drill,
  confirmed only as the title word of the Balbharati poem), `डवरा`, `ढोरं` (cattle), `सोंगणी` (harvest), `निंदण` (weeding).

## 4. Agriculture source text for translation

`scripts/fetch_agri_text.py` pulls 12 Marathi Wikipedia articles (CC BY-SA 4.0, revision ids in
`data/agri_text/provenance.json`) covering cotton, bollworm, leafhopper, soybean, tur, Nagpur orange, drip and sprinkler
irrigation, crop insurance, seed treatment and organic farming: 491 sentences in `data/agri_text/sources.tsv`
(`varhadi` column empty, `verified` false). It is standard Marathi **source** text; the register is encyclopedic, so a
second pass with advisory-style text (PDKV / Krishi Vigyan Kendra leaflets, whose licence must be checked) is still
needed before this resembles what a farmer asks. `data/agri_text/` is gitignored.
