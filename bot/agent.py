"""Agentic RAG answer with Gemini: the model decides what to search for, reads the retrieved passages and writes the answer.

Flow (bot/server.py then converts the Marathi answer to Varhadi): farmer question -> Gemini calls search_knowledge(query, crop)
one to MAX_SEARCHES times (it rewrites the question in English, tries again with other words if the first passages do not answer it)
-> answer in simple Marathi using ONLY the passages -> last line `SOURCES: 2,3` names the passages used -> we strip it and return
those passages as citations. Needs GEMINI_API_KEY (.env); without it the server keeps using the Groq chat without documents.
"""
import os
import re

from bot import rag

# On 2026-11-13 gemini-3.8/3.7-flash answered 503 "high demand" all day, 3.6/3.5-flash worked, 2.5-* were closed to new keys.
# Models are tried in this order when one is overloaded (503) or retired (404). Override the first with KD_GEMINI_MODEL.
MODELS = [os.environ.get("KD_GEMINI_MODEL", "gemini-3.6-flash"), "gemini-3.5-flash", "gemini-3.1-flash-lite"]
MAX_SEARCHES = 4
TOP_K = 5
EXCERPT = 900  # characters of each passage shown to the model

RAG_RULES = """
KNOWLEDGE BASE
- For every farming question, FIRST call search_knowledge. Write the query in English with the crop, the problem and the stage (the documents are in English). If the passages do not answer the question, search again with different words (at most {n} searches in total).
- Answer ONLY with facts found in the passages. Do not add facts from memory. If the passages do not contain the answer, say that plainly and send the farmer to the कृषी सहाय्यक / कृषी विज्ञान केंद्र / PDKV Akola.
- Turn what you found into simple spoken Marathi. Sowing time, seed rate, spacing, variety names and fertiliser schedules may be given when a passage states them. Never name a pesticide, fungicide or herbicide product or give its dose or mixing ratio, even if a passage does: say what to check and send the farmer to the कृषी सहाय्यक for the product and dose.
- Greetings and thanks need no search.
- Finish with one last line exactly like `SOURCES: 1,3` listing the numbers of the passages you used, or `SOURCES: none`. Put no numbers or brackets in the sentences themselves.
""".format(n=MAX_SEARCHES)


def enabled() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY")) and os.environ.get("KD_AGENT", "1") != "0" and rag.available()


def make_client():
    from google import genai  # imported here so the bot starts without the package when the agent is off
    from google.genai import types
    # new models answer 503 "high demand" in spikes: retry a few times with backoff before giving up (not 429: a daily quota does not come back in seconds)
    retry = types.HttpRetryOptions(attempts=5, initial_delay=2.0, max_delay=20.0, http_status_codes=[500, 503])
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(retry_options=retry))


def generate(client, contents, config, on_retry=lambda: None):
    """client.models.generate_content on the first model that is not overloaded (503), retired (404) or out of quota (429).
    The free tier of this key allows only 20 requests per day per model (seen 2026-11-13), so the fallback list matters."""
    for i, model in enumerate(MODELS):
        on_retry()
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:
            if getattr(e, "code", None) not in (503, 404, 429) or i == len(MODELS) - 1:
                raise


def split_sources(text: str, seen: list[dict]) -> tuple[str, list[dict]]:
    """Strip the trailing `SOURCES: 1,3` line; return (answer, the passages it names)."""
    m = re.search(r"\n?\s*SOURCES:\s*([\d,\s]+|none)\s*$", text.strip(), re.I)
    if not m:
        return text.strip(), []
    ids = {int(n) for n in re.findall(r"\d+", m.group(1))}
    return text.strip()[:m.start()].strip(), [s for i, s in enumerate(seen, 1) if i in ids]


def answer(messages: list[dict], system: str, client=None, search=rag.search) -> tuple[str, list[dict]]:
    """messages = [{"role": "user"|"assistant", "content": str}, ...] ending with the farmer's question.
    Returns (Marathi answer, cited passages [{title, publisher, url, page}])."""
    from google.genai import types

    client = client or make_client()
    seen: list[dict] = []

    def search_knowledge(query: str, crop: str = "") -> str:
        """Search the farm knowledge base (ICAR, PDKV Akola and other public agriculture documents) and return numbered passages.

        Args:
            query: English search words, for example "pigeonpea wilt management" or "cotton sowing time Vidarbha".
            crop: optional, one of cotton, soybean, tur, orange. Passages about this crop rank first.
        """
        hits = search(query, crop if crop in ("cotton", "soybean", "tur", "orange") else "", TOP_K)
        if not hits:
            return "No passages found. Try different words."
        out = []
        for h in hits:
            seen.append(h)
            out.append(f"[{len(seen)}] {h['title']} ({h['publisher']}, p.{h['page']}): {h['text'][:EXCERPT]}")
        return "\n\n".join(out)

    contents = [types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part(text=m["content"])]) for m in messages]
    cfg = types.GenerateContentConfig(
        system_instruction=system + RAG_RULES, tools=[search_knowledge], temperature=0.2, max_output_tokens=2048,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(maximum_remote_calls=MAX_SEARCHES))
    r = generate(client, contents, cfg, on_retry=seen.clear)  # a failed attempt may already have searched
    text, cited = split_sources(r.text or "", seen)
    return text, [{k: c[k] for k in ("title", "publisher", "url", "page")} for c in cited]
