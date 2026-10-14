"""Report writing with a small local LLM, plus a citation checker and a no-LLM template fallback.

The LLM never decides anything: the diagnosis, percentages and severity come from LightGBM and are
shown by the UI directly. The LLM only explains them using the retrieved sources.

Flow: write_report() fills PROMPT with the diagnosis, the patient's answers and the retrieved sources (S1..S8)
      -> Qwen 4B (4-bit) via Ollama writes a report where every sentence cites [S#] or [P]
      -> check_citations() verifies each sentence. If no LLM is available, template_report() is used instead.
"""
import re
import time

import config

HEADINGS = ["Why this fits", "Danger signs to check now", "Suggested next steps", "How to tell it apart"]
SEVERITY_LABELS = {1: "Emergency", 2: "Urgent", 3: "Serious", 4: "Moderate", 5: "Mild"}
# Common words ignored when comparing a sentence with its cited source.
STOPWORDS = set("""about above after again against also among because been before being below between both
could does doing during each from further have having here into itself just more most other over same
should some such than that their them then there these they this those through under until very what
when where which while with would your patient patients source sources consider covered usually often
may might can cannot will denies denied reports reported present presents absent known
typically check monitor assess signs""".split())

# The LLM's instructions; {placeholders} are filled per patient in write_report().
PROMPT = """You are the report writer for Curamate, an offline clinical decision-support tool.
A validated classifier has ALREADY chosen the diagnosis. Explain it to a clinician using ONLY the
patient answers [P] and the numbered sources [S#] below.

Rules:
- Do not change or question the diagnosis. Do not write any percentages or probabilities.
- End every sentence with its citation: [S#] for facts from a source, [P] for facts from the patient's answers.
- Use only facts found in the sources or patient answers. If a heading is not covered, write "Not covered by the sources."
- Only findings under "Patient HAS" are present. Findings under "Patient DENIES" are absent - never describe them as present.
- Under 200 words. Short sentences or bullet points. Use exactly these four headings:
### Why this fits
### Danger signs to check now
### Suggested next steps
### How to tell it apart

DIAGNOSIS (from classifier): {diagnosis} - severity {severity}/5 ({severity_label})
OTHER POSSIBILITIES: {alternatives}

PATIENT ANSWERS [P]:
Patient HAS: {yes}
Patient DENIES: {no}

FINDINGS THAT DROVE THE CLASSIFIER: {drivers}

SOURCES:
{sources}
"""


def pick_model(ollama):
    """First installed model from: configured model, then fallbacks. None if Ollama/LLM unavailable.
    Calls Ollama GET /api/tags. Output example: "curamate-qwen3.5-4b:q4_k_m"."""
    try:
        installed = set(ollama.models())
    except Exception:
        return None
    for name in [config.LLM_MODEL] + config.LLM_FALLBACKS:
        if name in installed or f"{name}:latest" in installed:
            return name
    return None


def write_report(ollama, ctx):
    """ctx: diagnosis, severity, alternatives, yes, no, drivers, sources. Returns dict with text + metadata.

    Calls Ollama POST /api/generate (~15-20 s on an M4).
    Output example: {"mode": "llm", "model": "curamate-qwen3.5-4b:q4_k_m", "seconds": 17.4,
                     "text": "### Why this fits\\nThe patient has a known peanut allergy [P] ...", "tokens": 233, "note": ""}
    mode = "template" when no LLM is available or the call fails.
    """
    model = pick_model(ollama)
    if model is None:
        return {"mode": "template", "model": None, "seconds": 0.0, "text": template_report(ctx),
                "note": "No local LLM available - report assembled directly from the retrieved sources."}
    prompt = PROMPT.format(
        diagnosis=ctx["diagnosis"], severity=ctx["severity"], severity_label=SEVERITY_LABELS[ctx["severity"]],
        alternatives=", ".join(ctx["alternatives"]) or "none",
        yes="; ".join(ctx["yes"]) or "none", no="; ".join(ctx["no"]) or "none",
        drivers="; ".join(ctx["drivers"]) or "none",
        # each source becomes "[S1] (Anaphylaxis - Danger signs) Stridor, hoarse voice ..."
        sources="\n".join(f"[{s['id']}] ({s['display']} - {s['section_title']}) {s['body']}" for s in ctx["sources"]),
    )
    t = time.time()
    try:
        res = ollama.generate(model, prompt)
    except Exception as e:
        return {"mode": "template", "model": model, "seconds": round(time.time() - t, 1),
                "text": template_report(ctx), "note": f"LLM call failed ({e.__class__.__name__}) - template used."}
    text = re.sub(r"<think>.*?</think>", "", res.get("response", ""), flags=re.S).strip()   # drop any hidden reasoning
    return {"mode": "llm", "model": model, "seconds": round(time.time() - t, 1), "text": text,
            "tokens": res.get("eval_count"), "note": ""}


def template_report(ctx):
    """Deterministic fallback: the retrieved chunks placed under the same headings."""
    by_section = {}
    for s in ctx["sources"]:
        by_section.setdefault((s["disease"] == ctx["disease_key"], s["section"]), []).append(s)
    pick = lambda main, sec: " ".join(f"{s['body']} [{s['id']}]" for s in by_section.get((main, sec), []))
    why = f"Positive findings: {'; '.join(ctx['yes'][:6]) or 'none'} [P]. " + (pick(True, "overview") or pick(True, "typical_features"))
    return "\n".join([
        "### Why this fits", why or "Not covered by the sources.",
        "### Danger signs to check now", pick(True, "red_flags") or "Not covered by the sources.",
        "### Suggested next steps", (pick(True, "next_steps") + " " + pick(True, "management")).strip() or "Not covered by the sources.",
        "### How to tell it apart", (pick(True, "differentiate") + " " + pick(False, "differentiate")).strip() or "Not covered by the sources.",
    ])


def _words(text):
    """Content words, cut to 4 letters so 'smoker' and 'smoking' both become 'smok'."""
    return {w[:4] for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOPWORDS}


_NEGATED = re.compile(r"\b(no|not|denies|denied|without|absent|absence|negative|lacks?)\b", re.I)
_ABOUT_PATIENT = re.compile(r"\bpatient(?:'s)?\s+(?:has|had|is|was|reports|reported|presents|describes|complains|with)\b"
                            r"|\bpatient's\b", re.I)


def _contradicts(claim, cites, denied):
    """A sentence about the patient that states a DENIED finding as present (e.g. 'the patient is a smoker'
    when the patient answered no to smoking)."""
    if not ("P" in cites or _ABOUT_PATIENT.search(claim)) or _NEGATED.search(claim):
        return False
    words = _words(claim)
    return any(d and d <= words for d in (_words(x) for x in denied))


def check_citations(text, sources, yes, no):
    """For each sentence: does it cite something, does the cited source exist, do its key words actually
    appear in the cited text, and does it contradict the patient's answers?
    Status: ok | no_citation | unknown_source | weak_support | contradicts_patient.

    Output example: [{"sentence": "Give intramuscular adrenaline ... [S5].", "cites": ["S5"], "status": "ok"}, ...]
    It is a word-overlap check (>= 30% of key words found in the cited source), not proof.
    """
    src = {s["id"]: _words(s["body"]) for s in sources}
    src["P"] = _words(" ".join(yes + no))
    checks = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line or line.startswith("#") or line.lower().startswith("not covered"):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line):
            cites = re.findall(r"\[(S\d+|P)\]", sentence.replace("], [", "][").replace(", S", "][S"))
            claim = re.sub(r"\[[^\]]*\]", "", sentence)
            words = _words(claim)
            if len(claim.split()) < 4:
                continue
            if _contradicts(claim, cites, no):
                status = "contradicts_patient"
            elif not cites:
                status = "no_citation"
            elif any(c not in src for c in cites):
                status = "unknown_source"
            else:
                support = set().union(*(src[c] for c in cites))
                status = "ok" if words and len(words & support) / len(words) >= 0.3 else "weak_support"
            checks.append({"sentence": sentence.strip(), "cites": cites, "status": status})
    return checks
