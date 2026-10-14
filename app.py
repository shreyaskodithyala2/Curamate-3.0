"""Curamate web app. Run from the curamate folder:  .venv/bin/python app.py  ->  http://127.0.0.1:5050

The browser keeps the interview state (answers so far) and sends it with every request, so the
server is stateless: each call rebuilds the feature vector and asks the engine what to do next.

Web API (browser <-> this Flask server, JSON over HTTP on 127.0.0.1):
  GET  /             the web page
  GET  /api/health   status of model, knowledge base, Ollama, embeddings, LLM
  GET  /api/meta     travel regions for the intake form
  POST /api/match    RAG search 1: free text -> matched symptom codes
  POST /api/step     one interview round: next question / detail form / done
  POST /api/report   final diagnosis + RAG search 2 + LLM report + citation check
LightGBM and ChromaDB run inside this process; Ollama (Nomic + Qwen) is called over localhost HTTP.
"""
import time

import numpy as np
from flask import Flask, jsonify, render_template, request

import config
from core.engine import Engine, entropy
from core.features import CHILD_ORDER, SCALE_HINTS, Features
from core.llm import SEVERITY_LABELS, check_citations, pick_model, write_report
from core.ollama import Ollama
from core.rag import KnowledgeBase, load_knowledge

app = Flask(__name__)
F = Features(config.DATA_DIR)
LABELS, CARDS = load_knowledge()
OLLAMA = Ollama()
DETAIL_TITLES = {"E_53": "Pain details", "E_129": "Skin lesion details", "E_151": "Swelling details"}
_services = {}


def service(name):
    """Load the model / knowledge base on first use, so /api/health can report what is missing."""
    if name not in _services:
        _services[name] = Engine(F) if name == "engine" else KnowledgeBase(OLLAMA)
    return _services[name]


def label(code):
    """Short readable name for a question code, e.g. "E_66" -> "shortness of breath"."""
    return LABELS.get(code, [F.question(code)])[0]


def display(disease):
    """Display name for a disease, e.g. "URTI" -> "Upper respiratory tract infection (common cold)"."""
    return CARDS[disease]["display"]


def bad_request(msg):
    return jsonify({"error": msg}), 400


# ---------- turning the browser's answers into model input ----------
def child_tokens(child, value):
    """One detail-form answer -> dataset tokens, e.g. ("E_56", 7) -> ["E_56_@_7"], ("E_55", ["V_89"]) -> ["E_55_@_V_89"]."""
    e = F.ev[child]
    if child in F.scale:
        v = int(value)
        return [f"{child}_@_{v}"] if v != 0 else []
    values = value if isinstance(value, list) else [value]
    allowed = {str(v) for v in e["possible-values"]}
    return [f"{child}_@_{v}" for v in values if str(v) in allowed and str(v) != str(e["default_value"])]


def build_state(payload):
    """answers: {code: yes|no|unknown}, details: {root: {child: value|'unknown'}}, travel: unknown|no|V_x.

    Input:  the browser's state, e.g. {"answers": {"E_53": "yes", "E_91": "no", "E_12": "unknown"}, "details": {...}}
    Output: (state, excluded) - state in the format features.encode() expects
            ({"E_53": (tokens, known), "E_91": False}); excluded = "don't know" questions (never asked again)
    """
    answers, details = payload.get("answers", {}), payload.get("details", {})
    state, excluded = {}, set()
    for code, a in answers.items():
        if code not in F.children:
            continue
        if a == "no":
            state[code] = False
        elif a == "yes":
            tokens, known = [code], set()
            for child, value in (details.get(code) or {}).items():
                if child in F.children[code] and value not in ("unknown", None):
                    known.add(child)
                    tokens += child_tokens(child, value)
            state[code] = (tokens, known)
        else:
            excluded.add(code)
    travel = payload.get("travel", "unknown")
    if travel == "no":
        state["E_204"] = False
    elif travel in F.ev["E_204"]["possible-values"] and travel != "V_10":
        state["E_204"] = ([f"E_204_@_{travel}"], None)
    return state, excluded


def describe(code, val):
    """Readable answer, e.g. 'pain (where: forehead, temple(L); character: heavy; intensity 6/10)'."""
    tokens, _ = val
    parts = {}
    for t in tokens:
        b, _, v = t.partition("_@_")
        if b == code or not v:
            continue
        short = {"E_55": "where", "E_57": "spreads to", "E_54": "character", "E_56": "intensity", "E_58": "precision",
                 "E_59": "onset speed", "E_133": "where", "E_130": "colour", "E_136": "itch", "E_134": "pain",
                 "E_132": "swelling", "E_135": "larger than 1cm", "E_131": "peeling", "E_152": "where"}.get(b, b)
        parts.setdefault(short, []).append(f"{v}/10" if b in F.scale else F.value_label(b, v))
    detail = "; ".join(f"{k}: {', '.join(v)}" for k, v in parts.items())
    return label(code) + (f" ({detail})" if detail else "")


def patient_lists(state):
    """State -> readable "Patient HAS" / "Patient DENIES" lists for the LLM prompt and the report page."""
    yes, no = [], []
    for code, val in state.items():
        if code == "E_204":
            (yes if val else no).append(f"travelled to {F.value_label('E_204', val[0][0].split('_@_')[1])} in the last 4 weeks"
                                        if val else "no travel abroad in the last 4 weeks")
        elif val is False:
            no.append(label(code))
        else:
            yes.append(describe(code, val))
    return yes, no


def detail_form(root):
    """Builds the follow-up form shown after a YES to pain / skin lesions / swelling
    (location checkboxes, 0-10 sliders, single choices)."""
    fields = []
    for child in CHILD_ORDER.get(root, F.children[root]):
        e = F.ev[child]
        if child in F.scale:
            fields.append({"code": child, "label": e["question_en"], "type": "scale",
                           "hints": list(SCALE_HINTS.get(child, ("0", "10")))})
            continue
        opts = [{"value": str(v), "label": F.value_label(child, v)} for v in e["possible-values"]]
        if e["data_type"] == "M":
            opts = sorted([o for o in opts if o["value"] != str(e["default_value"])], key=lambda o: o["label"].lower())
            fields.append({"code": child, "label": e["question_en"], "type": "multi", "options": opts})
        else:
            fields.append({"code": child, "label": e["question_en"], "type": "choice", "options": opts})
    return {"root": root, "title": DETAIL_TITLES.get(root, "Details"), "fields": fields}


def differential(p, n=8):
    """Top n diseases with display names and severity labels, for the live differential panel."""
    out = service("engine").differential(p, n)
    for d in out:
        d["display"] = display(d["disease"])
        d["severity_label"] = SEVERITY_LABELS[d["severity"]]
    return out


def predict(payload):
    """Browser state -> LightGBM prediction (shared by /api/step and /api/report).
    Output: (age, sex, state, excluded, x = 905 numbers, p = 49 percentages), or None if age/sex are invalid."""
    try:
        age, sex = int(payload["age"]), payload["sex"]
    except (KeyError, TypeError, ValueError):
        return None
    if not 0 <= age <= 120 or sex not in ("M", "F"):
        return None
    state, excluded = build_state(payload)
    x = F.encode(age, sex, state)
    return age, sex, state, excluded, x, service("engine").predict(x)[0]


# ---------- routes ----------
@app.get("/")
def index():
    """API: GET / -> the single-page web interface (templates/index.html)."""
    return render_template("index.html")


@app.get("/api/meta")
def meta():
    """API: GET /api/meta -> {"regions": [{"value": "V_1", "label": "West Africa"}, ...], "max_questions": 15, ...}
    Fills the travel dropdown on the intake form."""
    regions = [{"value": v, "label": F.value_label("E_204", v)} for v in F.ev["E_204"]["possible-values"] if v != "V_10"]
    return jsonify({"regions": regions, "max_questions": config.MAX_QUESTIONS,
                    "stop_confidence": config.STOP_CONFIDENCE, "min_questions": config.MIN_QUESTIONS})


@app.get("/api/health")
def health():
    """API: GET /api/health -> {"model": {"ok": true, "detail": "LightGBM: 5,390 trees, ..."}, "kb": {...},
    "ollama": {...}, "embed": {...}, "llm": {...}}  - the status pills at the top of the page.
    First call also loads LightGBM and opens ChromaDB; calls Ollama GET /api/tags."""
    out = {}
    try:
        e = service("engine")
        out["model"] = {"ok": True, "detail": f"LightGBM: {e.booster.num_trees():,} trees, {F.n_features} features"}
    except Exception as ex:
        out["model"] = {"ok": False, "detail": f"Not trained yet ({ex.__class__.__name__}) - run scripts.train_model"}
    try:
        models = OLLAMA.models()
        out["ollama"] = {"ok": True, "detail": "Ollama running locally"}
        out["embed"] = {"ok": any(m.startswith(config.EMBED_MODEL) for m in models), "detail": config.EMBED_MODEL}
        llm = pick_model(OLLAMA)
        out["llm"] = {"ok": llm is not None, "detail": llm or "no Qwen model - template reports"}
    except Exception:
        out["ollama"] = {"ok": False, "detail": "Ollama not reachable - start the Ollama app"}
    try:
        c = service("kb").counts()
        out["kb"] = {"ok": True, "detail": f"Chroma: {c['symptom_docs']} symptom docs, {c['card_chunks']} card chunks"}
    except Exception as ex:
        out["kb"] = {"ok": False, "detail": f"Knowledge base missing ({ex.__class__.__name__}) - run scripts.build_kb"}
    return jsonify(out)


@app.post("/api/match")
def match():
    """API: POST /api/match - RAG search 1 (symptom matching).
    Request:  {"text": "sudden sharp chest pain, short of breath, no fever"}
    Response: {"results": [{"phrase": "chest pain", "negated": false, "selected": "E_53",
                            "options": [{"code": "E_53", "score": 0.82, "label": ...}, ...]}, ...], "ms": 84}
    Calls Ollama POST /api/embed (Nomic) and queries the Chroma `symptoms` collection."""
    text =(request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return bad_request("Describe at least one symptom.")
    t = time.time()
    results = service("kb").match(text)
    for r in results:
        for o in r["options"]:
            o["label"] = label(o["code"])
            o["history"] = bool(F.ev[o["code"]]["is_antecedent"])
    return jsonify({"results": results, "ms": round((time.time() - t) * 1000), "top_k": config.MATCH_TOP_K,
                    "threshold": config.MATCH_MIN_SCORE})


@app.post("/api/step")
def step():
    """API: POST /api/step - one round of the interview (called after every answer).
    Request:  the whole browser state {"age": 52, "sex": "F", "travel": "no", "answers": {"E_53": "yes", ...},
              "details": {...}, "asked": [...], "details_done": [...], "finish": false}
    Response: {"status": "question" | "details" | "done", "question": {...best question...},
               "differential": [...], "uncertainty": 1.9, "reason": "Top diagnosis is 55% - need 90%."}
    Order: predict -> show a detail form if needed -> stop check -> otherwise pick the best question."""
    payload =request.get_json(silent=True) or {}
    pred = predict(payload)
    if pred is None:
        return bad_request("Age (0-120) and sex are required.")
    age, sex, state, excluded, x, p = pred
    if not any(v is not False for k, v in state.items() if k != "E_204"):
        return bad_request("Confirm at least one symptom the patient HAS before starting.")
    eng = service("engine")
    n_asked = len(payload.get("asked", []))
    h = float(entropy(p))
    out = {"differential": differential(p), "uncertainty": round(h, 2), "suspects": round(2 ** h, 1),
           "n_asked": n_asked, "max_questions": config.MAX_QUESTIONS}

    done_details = set(payload.get("details_done", []))
    # a YES to pain / rash / swelling whose detail form hasn't been filled yet -> show that form first
    pending =[c for c, a in payload.get("answers", {}).items()
               if a == "yes" and F.children.get(c) and c not in done_details]
    if pending:
        out.update(status="details", details=detail_form(pending[0]),
                   reason="A few details about this finding help the model.")
        return jsonify(out)

    # stopping rule (core/engine.py stop_check); "Finish now" on the page skips it
    stop, reason = (True, "Finished early by the clinician.") if payload.get("finish") else eng.stop_check(p, n_asked)
    out["reason"] = reason
    if stop:
        out["status"] = "done"
        return jsonify(out)
    t = time.time()
    ranked = eng.rank_questions(age, sex, state, excluded, p)      # what-if yes/no -> highest information gain first
    if not ranked:
        out.update(status="done", reason="No further useful questions to ask.")
        return jsonify(out)
    for q in ranked[:4]:
        q["label"] = label(q["code"])
        for side in ("if_yes", "if_no"):
            q[side]["display"] = display(q[side]["disease"])
    out.update(status="question", question=ranked[0], alternatives=ranked[1:4], n_candidates=len(ranked),
               what_if_ms=round((time.time() - t) * 1000))
    return jsonify(out)


@app.post("/api/report")
def report():
    """API: POST /api/report - the final report (called once, when the interview is done).
    Request:  the whole browser state (same as /api/step)
    Response: {"diagnosis": {"display": "Pulmonary embolism", "prob": 0.992, "severity": 2, "icd10": "I26"},
               "differential": [...], "findings": [...], "sources": [S1..S8], "report": {"text": ...}, "citations": [...]}
    Steps: LightGBM prediction + key findings -> RAG search 2 (Ollama /api/embed + Chroma `disease_cards`)
           -> Qwen writes the cited report (Ollama /api/generate) -> citation check."""
    payload =request.get_json(silent=True) or {}
    pred = predict(payload)
    if pred is None:
        return bad_request("Age (0-120) and sex are required.")
    age, sex, state, excluded, x, p = pred
    eng, kb = service("engine"), service("kb")
    k = int(np.argmax(p))
    disease = F.classes[k]
    diff = differential(p)
    alternatives = [d["disease"] for d in diff[1:3] if d["prob"] >= config.ALT_MIN_PROB]
    yes, no = patient_lists(state)

    findings = []
    for item in eng.key_findings(x, state, k):
        name = {"AGE": f"age {age}", "SEX_F": "female" if sex == "F" else "male"}.get(item["code"]) or label(item["code"])
        findings.append({"label": name, "answer": item["answer"], "effect": item["effect"]})
    drivers = [f"{f['label']} ({f['answer'] or 'demographic'})" for f in findings if f["effect"] > 0]

    t = time.time()
    query, sources = kb.report_sources(disease, alternatives, [y for y in yes])   # RAG search 2 -> S1..S8
    retrieval_ms = round((time.time() - t) * 1000)
    ctx = {"diagnosis": display(disease), "disease_key": disease, "severity": int(F.severity[k]),
           "alternatives": [display(a) for a in alternatives], "yes": yes, "no": no, "drivers": drivers,
           "sources": sources}
    rep = write_report(OLLAMA, ctx)                            # Qwen writes the cited report (or template fallback)
    checks = check_citations(rep["text"], sources, yes, no)    # verify every sentence against its source

    col = eng.freq_col
    unasked = [r for r in F.disease_roots[disease] if r in F.binary_roots and r not in state and r not in excluded]
    unasked.sort(key=lambda r: -eng.freq[k, col[r]])
    p_top = float(p[k])
    return jsonify({
        "diagnosis": {**diff[0], "icd10": F.cond[disease]["icd10-id"].upper()},
        "differential": diff,
        "confidence_note": ("High confidence." if p_top >= config.STOP_CONFIDENCE else
                            "Moderate confidence - review the differential." if p_top >= 0.5 else
                            "Low confidence - the presentation may be outside Curamate's 49 trained conditions."),
        "findings": findings,
        "patient": {"yes": yes, "no": no},
        "not_yet_asked": [{"label": label(r), "freq": round(float(eng.freq[k, col[r]]), 2)} for r in unasked[:8]],
        "retrieval": {"query": query, "top_k": config.REPORT_TOP_K, "ms": retrieval_ms},
        "sources": [{key: s[key] for key in ("id", "display", "section", "section_title", "body", "score",
                                             "references", "origin")} for s in sources],
        "report": rep,
        "citations": checks,
        "questions_asked": len(payload.get("asked", [])),
    })


if __name__ == "__main__":
    print(f"Curamate running at http://{config.HOST}:{config.PORT}  (local only)")
    app.run(host=config.HOST, port=config.PORT, debug=False, threaded=True)
