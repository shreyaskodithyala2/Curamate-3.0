"""RAG with ChromaDB (HNSW index, cosine similarity) and Nomic embeddings from Ollama.

Use 1 - symptom matching: doctor's free text -> phrases -> nearest of the 209 DDXPlus questions.
Use 2 - report retrieval: diagnosed disease -> most relevant knowledge-base chunks (metadata filter + cosine).

Two Chroma collections (built once by scripts/build_kb.py, stored in artifacts/chroma):
  symptoms       674 tiny chunks: each DDXPlus question + each everyday phrasing ("can't breathe")
  disease_cards  294 chunks: 49 diseases x 6 sections (overview, danger signs, tests, treatment, tell-apart, typical features)
Chroma runs inside this Python process: SQLite stores text + labels and applies filters, HNSW finds the closest vectors.
"""
import json
import re

import chromadb
from chromadb.config import Settings

import config

SYMPTOMS, CARDS = "symptoms", "disease_cards"
SECTION_TITLES = {
    "overview": "Overview", "typical_features": "Typical features (DDXPlus)", "red_flags": "Danger signs",
    "next_steps": "Assessment and investigations", "management": "Initial management",
    "differentiate": "How to tell it apart",
}
# Nomic is trained with task prefixes: documents and queries are embedded slightly differently.
DOC_PREFIX, QUERY_PREFIX = "search_document: ", "search_query: "


def chroma_client():
    """Opens ChromaDB embedded in this process, saved to artifacts/chroma, with telemetry off (fully offline)."""
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR), settings=Settings(anonymized_telemetry=False))


def load_knowledge():
    """Reads knowledge/symptom_labels.json (everyday phrases) and knowledge/disease_cards.json (disease sections)."""
    labels = json.load(open(config.KNOWLEDGE_DIR / "symptom_labels.json", encoding="utf-8"))
    cards = json.load(open(config.KNOWLEDGE_DIR / "disease_cards.json", encoding="utf-8"))
    return labels, cards


# ---------- splitting the doctor's text into symptom phrases ----------
_SPLIT = re.compile(r"[,;\n]+|\.(?:\s|$)|\s+(?:and|but|also|plus)\s+", re.I)              # where to cut the text
_DEMOGRAPHIC = re.compile(r"^\d{1,3}\s*(?:-?\s*(?:years?|yrs?|yo|y/o|y))?(?:\s*-?\s*old)?\s*"   # "26F", "34 yo male"
                          r"(?:m|f|male|female|man|woman|boy|girl)?\b[\s,:-]*", re.I)
_FILLER = re.compile(r"^(?:(?:the\s+)?(?:patient|pt)\s+(?:has|have|is|reports|complains of|c/o)?|c/o|complains of|"
                     r"reports|presents with|presenting with|he has|she has|they have|has|having|have|i have|"
                     r"feels|feeling|with|male|female)\s+", re.I)                             # "pt c/o", "patient has"
_NEGATION = re.compile(r"^(?:no history of|no hx of|no|not|denies|denied|without|never|negative for|nil|"
                       r"absence of|does not have|doesn't have)\b\s*", re.I)                  # marks a phrase as NO
# Intensity words dominate short phrases ("mild fever" -> "mild lightheadedness"), so they are removed
# before embedding. Pain character and intensity are captured later by the follow-up detail form.
_MODIFIERS = re.compile(r"\b(?:mild|mildly|moderate|severe|severely|slight|slightly|bad|terrible|very|really|quite|"
                        r"some|a bit of|a little|sharp|dull|constant|occasional|intermittent|persistent|sudden|suddenly|acute)\b\s*", re.I)


def split_phrases(text):
    """'sharp chest pain, short of breath, no fever or cough' ->
       [('sharp chest pain', False), ('short of breath', False), ('fever', True), ('cough', True)]

    Input: the doctor's free text. Output: list of (phrase, is_it_a_NO)."""
    out, seen = [], set()
    for part in _SPLIT.split(text or ""):
        part = _DEMOGRAPHIC.sub("", (part or "").strip())
        for _ in range(2):
            part = _FILLER.sub("", part).strip()
        negated = bool(_NEGATION.match(part))
        pieces = re.split(r"\s+(?:or|nor)\s+|/", _NEGATION.sub("", part)) if negated else [part]   # "no fever or cough" -> 2 NOs
        for p in pieces:
            p = p.strip(" .-")
            if len(p) >= 3 and p.lower() not in seen:
                seen.add(p.lower())
                out.append((p, negated))
    return out


class KnowledgeBase:
    def __init__(self, ollama):
        """Opens the two Chroma collections built by scripts/build_kb.py."""
        self.ollama = ollama
        client = chroma_client()
        self.symptoms = client.get_collection(SYMPTOMS)
        self.cards = client.get_collection(CARDS)

    def counts(self):
        return {"symptom_docs": self.symptoms.count(), "card_chunks": self.cards.count()}

    # ---------- use 1: symptom matching ----------
    def match(self, text):
        """RAG search 1: the doctor's words -> DDXPlus question codes (called by POST /api/match).

        Input:  "26F, wheezing, known peanut allergy, mild swelling of lips, no fever"
        Output: one entry per phrase, e.g.
          {"phrase": "swelling of lips", "negated": False, "selected": "E_151",
           "options": [{"code": "E_151", "score": 0.817, ...}, {"code": "E_9", "score": 0.629, ...}, ...]}
        "selected" is pre-ticked only when similarity >= 0.65; the doctor confirms every match.
        """
        phrases = split_phrases(text)
        if not phrases:
            return []
        vecs = self.ollama.embed([QUERY_PREFIX + (_MODIFIERS.sub("", p).strip() or p) for p, _ in phrases])   # Ollama /api/embed
        # several documents per question (question text + plain-language labels) -> fetch extra, keep best per code
        res = self.symptoms.query(query_embeddings=vecs.tolist(), n_results=config.MATCH_TOP_K * 6,          # Chroma: 18 nearest
                                  include=["metadatas", "distances"])
        results = []
        for (phrase, negated), metas, dists in zip(phrases, res["metadatas"], res["distances"]):
            best = {}
            for m, d in zip(metas, dists):
                score = 1.0 - d                                     # cosine distance -> similarity
                if score > best.get(m["code"], {"score": -1})["score"]:
                    best[m["code"]] = {"code": m["code"], "question": m["question"], "score": round(score, 3),
                                       "matched_on": m["text"]}
            options = sorted(best.values(), key=lambda o: -o["score"])[:config.MATCH_TOP_K]
            confident = bool(options) and options[0]["score"] >= config.MATCH_MIN_SCORE
            results.append({"phrase": phrase, "negated": negated, "options": options,
                            "selected": options[0]["code"] if confident else None})
        return results

    # ---------- use 2: report retrieval ----------
    def report_sources(self, diagnosis, alternatives, findings):
        """Top-k chunks of the diagnosed disease ranked by cosine similarity to the patient's picture,
        plus the 'how to tell apart' chunk of each runner-up.

        Input:  "Anaphylaxis", ["Bronchospasm / acute asthma exacerbation", "Scombroid food poisoning"], patient's YES findings
        Output: (query text, sources) where sources = [{"id": "S1", "disease": ..., "section": "red_flags",
                 "body": "Stridor, hoarse voice ...", "score": 0.765, ...}, ...]  -> S1..S8, cited by the LLM
        """
        query = (f"{diagnosis}: danger signs, investigations and initial management "
                 f"for a patient with {', '.join(findings[:8]) or 'these symptoms'}")
        qv = self.ollama.embed([QUERY_PREFIX + query])[0].tolist()
        sources = []
        # where={...}: SQLite keeps only this disease's 6 chunks, then HNSW ranks them by cosine similarity
        main = self.cards.query(query_embeddings=[qv], n_results=config.REPORT_TOP_K, where={"disease": diagnosis},
                                include=["documents", "metadatas", "distances"])
        for doc, m, d in zip(main["documents"][0], main["metadatas"][0], main["distances"][0]):
            sources.append({**m, "text": doc, "score": round(1 - d, 3)})
        for alt in alternatives:
            q = self.ollama.embed([QUERY_PREFIX + f"how to tell {alt} apart from {diagnosis}"])[0].tolist()
            r = self.cards.query(query_embeddings=[q], n_results=config.ALT_TOP_K,
                                 where={"$and": [{"disease": alt}, {"section": "differentiate"}]},   # 2 filters
                                 include=["documents", "metadatas", "distances"])
            for doc, m, d in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
                sources.append({**m, "text": doc, "score": round(1 - d, 3)})
        for i, s in enumerate(sources, 1):
            s["id"] = f"S{i}"                                       # citation tags for the LLM
        return query, sources
