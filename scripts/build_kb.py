"""Build the two Chroma collections (ingestion pipeline).

  symptoms       - every askable DDXPlus question + plain-language phrases  (RAG use 1: symptom matching)
  disease_cards  - 6 sections per disease: 5 synthesized clinical sections + 1 generated from the dataset
                   (RAG use 2: report retrieval)

Run from the curamate folder:  .venv/bin/python -m scripts.build_kb
Needs Ollama running with `nomic-embed-text`.

Input:  knowledge/symptom_labels.json, knowledge/disease_cards.json + the two DDXPlus JSON dictionaries
Steps:  chunk the text -> embed every chunk with Nomic (Ollama POST /api/embed) -> store in ChromaDB
Output: artifacts/chroma/ with 674 symptom chunks and 294 disease chunks (~10 seconds)
"""
import time

import config
from core.features import Features
from core.ollama import Ollama
from core.rag import CARDS, DOC_PREFIX, SECTION_TITLES, SYMPTOMS, chroma_client, load_knowledge

CARD_SECTIONS = ["overview", "red_flags", "next_steps", "management", "differentiate"]
SKIP_MATCHING = {"E_204"}   # travel is a dedicated field on the intake form


def symptom_docs(f, labels):
    """Chunks for the `symptoms` collection: one chunk = one way of saying a symptom.

    Output: ids, texts, labels for 674 chunks, e.g. for shortness of breath (E_66):
      E_66::0 "Are you experiencing shortness of breath ...?"  E_66::5 "can't breathe"  -> all labelled code = E_66
    Short phrases get their own chunk so a doctor's short phrase matches them directly.
    """
    ids, texts, metas = [], [], []
    for code in f.roots:
        if code in SKIP_MATCHING:
            continue
        question = f.question(code)
        for i, text in enumerate([question] + labels.get(code, [])):
            ids.append(f"{code}::{i}")
            texts.append(text)
            metas.append({"code": code, "question": question, "text": text,     # "code" maps a match back to the question
                          "kind": "question" if i == 0 else "label",
                          "history": bool(f.ev[code]["is_antecedent"])})
    return ids, texts, metas


def typical_features_text(f, labels, disease, card):
    """Section generated straight from release_conditions.json: what the dataset links to this disease."""
    name = lambda k: labels.get(f.root_of[k], [f.question(f.root_of[k])])[0]
    cond = f.cond[disease]
    symptoms = sorted({name(k) for k in cond["symptoms"]})
    history = sorted({name(k) for k in cond["antecedents"] if f.root_of[k] not in SKIP_MATCHING})
    return (f"In the DDXPlus clinical knowledge base, {card['display']} (ICD-10 {cond['icd10-id'].upper()}, "
            f"severity {cond['severity']} on a scale where 1 is most severe) is associated with these symptoms: "
            f"{', '.join(symptoms)}. Relevant history and risk factors: {', '.join(history) or 'none listed'}.")


def card_docs(f, labels, cards):
    """Chunks for the `disease_cards` collection: one chunk = one section of one disease (49 x 6 = 294).

    Example chunk: id "d07-red_flags", text "Anaphylaxis - Danger signs: Stridor, hoarse voice ...",
                   labels {"disease": "Anaphylaxis", "section": "red_flags", ...}
    The disease + section labels are what report retrieval filters on.
    """
    missing = set(f.classes) - set(cards)
    if missing:
        raise SystemExit(f"disease_cards.json is missing: {sorted(missing)}")
    ids, texts, metas = [], [], []
    for n, disease in enumerate(f.classes):
        card = cards[disease]
        sections = {s: card[s] for s in CARD_SECTIONS}
        sections["typical_features"] = typical_features_text(f, labels, disease, card)
        for section, body in sections.items():
            ids.append(f"d{n:02d}-{section}")
            texts.append(f"{card['display']} - {SECTION_TITLES[section]}: {body}")   # disease name in front -> clearer embedding
            metas.append({"disease": disease, "display": card["display"], "section": section,
                          "section_title": SECTION_TITLES[section], "body": body,
                          "severity": int(f.cond[disease]["severity"]), "icd10": f.cond[disease]["icd10-id"].upper(),
                          "references": "; ".join(card["references"]) if section != "typical_features"
                          else "DDXPlus release_conditions.json (Tchango et al., 2022)",
                          "origin": "dataset" if section == "typical_features" else "synthesized-unreviewed"})
    return ids, texts, metas


def ingest(client, ollama, name, ids, texts, metas):
    """Embeds every chunk and stores it in a fresh Chroma collection (id + text + 768-number vector + labels)."""
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)                    # rebuild from scratch on every run
    # HNSW index with cosine distance; we always pass our own Nomic vectors
    col = client.create_collection(name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None)
    t = time.time()
    vecs = ollama.embed([DOC_PREFIX + t_ for t_ in texts])           # Ollama POST /api/embed (Nomic)
    col.add(ids=ids, embeddings=vecs.tolist(), documents=texts, metadatas=metas)   # SQLite: text + labels; HNSW: vectors
    print(f"  {name:<14} {col.count():>4} documents  ({vecs.shape[1]}-dim, {time.time() - t:.1f}s)")


def main():
    f = Features(config.DATA_DIR)
    labels, cards = load_knowledge()
    missing = [r for r in f.roots if r not in labels and r not in SKIP_MATCHING]
    if missing:
        print(f"note: no plain-language labels for {missing}; question text only")
    ollama = Ollama()
    if not any(m.startswith(config.EMBED_MODEL) for m in ollama.models()):
        raise SystemExit(f"Ollama model '{config.EMBED_MODEL}' not found - run: ollama pull {config.EMBED_MODEL}")
    client = chroma_client()
    print("Embedding with Nomic via local Ollama and storing in Chroma:")
    ingest(client, ollama, SYMPTOMS, *symptom_docs(f, labels))
    ingest(client, ollama, CARDS, *card_docs(f, labels, cards))
    print(f"saved to {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
