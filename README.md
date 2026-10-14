# Curamate

Offline clinical decision support: a doctor types symptoms, Curamate runs an adaptive interview
(LightGBM + information gain), then writes a cited summary using a local knowledge base (RAG) and a
small local LLM. Everything runs on device without the need to connect to the internet.

```
doctor's text ──► split into phrases ──► Nomic embeddings ──► Chroma (HNSW, cosine) ──► matched symptoms
                                                                                          │ doctor confirms
                                                                                          ▼
                     ┌──────────────── LightGBM (5,000+ trees, Yes/No/Unknown inputs) ◄───┘
                     │                        │
                     │   not confident? ──► what-if on every candidate question ──► ask the
                     │                      highest information-gain question ──► answer ─┐
                     │                                                                    │
                     │◄───────────────────────────────────────────────────────────────────┘
                     ▼ confident (stopping rule)
     diagnosis + differential ──► Nomic query ──► Chroma: disease-filtered cards, cosine-ranked
                                                        │
                                                        ▼
                               Qwen (Ollama, 4-bit) writes the summary with [S#] citations
                                                        │
                                                        ▼
                                  citation checker ──► dashboard
```

## Run it

Requirements: Python 3.12, [Ollama](https://ollama.com) running locally, ~25 GB free disk
for the one-time LLM build.

```bash
git clone https://github.com/shreyaskodithyala2/Curamate-3.0.git && cd Curamate-3.0
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

bash scripts/unpack_data.sh                # DDXPlus dataset -> data/ddxplus/ (see "Dataset" below)
ollama pull nomic-embed-text               # embedding model for RAG
bash scripts/quantize_llm.sh               # Qwen3.5-4B -> 4-bit Q4_K_M -> Ollama (~20 min, once; files in ../llm_build)

.venv/bin/python -m scripts.train_model    # ~15 min, ~2-3 GB RAM -> artifacts/lgbm_model.txt
.venv/bin/python -m scripts.build_kb       # ~15 s -> artifacts/chroma/
.venv/bin/python -m scripts.evaluate       # optional: test-set metrics -> artifacts/eval.json
.venv/bin/python app.py                    # open http://127.0.0.1:5050
```

After setup nothing needs the internet: Flask binds to 127.0.0.1, Chroma is a local folder, Ollama
serves the embedding model and the LLM on localhost, the web page loads no external assets, and Chroma
telemetry is disabled.

## Dataset

Curamate is trained on **DDXPlus** (Fansi Tchango et al., *DDXPlus: A New Dataset for Automatic Medical
Diagnosis*, NeurIPS 2022 Datasets & Benchmarks; [paper](https://arxiv.org/abs/2205.09148)), released under
**CC-BY 4.0** by its authors for research use - not for clinical decision-making.

Raw CSVs are not committed. The dataset is included as **one zip split into 45 MB parts**
(`data/ddxplus.zip.part-aa` ... `-ad`, 171 MB total) because GitHub rejects files over 100 MB.
`scripts/unpack_data.sh` joins the parts, checks them against `data/ddxplus.zip.sha256`, and unpacks:

```
data/ddxplus/
├── train.csv               1,025,602 patients (640 MB)
├── validate.csv              132,448 patients
├── test.csv                  134,529 patients
├── release_evidences.json    223 questions (symptoms + history) and their answer options
├── release_conditions.json   49 conditions: ICD-10 code, severity, linked symptoms/history
└── README.md                 the dataset's own documentation and licence notice
```

To do it by hand: `cat data/ddxplus.zip.part-* > data/ddxplus.zip && unzip data/ddxplus.zip -d data`
(Windows: `copy /b data\ddxplus.zip.part-* data\ddxplus.zip`, then extract). To keep the dataset elsewhere,
set `CURAMATE_DATA=/path/to/ddxplus`.

Not committed either (all rebuilt by the scripts above): the trained model and Chroma database
(`artifacts/`), and the LLM weights (`../llm_build/`, ~20 GB including the original download).

## Project layout

| Path | What it does |
|---|---|
| `config.py` | Every setting: models, top-k, thresholds, stopping rule |
| `core/features.py` | Answers → 905-number vector (yes = 1, no = 0, unknown = NaN); training-time interview simulation |
| `core/engine.py` | LightGBM prediction, information-gain question selection, stopping rule, key findings |
| `core/rag.py` | Chroma collections: symptom matching (use 1) and report retrieval (use 2) |
| `core/llm.py` | Prompt, local LLM call, citation checker, template fallback |
| `core/ollama.py` | Minimal localhost client for embeddings and generation |
| `scripts/train_model.py` | Builds training data (complete + interview-style records) and trains LightGBM |
| `scripts/build_kb.py` | Ingestion: embeds symptom phrases and disease cards, stores them in Chroma |
| `scripts/evaluate.py` | Accuracy on unseen patients + full interview simulation |
| `scripts/unpack_data.sh` | Joins the split dataset zip, verifies the checksum, unpacks to `data/ddxplus/` |
| `scripts/quantize_llm.sh` | Qwen3.5-4B bf16 → GGUF → 4-bit Q4_K_M with llama.cpp, quality check, import into Ollama |
| `llm/Modelfile` | Registers the quantized GGUF with Ollama as `curamate-qwen3.5-4b:q4_k_m` |
| `knowledge/symptom_labels.json` | Plain-language phrases for each DDXPlus question |
| `knowledge/disease_cards.json` | 49 disease cards (overview, danger signs, investigations, management, differentiation) |
| `app.py`, `templates/`, `static/` | Flask API and the web interface |

## Key settings (config.py)

| Setting | Value | Why |
|---|---|---|
| `MATCH_TOP_K` | 3 | candidate symptoms shown per phrase |
| `MATCH_MIN_SCORE` | 0.65 | cosine similarity needed to pre-select a match; below it the doctor must choose |
| `TOP_SUSPECTS` | 5 | follow-up questions come from these diseases' question lists |
| Stopping rule | ≥ 90 % top diagnosis, ≥ 3 questions, no other emergency above 1 %, max 15 questions | |
| `REPORT_TOP_K` | 6 | all 6 sections of the diagnosed disease's card, ranked by cosine similarity |
| `ALT_TOP_K` | 1 | best "how to tell apart" chunk for each runner-up |

## The LLM: Qwen3.5-4B quantized to 4-bit by us

| | Original (bf16) | Ours (Q4_K_M) |
|---|---|---|
| Size on disk | 8.0 GB | 2.6 GB |
| Bits per weight | 16 | 5.13 average (216 matrices at 4-bit, 33 at 6-bit, 177 small tensors at 32-bit) |
| Perplexity on Curamate's clinical text | 5.50 | 5.75 (+4.4 %) |
| Memory while running (Ollama, 4k context) | - | 4.8 GB |
| Curamate report | - | ~12-17 s at ~18 tokens/s on an M4 |

## Limitations

- The disease cards are **synthesized** for this project to align with the cited guidance (WHO, CDC,
  NICE, ESC, GINA, GOLD, …) and have **not been clinically reviewed**.
- The LLM only explains; the diagnosis and percentages come from LightGBM. Sentences whose citations
  don't check out are highlighted, but the checker is a word-overlap heuristic, not proof.
- Research and education only - not for clinical decision-making.
