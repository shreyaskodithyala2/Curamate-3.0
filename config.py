"""All Curamate settings in one place.

Every path, model name, threshold and limit lives here, so behaviour can be tuned without touching the logic.
Read by: training (scripts/), the engine (core/engine.py), RAG (core/rag.py), the LLM (core/llm.py) and app.py.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("CURAMATE_DATA", ROOT / "data" / "ddxplus"))   # unpack with scripts/unpack_data.sh
KNOWLEDGE_DIR = ROOT / "knowledge"           # symptom_labels.json + disease_cards.json (RAG source text)
ARTIFACTS = ROOT / "artifacts"               # everything the scripts generate
MODEL_FILE = ARTIFACTS / "lgbm_model.txt"    # trained LightGBM model (5,390 trees)
FREQ_FILE = ARTIFACTS / "answer_freq.json"   # P(yes | disease) table used to pick questions
CHROMA_DIR = ARTIFACTS / "chroma"            # ChromaDB vector database folder

# ---- Local services (Ollama runs on this machine; nothing leaves it) ----
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = "nomic-embed-text"
# Qwen3.5-4B quantized by us to 4-bit (Q4_K_M) with llama.cpp - see scripts/quantize_llm.sh
LLM_MODEL = os.environ.get("CURAMATE_LLM", "curamate-qwen3.5-4b:q4_k_m")
LLM_FALLBACKS = ["qwen3.5:4b", "qwen3.5:2b"]   # only 4B/2B; if none is installed the report uses a template
LLM_OPTIONS = {"temperature": 0.2, "num_ctx": 4096, "num_predict": 500}   # low randomness, 4k context, max answer length
LLM_KEEP_ALIVE = "2m"          # unload the LLM from memory 2 minutes after the last report

# ---- Training ----
TRAIN_FRACTION = 0.15          # share of train.csv patients used (keeps RAM use around 3 GB)
INTERVIEW_COPIES = 2           # interview-style partial copies per patient
TRAIN_THREADS = max(2, (os.cpu_count() or 4) - 2)   # leave cores free so the Mac stays responsive

# ---- RAG use 1: symptom matching ----
MATCH_TOP_K = 3                # candidate symptoms shown per phrase
MATCH_MIN_SCORE = 0.65         # cosine similarity needed to pre-select the best match

# ---- Interview (stopping rule) ----
TOP_SUSPECTS = 5               # follow-up questions come from these diseases' question lists
STOP_CONFIDENCE = 0.90         # stop only when the top disease is at least 90%...
MIN_QUESTIONS = 3              # ...at least 3 follow-up questions were asked...
EMERGENCY_THRESHOLD = 0.01     # ...and no other severity 1-2 disease is above 1%
MAX_QUESTIONS = 15             # hard limit

# ---- RAG use 2: report retrieval ----
REPORT_TOP_K = 6               # chunks for the main diagnosis: each card has 6 sections and the report needs
                               # all of them (k=5 once dropped "Initial management" for anaphylaxis)
ALT_TOP_K = 1                  # "how to tell apart" chunk per runner-up disease
ALT_MIN_PROB = 0.01            # only runner-ups at least 1% likely get a "how to tell apart" chunk

HOST, PORT = "127.0.0.1", 5050  # local only
