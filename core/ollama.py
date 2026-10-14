"""Tiny client for the local Ollama server (embeddings + text generation). Localhost only.

Ollama runs separately on this machine (http://127.0.0.1:11434) and serves two models:
  - nomic-embed-text             -> turns text into 768-number "meaning fingerprints" (used by RAG)
  - curamate-qwen3.5-4b:q4_k_m   -> our 4-bit Qwen LLM that writes the report

Curamate talks to it with plain HTTP + JSON (no AI library is imported in Python).
API calls made from here:
  GET  /api/tags      -> list installed models
  POST /api/embed     -> texts in, embeddings out
  POST /api/generate  -> prompt in, generated text out
"""
import json
import urllib.request

import numpy as np

import config


class Ollama:
    def __init__(self, url=config.OLLAMA_URL):
        self.url = url.rstrip("/")          # e.g. "http://127.0.0.1:11434"

    def _post(self, path, body, timeout):
        """Send a JSON POST to Ollama and return its JSON reply (shared by embed and generate)."""
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def models(self):
        """API: GET /api/tags -> names of installed models.
        Output example: ["curamate-qwen3.5-4b:q4_k_m", "nomic-embed-text:latest"]
        Used by the health check and to pick which LLM to use."""
        with urllib.request.urlopen(self.url + "/api/tags", timeout=3) as r:
            return [m["name"] for m in json.load(r)["models"]]

    def embed(self, texts, batch=64):
        """API: POST /api/embed  {"model": "nomic-embed-text", "input": [texts]} -> {"embeddings": [[768 numbers], ...]}

        Input:  list of strings, e.g. ["search_query: short of breath"]
        Output: numpy array (len(texts) x 768), each row scaled to length 1
                so a dot product between two rows = their cosine similarity.
        """
        out = []
        for i in range(0, len(texts), batch):            # send 64 texts per request
            res = self._post("/api/embed", {"model": config.EMBED_MODEL, "input": texts[i:i + batch]}, timeout=120)
            out += res["embeddings"]
        E = np.array(out, dtype=np.float32)
        return E / np.linalg.norm(E, axis=1, keepdims=True)   # normalise -> cosine = dot product

    def generate(self, model, prompt):
        """API: POST /api/generate -> the LLM's answer.

        Request:  {"model": ..., "prompt": ..., "stream": false, "think": false,
                   "keep_alive": "2m", "options": {"temperature": 0.2, "num_ctx": 4096, "num_predict": 500}}
        Output:   Ollama's JSON, e.g. {"response": "### Why this fits ...", "eval_count": 233, "total_duration": ...}
        stream=false -> wait for the full answer; think=false -> skip Qwen's hidden reasoning (faster);
        keep_alive -> unload the model from RAM 2 minutes after the last report.
        """
        body = {"model": model, "prompt": prompt, "stream": False, "think": False,
                "keep_alive": config.LLM_KEEP_ALIVE, "options": config.LLM_OPTIONS}
        return self._post("/api/generate", body, timeout=300)
