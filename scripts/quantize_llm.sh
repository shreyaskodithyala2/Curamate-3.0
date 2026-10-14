#!/usr/bin/env bash
# Quantize Qwen3.5-4B to 4-bit (Q4_K_M) ourselves and load it into Ollama.
#
#   original Hugging Face weights (bf16, 8.7 GB)
#     -> 1. convert to GGUF, still 16-bit        (llama.cpp convert_hf_to_gguf.py)
#     -> 2. quantize to Q4_K_M, 5.13 bits/weight (llama.cpp llama-quantize)      -> 2.7 GB
#     -> 3. measure quality loss (perplexity)    (llama.cpp llama-perplexity)
#     -> 4. register with Ollama                 (ollama create)
#
# Run from anywhere:  bash scripts/quantize_llm.sh
# Big files go to ../llm_build (outside the repo) unless WORK is set.
# Needs: git, python3.12, ~20 GB free disk, internet for the one-time download.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${WORK:-$REPO/../llm_build}"
MODEL=Qwen/Qwen3.5-4B
NAME=curamate-qwen3.5-4b:q4_k_m
mkdir -p "$WORK" && cd "$WORK"

# 0. tools: llama.cpp source + a venv with its converter requirements (+ cmake to compile)
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
[ -d .venv ] || python3.12 -m venv .venv
.venv/bin/pip install -q -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt cmake huggingface_hub
PATH="$PWD/.venv/bin:$PATH"
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build llama.cpp/build --target llama-quantize llama-perplexity -j 8

# 0b. original full-precision weights from Hugging Face (public, Apache-2.0)
[ -d Qwen3.5-4B ] || hf download "$MODEL" --local-dir Qwen3.5-4B

# 1. Hugging Face safetensors -> one GGUF file, still bf16 (lossless).
#    --no-mtp drops the multi-token-prediction draft layer; it is only used for speculative decoding,
#    and older Ollama versions fail to load a model that contains it.
python llama.cpp/convert_hf_to_gguf.py Qwen3.5-4B --outtype bf16 --no-mtp --outfile qwen3.5-4b-bf16.gguf

# 2. bf16 -> Q4_K_M. Most weight matrices become 4-bit k-quants; the most sensitive ones
#    (token embeddings, half of the attn_qkv / attn_v / ffn_down matrices) are kept at 6-bit;
#    small 1-D tensors (norms, biases) stay float32.
./llama.cpp/build/bin/llama-quantize qwen3.5-4b-bf16.gguf qwen3.5-4b-q4_k_m.gguf Q4_K_M 8

# 3. quality check: perplexity on Curamate's own clinical text (lower = better)
python3 -c "
import json; c = json.load(open('$REPO/knowledge/disease_cards.json'))
open('eval_text.txt', 'w').write('\n\n'.join(v['overview'] + ' ' + v['red_flags'] + ' ' + v['next_steps'] + ' '
    + v['management'] + ' ' + v['differentiate'] for k, v in c.items() if not k.startswith('_')))"
for m in bf16 q4_k_m; do
  ./llama.cpp/build/bin/llama-perplexity -m "qwen3.5-4b-$m.gguf" -f eval_text.txt -c 512 -ngl 99 2>&1 | grep "Final estimate"
done

# 4. register with Ollama (reuses Ollama's built-in Qwen3.5 chat format)
#    The Modelfile in the repo has a placeholder FROM line; point it at the file we just built.
sed "s#^FROM .*#FROM $PWD/qwen3.5-4b-q4_k_m.gguf#" "$REPO/llm/Modelfile" > Modelfile.local
ollama create "$NAME" -f Modelfile.local
ollama run "$NAME" "In one sentence, what is anaphylaxis?" --think=false
