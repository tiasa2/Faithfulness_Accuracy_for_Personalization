#!/usr/bin/env bash
# Held-out OpinionQA evaluation of one model: 100 test respondents x 15 questions, persona + axioms
# prompt, greedy decoding -> Qwen3-32B labels -> summary.
#   bash scripts/evaluate.sh <model: HF id or checkpoint dir> <output prefix>
# e.g. bash scripts/evaluate.sh Qwen/Qwen3-8B runs/untrained
# writes <prefix>_generations.json, <prefix>_labeled.json, <prefix>_summary.json
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="$1"; OUT="$2"
mkdir -p "$(dirname "$OUT")"
python scripts/generate.py --mode eval --split test --model "$MODEL" --output "${OUT}_generations.json"
python scripts/judge.py --input "${OUT}_generations.json" --output "${OUT}_labeled.json"
python scripts/summarize.py "${OUT}_labeled.json" --out "${OUT}_summary.json"
