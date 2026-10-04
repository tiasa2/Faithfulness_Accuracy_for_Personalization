#!/usr/bin/env bash
# PrefEval grid (paper App. I): {untrained, curriculum} x {zero-shot, remind} x {10, 0} filler turns,
# then both judges. Needs ./PrefEval (see README) and API keys in .env.
#   bash scripts/prefeval/run.sh <curriculum checkpoint dir> [output dir]
set -euo pipefail
cd "$(dirname "$0")/../.."
CKPT="$1"; OUT="${2:-runs/prefeval}"
mkdir -p "$OUT"
set -a; [ -f .env ] && source .env; set +a
for turns in 10 0; do
  for task in remind zero-shot; do
    for model in untrained curriculum; do
      path=$([ "$model" = untrained ] && echo Qwen/Qwen3-8B || echo "$CKPT")
      python scripts/prefeval/generate.py --model "$path" --task "$task" --inter_turns "$turns" \
          --output "$OUT/${model}_${task}_${turns}turns.json"
    done
    for judge in "--provider openai --model gpt-5" "--provider anthropic --model claude-sonnet-5"; do
      python scripts/prefeval/judge.py "$OUT/untrained_${task}_${turns}turns.json" "$OUT/curriculum_${task}_${turns}turns.json" $judge
    done
  done
done
