#!/usr/bin/env bash
# Curriculum iterative DPO on Qwen3-8B (paper Sec. 5.1, App. G). Run from anywhere.
#
# Each iteration i = 1..3:
#   1. rollouts: five prompting setups for every (training respondent, question), T = 0.6
#   2. labels: correctness and faithfulness F with the Qwen3-32B judge
#   3. DPO on curriculum pairs: iteration 1 rejects correct-but-unfaithful responses,
#      later iterations reject incorrect lowest-reward responses (faithful_accuracy/pairs.py)
#   4. held-out evaluation of the new checkpoint
#
# Env: RUN_DIR (runs/curriculum), NUM_ITERS (3), REPORT_TO (none | wandb)
# The paper's trained model is $RUN_DIR/iter3/check_point_curriculum.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR="${RUN_DIR:-runs/curriculum}"
NUM_ITERS="${NUM_ITERS:-3}"
REPORT_TO="${REPORT_TO:-none}"
mkdir -p "$RUN_DIR"

model="Qwen/Qwen3-8B"
bash scripts/evaluate.sh "$model" "$RUN_DIR/iter0_heldout"
for i in $(seq 1 "$NUM_ITERS"); do
  dir="$RUN_DIR/iter$i"; mkdir -p "$dir"
  python scripts/generate.py --mode setups --split train --model "$model" --output "$dir/rollouts.json"
  python scripts/judge.py --input "$dir/rollouts.json" --output "$dir/rollouts_labeled.json"
  cat > "$dir/dpo.yaml" <<EOT
train_file: $dir/rollouts_labeled.json
iteration: $i
model_name_or_path: $model
output_dir: $dir
run_name: curriculum_iter$i
learning_rate: 5.0e-7
lr_scheduler_type: cosine
num_train_epochs: 1
max_grad_norm: 1.0
beta: 0.1
loss_type: sigmoid
label_smoothing: 0.0
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
gradient_checkpointing: true
max_length: 12288
max_prompt_length: 8192
bf16: true
logging_steps: 1
remove_unused_columns: false
do_eval: false
report_to: $REPORT_TO
EOT
  accelerate launch --num_processes 1 --num_machines 1 --mixed_precision bf16 scripts/train_dpo.py "$dir/dpo.yaml"
  model="$dir/check_point_curriculum"
  bash scripts/evaluate.sh "$model" "$dir/heldout"
done
