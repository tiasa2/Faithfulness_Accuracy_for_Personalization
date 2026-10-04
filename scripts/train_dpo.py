"""One curriculum DPO iteration (paper Sec. 5.1, App. G).

  accelerate launch --num_processes 1 --mixed_precision bf16 scripts/train_dpo.py config.yaml

The YAML holds TRL DPOConfig / ModelConfig fields plus:
  train_file   labelled rollouts from scripts/judge.py
  iteration    curriculum iteration (1 = faithfulness objective, >= 2 = combined objective)
The final model is written to <output_dir>/check_point_curriculum.
"""
import os
import sys
from dataclasses import dataclass, field

import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import DPOConfig, ModelConfig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from faithful_accuracy.dpo_trainer import MyDPOTrainer  # noqa: E402
from faithful_accuracy.pairs import MIN_REWARD_SPREAD, build_pairs  # noqa: E402


@dataclass
class ScriptArguments:
    train_file: str = field(metadata={"help": "labelled rollouts (scripts/judge.py output)"})
    iteration: int = field(default=1, metadata={"help": "curriculum iteration"})
    tokenizer: str = field(default="Qwen/Qwen3-8B")
    eot_token: str = field(default="<|im_end|>")
    margin_scale: float = field(default=1.0)
    min_reward_spread: float = field(default=MIN_REWARD_SPREAD)


def main():
    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)
    script_args, training_args, model_config = HfArgumentParser((ScriptArguments, DPOConfig, ModelConfig)).parse_dict(config)
    training_args.save_only_model = True

    import json
    records = json.load(open(script_args.train_file))
    pairs = build_pairs(records, script_args.iteration, script_args.eot_token, script_args.margin_scale,
                        script_args.min_reward_spread)
    if not pairs["prompt"]:
        raise SystemExit("no preference pairs survive the curriculum filters")
    train_dataset = Dataset.from_dict(pairs)
    print(f"iteration {script_args.iteration}: {len(train_dataset)} pairs from {len(records)} prompts")

    load = dict(trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **load)
    model.config.use_cache = False
    model_ref = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **load)
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token

    trainer = MyDPOTrainer(
        model,
        model_ref,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        beta=training_args.beta,
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        loss_type=training_args.loss_type,
    )
    steps_per_epoch = len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    trainer.create_optimizer_and_scheduler(int(steps_per_epoch * training_args.num_train_epochs))
    trainer.train()
    trainer.model.save_pretrained(os.path.join(training_args.output_dir, "check_point_curriculum"))


if __name__ == "__main__":
    main()
