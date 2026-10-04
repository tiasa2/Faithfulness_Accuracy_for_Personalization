"""Generate OpinionQA responses with vLLM.

Two modes:
  setups  the five prompting setups per (respondent, question): Base, Persona, Axioms,
          Persona_and_Axioms, Relevant_Axioms. Used for curriculum training rollouts
          (--split train) and for Table 3 (--split all). Defaults: temperature 0.6, thinking off.
  eval    the persona + axioms prompt only (App. B). Used for held-out evaluation (--split test)
          and the axiom-relevance ablation (--axioms ...). Defaults: greedy, chat-template thinking default.

Examples
  python scripts/generate.py --mode setups --split train --model Qwen/Qwen3-8B --output rollouts.json
  python scripts/generate.py --mode eval --split test --model runs/curriculum/iter3/check_point_curriculum --output heldout.json
  python scripts/generate.py --mode eval --split test --model Qwen/Qwen3-8B --axioms minus_relevant \
      --relevance data/opinionqa/relevant_axioms_human.json --output ablation_minus_relevant.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from faithful_accuracy import opinionqa as O  # noqa: E402
from faithful_accuracy.templates import SETUPS, eval_prompt, setup_prompts  # noqa: E402

SPLITS = {"train": "data/opinionqa/train.csv", "test": "data/opinionqa/test.csv", "all": "data/opinionqa/respondents.csv"}


def build_records(df, mode, axioms_mode, relevance, seed):
    """One record per respondent (gold answers, persona, axioms) plus the prompts to run, in order."""
    records, prompts = [], []
    for i in range(len(df)):
        rec = {"Index": i, "Persona": O.persona_text(df, i)}
        for key in O.question_keys(df):
            c = O.cell(df, key, i)
            options = O.options_text(c["option_mapping"])
            keep = O.ablation_filter(axioms_mode, relevance, df, i, key, seed) if mode == "eval" else None
            axioms = O.axioms_text(df, i, key, keep=keep)
            rec.update({f"Question_{key}": c["question"], f"Option_{key}": O.options_dict(options),
                        f"Answer_{key}": c["response"], f"Axioms_for_{key}": axioms})
            if mode == "setups":
                relevant = O.axioms_text(df, i, key, source_keys=relevance.get(key, []))
                rec[f"Relevant_Axioms_for_{key}"] = relevant
                texts = setup_prompts(c["question"], options, rec["Persona"], axioms, relevant)
                prompts += [(i, f"{s}_{key}", texts[s]) for s in SETUPS]
            else:
                prompts.append((i, f"Response_{key}", eval_prompt(c["question"], options, rec["Persona"], axioms)))
        records.append(rec)
    return records, prompts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["setups", "eval"], required=True)
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--model", required=True, help="HF id or checkpoint directory")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--output", required=True)
    ap.add_argument("--axioms", default="all", choices=["all", "minus_relevant", "only_relevant", "minus_random"],
                    help="axiom-relevance ablation (eval mode)")
    ap.add_argument("--relevance", default="data/opinionqa/relevant_axioms_human.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, help="default: 0.6 (setups) / 0.0 (eval)")
    ap.add_argument("--thinking", choices=["on", "off", "default"], help="default: off (setups) / default (eval)")
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    args = ap.parse_args()
    temperature = args.temperature if args.temperature is not None else (0.6 if args.mode == "setups" else 0.0)
    thinking = args.thinking or ("off" if args.mode == "setups" else "default")

    df = O.load_respondents(SPLITS[args.split])
    relevance = O.load_relevance_map(args.relevance)
    records, prompts = build_records(df, args.mode, args.axioms, relevance, args.seed)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    template_kwargs = {} if thinking == "default" else {"enable_thinking": thinking == "on"}
    chats = [tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                           add_generation_prompt=True, **template_kwargs) for _, _, text in prompts]
    llm = LLM(model=args.model, tokenizer=args.tokenizer, dtype="bfloat16", load_format="auto", seed=args.seed,
              tensor_parallel_size=args.tensor_parallel_size)
    params = SamplingParams(temperature=temperature, top_p=0.95 if temperature > 0 else 1.0, max_tokens=args.max_tokens,
                            n=1, stop_token_ids=[tokenizer.eos_token_id])
    outputs = llm.generate(chats, sampling_params=params, use_tqdm=True)

    for (i, field, _), out in zip(prompts, outputs):
        records[i][field] = out.outputs[0].text
    with open(args.output, "w") as f:
        json.dump(records, f, indent=4)
    print(f"wrote {len(records)} respondents x {len(prompts) // len(records)} responses -> {args.output}")


if __name__ == "__main__":
    main()
