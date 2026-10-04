"""PrefEval explicit-preference generation task (Zhao et al., ICLR 2025), official protocol (paper App. I).

1. The model first answers the bare preference statement; its own reply becomes the assistant turn.
2. Conversation = [system, preference, model's reply, N filler user/assistant pairs, question], where the
   question gets PrefEval's REMINDER sentence for --task remind, and " (Please respond within 300 words.)".
   Filler = the first N pairs of PrefEval/benchmark_dataset/filtered_inter_turns.json (same for every item).
PrefEval config: system prompt "You are a helpful assistant.", 300-token budget, greedy, Qwen3 thinking off.

Needs a clone of PrefEval for the filler turns (see README).

  python scripts/prefeval/generate.py --model Qwen/Qwen3-8B --task remind --inter_turns 10 --output untrained_reminder_10turns.json
"""
import argparse
import json
import re

# verbatim from PrefEval utils/common_utils.py / config.yaml
REMINDER = "\nIn your response, please ensure that you take into account our earlier discussion, and provide an answer that is consistent with my preference."
SYSTEM_PROMPT = "You are a helpful assistant."
MAX_GEN_TOKENS = 300
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text):
    return THINK_RE.sub("", text).strip()


def load_inter_turns(path, inter_turns):
    """First inter_turns user/assistant pairs of the flattened filler conversations."""
    if inter_turns <= 0:
        return []
    flat = [m for conv in json.load(open(path)) for m in conv["conversation"]]
    msgs = flat[: inter_turns * 2]
    if len(msgs) != inter_turns * 2 or msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant":
        raise ValueError(f"cannot take {inter_turns} filler pairs from {path}")
    return [{"role": m["role"], "content": m["content"]} for m in msgs]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF id or checkpoint directory")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    ap.add_argument("--task", required=True, choices=["zero-shot", "remind"])
    ap.add_argument("--inter_turns", type=int, required=True, help="filler pairs between preference and question (0 or 10)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--data", default="data/prefeval_explicit.json")
    ap.add_argument("--inter_turns_path", default="PrefEval/benchmark_dataset/filtered_inter_turns.json")
    ap.add_argument("--enable_thinking", action="store_true", help="Qwen3 thinking (off in the paper)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = json.load(open(args.data))
    filler = load_inter_turns(args.inter_turns_path, args.inter_turns)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    llm = LLM(model=args.model, tokenizer=args.tokenizer, dtype="bfloat16", load_format="auto", seed=42)
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_GEN_TOKENS if not args.enable_thinking else 8192,
                            n=1, stop_token_ids=[tokenizer.eos_token_id])

    def chat(conversations):
        prompts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=args.enable_thinking) for c in conversations]
        return [o.outputs[0].text for o in llm.generate(prompts, sampling_params=params, use_tqdm=True)]

    # stage 1: the model's own acknowledgment of the preference
    acks = [strip_think(a) for a in chat([[{"role": "system", "content": SYSTEM_PROMPT},
                                           {"role": "user", "content": r["preference"]}] for r in rows])]
    # stage 2: preference, acknowledgment, filler turns, question
    conversations = []
    for r, ack in zip(rows, acks):
        question = r["question"] + (REMINDER if args.task == "remind" else "") + f" (Please respond within {MAX_GEN_TOKENS} words.)"
        conversations.append([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": r["preference"]},
                              {"role": "assistant", "content": ack}, *filler, {"role": "user", "content": question}])
    responses = chat(conversations)

    out = [{"index": r["index"], "topic": r["topic"], "preference": r["preference"], "question": r["question"],
            "explanation": r["explanation"], "task": args.task, "inter_turns": args.inter_turns, "model": args.model,
            "pref_ack": ack, "response_raw": resp, "response": strip_think(resp)}
           for r, ack, resp in zip(rows, acks, responses)]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)
    print(f"wrote {len(out)} records -> {args.output}")


if __name__ == "__main__":
    main()
