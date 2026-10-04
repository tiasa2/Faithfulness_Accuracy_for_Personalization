"""Label generations with correctness and faithfulness F (Qwen3-32B judge).

Input: output of scripts/generate.py. Output: one record per (respondent, question) with
  prompt, responses, response_types, correctness, grounding_scores, contradiction_labels,
  contradiction_rate, rewards (= F), num_axioms
Every response of a record is judged against the person's full axiom set.

  python scripts/judge.py --input heldout.json --output heldout_labeled.json
"""
import argparse
import json
import os
import sys

os.environ.setdefault("VLLM_USE_V1", "0")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from faithful_accuracy.judge import JUDGE_MODEL, FaithfulnessJudge, axiom_lines  # noqa: E402
from faithful_accuracy.parsing import selected_letter  # noqa: E402
from faithful_accuracy.templates import SETUPS, record_prompt  # noqa: E402


def to_records(generations):
    """Flatten respondent-level generations into (respondent, question) records."""
    out = []
    for g in generations:
        keys = [k[len("Question_"):] for k in g if k.startswith("Question_")]
        for key in keys:
            slots = ["Response"] if f"Response_{key}" in g else [s for s in SETUPS if f"{s}_{key}" in g]
            options, axioms = g[f"Option_{key}"], g[f"Axioms_for_{key}"]
            gold = next((letter for letter, text in options.items() if text == g[f"Answer_{key}"]), None)
            responses = [g[f"{s}_{key}"] for s in slots]
            out.append({
                "prompt": record_prompt(g[f"Question_{key}"], options, g["Persona"], axioms),
                "responses": responses,
                "response_types": slots,
                "axioms": axioms,
                "num_axioms": [len(axiom_lines(axioms))] * len(slots),
                "correctness": [int(gold is not None and selected_letter(r) == gold) for r in responses],
            })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--judge_model", default=JUDGE_MODEL)
    args = ap.parse_args()

    records = to_records(json.load(open(args.input)))
    items = [(rec["axioms"], resp) for rec in records for resp in rec["responses"]]
    scores = iter(FaithfulnessJudge(args.judge_model).score(items))
    for rec in records:
        s = [next(scores) for _ in rec["responses"]]
        rec["grounding_scores"] = [x["grounding"] for x in s]
        rec["contradiction_labels"] = [x["contradiction_labels"] for x in s]
        rec["contradiction_rate"] = [x["contradiction_rate"] for x in s]
        rec["rewards"] = [x["reward"] for x in s]
        del rec["axioms"]
    with open(args.output, "w") as f:
        json.dump(records, f, indent=4, ensure_ascii=False)
    print(f"wrote {len(records)} labelled records -> {args.output}")


if __name__ == "__main__":
    main()
