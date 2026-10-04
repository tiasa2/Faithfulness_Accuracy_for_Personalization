"""Label PersonaMem-v2 generations with correctness and faithfulness F (same judge as OpinionQA).

  python scripts/personamem/judge.py --input personamem_untrained.json --output personamem_untrained_labeled.json
"""
import argparse
import json
import os
import sys

os.environ.setdefault("VLLM_USE_V1", "0")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from faithful_accuracy.judge import JUDGE_MODEL, FaithfulnessJudge, axiom_lines  # noqa: E402
from faithful_accuracy.parsing import selected_letter  # noqa: E402
from faithful_accuracy.templates import record_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="scripts/personamem/generate.py --generate output")
    ap.add_argument("--output", required=True)
    ap.add_argument("--judge_model", default=JUDGE_MODEL)
    args = ap.parse_args()

    rows = json.load(open(args.input))
    scores = FaithfulnessJudge(args.judge_model).score([(r["axioms"], r.get("Response", "")) for r in rows])
    out = []
    for i, (r, s) in enumerate(zip(rows, scores)):
        gold = str(r.get("correct_option", r.get("answer", ""))).upper()
        pred = selected_letter(r.get("Response", ""))
        out.append({
            "index": i,
            "persona_id": r.get("persona_id"),
            "question": r.get("question"),
            "options": r.get("options"),
            "correct_option": gold,
            "selected_answer": pred,
            "prompt": record_prompt(r.get("question", ""), r.get("options_text", ""), r.get("persona", ""), r["axioms"]),
            "responses": [r.get("Response", "")],
            "num_axioms": [len(axiom_lines(r["axioms"]))],
            "correctness": [int(pred == gold)],
            "grounding_scores": [s["grounding"]],
            "contradiction_labels": [s["contradiction_labels"]],
            "contradiction_rate": [s["contradiction_rate"]],
            "rewards": [s["reward"]],
        })
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"wrote {len(out)} labelled records -> {args.output}")


if __name__ == "__main__":
    main()
