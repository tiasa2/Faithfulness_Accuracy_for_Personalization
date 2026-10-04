"""Accuracy, mean faithfulness and C&F quadrant counts of a labelled held-out file.

  python scripts/summarize.py heldout_labeled.json [--out summary.json]

A response is faithful when F == 1. This is the per-run summary behind Tables 4-5 and Figs. 2-3.
"""
import argparse
import json
import statistics


def summarize(records):
    acc = [r["correctness"][0] for r in records]
    F = [r["rewards"][0] for r in records]
    quad = {"correct_faithful": 0, "correct_unfaithful": 0, "incorrect_faithful": 0, "incorrect_unfaithful": 0}
    for c, f in zip(acc, F):
        quad[f"{'correct' if c == 1 else 'incorrect'}_{'faithful' if f >= 1 else 'unfaithful'}"] += 1
    return {"n": len(records), "accuracy": statistics.mean(acc), "faithfulness": statistics.mean(F), **quad}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labeled")
    ap.add_argument("--out")
    args = ap.parse_args()
    summary = summarize(json.load(open(args.labeled)))
    print(json.dumps(summary, indent=2))
    if args.out:
        json.dump(summary, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
