#!/usr/bin/env python3
"""NLI faithfulness baseline (paper Sec. 4.3, App. D.5).

Splits each reasoning chain on "." and checks every step against every axiom with an NLI model
(cross-encoder/nli-deberta-v3-large). A step scores 0 if any axiom is predicted as contradiction,
else 1; a response's NLI faithfulness is the mean step score.

  python scripts/nli_baseline.py data/human_eval/chains_batch1.json data/human_eval/chains_batch2.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - keeps the script usable without tqdm.
    def tqdm(iterable=None, **_: Any):
        return iterable if iterable is not None else range(0)


DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-large"
LABEL_MAPPING = ["contradiction", "entailment", "neutral"]


def extract_axioms(prompt: str) -> list[str]:
    marker = "Axioms:"
    idx = prompt.find(marker)
    if idx < 0:
        return []
    text = prompt[idx + len(marker) :].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and lines[0].lower().startswith("the following axioms represent"):
        lines = lines[1:]
    return lines


def extract_reasoning(response: str) -> str:
    match = re.search(r"###\s*Reasoning\s*:\s*(.*)", response, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else str(response or "").strip()


def reasoning_steps(reasoning: str) -> list[str]:
    return [step.strip() for step in reasoning.split(".") if step.strip()]


class NLIFaithfulnessScorer:
    def __init__(self, model_name: str, batch_size: int) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded {model_name} on {self.device}")

    def predict_labels(self, premises: list[str], hypotheses: list[str]) -> list[str]:
        labels: list[str] = []
        for start in range(0, len(premises), self.batch_size):
            batch_premises = premises[start : start + self.batch_size]
            batch_hypotheses = hypotheses[start : start + self.batch_size]
            features = self.tokenizer(
                batch_premises,
                batch_hypotheses,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**features).logits
            labels.extend(LABEL_MAPPING[i] for i in logits.argmax(dim=1).tolist())
        return labels

    def score_response(self, axioms: list[str], response: str) -> tuple[float, list[dict[str, Any]]]:
        steps = reasoning_steps(extract_reasoning(response))
        if not steps or not axioms:
            return 0.0, []

        all_premises: list[str] = []
        all_hypotheses: list[str] = []
        spans: list[tuple[int, int]] = []
        for step in steps:
            start = len(all_premises)
            all_premises.extend(axioms)
            all_hypotheses.extend([step] * len(axioms))
            spans.append((start, len(all_premises)))

        labels = self.predict_labels(all_premises, all_hypotheses)
        step_records: list[dict[str, Any]] = []
        step_scores: list[int] = []
        for step, (start, end) in zip(steps, spans):
            step_labels = labels[start:end]
            contradiction_indices = [
                idx for idx, label in enumerate(step_labels) if label == "contradiction"
            ]
            score = 0 if contradiction_indices else 1
            step_scores.append(score)
            step_records.append(
                {
                    "step": step,
                    "score": score,
                    "labels": step_labels,
                    "contradiction_axiom_indices": contradiction_indices,
                }
            )

        return sum(step_scores) / len(step_scores), step_records


def process_file(
    input_path: Path,
    output_path: Path,
    scorer: NLIFaithfulnessScorer,
    show_progress: bool,
) -> list[dict[str, Any]]:
    with input_path.open(encoding="utf-8") as f:
        rows = json.load(f)

    total_responses = sum(len(row.get("responses", [])) for row in rows)
    progress = tqdm(
        total=total_responses,
        desc=input_path.name,
        unit="response",
        disable=not show_progress,
    )
    summary_rows: list[dict[str, Any]] = []
    try:
        for row_idx, row in enumerate(rows):
            axioms = extract_axioms(row.get("prompt", ""))
            scores: list[float] = []
            labels_by_response: list[list[dict[str, Any]]] = []
            for response_idx, response in enumerate(row.get("responses", []), start=1):
                score, step_labels = scorer.score_response(axioms, response)
                scores.append(score)
                labels_by_response.append(step_labels)
                progress.update(1)
                progress.set_postfix(row=row_idx + 1, response=response_idx, refresh=False)

            row["nli_model"] = scorer.model_name
            row["nli_faithfulness"] = scores
            row["nli_step_labels"] = labels_by_response
            summary_rows.append(
                {
                    "row": row_idx,
                    "n_responses": len(scores),
                    "mean_nli_faithfulness": sum(scores) / len(scores) if scores else 0.0,
                    **{f"nli_faithfulness_{i + 1}": score for i, score in enumerate(scores)},
                }
            )
    finally:
        progress.close()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    return summary_rows


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="labelled JSON files (one record per query, five reasoning chains each)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output-suffix",
        default="_nli_faithfulness.json",
        help="Suffix for scored JSON outputs.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    args = parser.parse_args()

    scorer = NLIFaithfulnessScorer(args.model, args.batch_size)
    for input_path in args.inputs:
        output_path = input_path.with_name(input_path.stem + args.output_suffix)
        summary_path = output_path.with_suffix(".summary.csv")
        summary_rows = process_file(input_path, output_path, scorer, not args.no_progress)
        write_summary(summary_path, summary_rows)
        print(f"Wrote {output_path}")
        print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
