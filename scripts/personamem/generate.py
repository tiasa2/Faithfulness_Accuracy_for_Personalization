#!/usr/bin/env python3
"""PersonaMem-v2 transfer evaluation (paper App. H): build persona/axiom MCQ records and generate.

Each benchmark row becomes a multiple-choice question: its profile fields form the persona and its
stated traits / values / row-level preferences form the axioms. Answer options are shuffled
deterministically (seed + row index).

  python scripts/personamem/generate.py --generate --model Qwen/Qwen3-8B --output personamem_untrained.json
  python scripts/personamem/judge.py --input personamem_untrained.json --output personamem_untrained_labeled.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
from typing import Any, Dict, List, Mapping, Sequence, Union

from datasets import load_dataset

PERSONA_FIELDS: Sequence[str] = (
    "name",
    "age",
    "gender",
    "sexual_orientation",
    "race_ethnicity",
    "political_affiliation",
    "location",
    "education",
    "occupation",
    "family_background",
    "hobbies_interests",
    "technology_use",
    "speaking_style_with_chatbot",
)

AXIOM_FIELDS: Sequence[str] = (
    "personality.traits",
    "personality.conflict_style",
    "values_beliefs",
)

OPTION_LABELS: Sequence[str] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

PERSONA_LABELS: Mapping[str, str] = {
    "name": "Name",
    "age": "Age",
    "gender": "Gender",
    "sexual_orientation": "Sexual orientation",
    "race_ethnicity": "Race / ethnicity",
    "political_affiliation": "Political affiliation",
    "location": "Location",
    "education": "Education",
    "occupation": "Occupation",
    "family_background": "Family background",
    "hobbies_interests": "Hobbies and interests",
    "technology_use": "Technology use",
    "speaking_style_with_chatbot": "Speaking style with chatbot",
}


def flatten_path_dict(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """
    Flatten nested dicts using dotted paths (e.g. family_background.spouse.name).
    Avoids collisions like top-level 'name' being overwritten by a nested 'name'.
    """
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(flatten_path_dict(v, path))
            else:
                out[path] = v
        return out
    return {prefix: obj} if prefix else {}


def build_flat_persona(expanded: Mapping[str, Any]) -> Dict[str, Any]:
    """One display row per scalar; nested structures become dotted keys."""
    flat: Dict[str, Any] = {}
    for field in PERSONA_FIELDS:
        if field not in expanded:
            continue
        val = expanded[field]
        if isinstance(val, dict):
            flat.update(flatten_path_dict(val, field))
        else:
            flat[field] = val
    return flat


def extract_strings(v: Any) -> List[str]:
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, list):
        res: List[str] = []
        for item in v:
            res.extend(extract_strings(item))
        return res
    if isinstance(v, dict):
        res = []
        for val in v.values():
            res.extend(extract_strings(val))
        return res
    if v is None:
        return []
    return [str(v)]


def _field_get(expanded: Mapping[str, Any], dotted: str) -> Any:
    val: Any = expanded
    for part in dotted.split("."):
        if isinstance(val, dict) and part in val:
            val = val[part]
        else:
            return None
    return val


def label_for_persona_key(key: str) -> str:
    if key in PERSONA_LABELS:
        return PERSONA_LABELS[key]
    if "." in key:
        parts = key.split(".")
        return " — ".join(
            PERSONA_LABELS.get(p, p.replace("_", " ").strip().title()) for p in parts
        )
    return key.replace("_", " ").strip().title()


def format_persona_block(flat_persona: Mapping[str, Any]) -> str:
    """Persona block in the same layout as OpinionQA personas (header line + one field per line)."""
    lines = ["Persona description :"]
    for k, v in flat_persona.items():
        if v is None:
            continue
        if isinstance(v, float) and v != v:  # NaN
            continue
        s = str(v).strip()
        if not s:
            continue
        lines.append(f"{label_for_persona_key(k)}: {s}")
    return "\n".join(lines) + "\n"


def format_axioms_block(axiom_strings: Sequence[str]) -> str:
    """Match create_axioms() opening line; one belief per line like map_axiom_to_response."""
    header = "The following axioms represent the persona's beliefs or preferences:\n"
    cleaned = [s.strip() for s in axiom_strings if s and str(s).strip()]
    if not cleaned:
        return header.rstrip()
    return header + "\n".join(cleaned) + "\n"


def parse_expanded_persona(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return ast.literal_eval(raw)


def collect_axiom_strings(expanded: Mapping[str, Any]) -> List[str]:
    raw: List[str] = []
    for field in AXIOM_FIELDS:
        val = _field_get(expanded, field)
        raw.extend(extract_strings(val))
    return raw


def collect_row_preference_strings(row: Mapping[str, Any]) -> List[str]:
    """Add PersonaMem's row-level preference as ordinary axiom context."""
    raw: List[str] = []
    preference = row.get("preference")
    if preference:
        raw.append(f"The person has the following preference: {preference}")

    prev_pref = row.get("prev_pref")
    if prev_pref:
        raw.append(f"The person's previous related preference was: {prev_pref}")

    topic_preference = row.get("topic_preference")
    if topic_preference:
        raw.append(f"This preference is related to the topic: {topic_preference}")

    return raw


def parse_user_query_cell(cell: Union[str, Dict[str, Any]]) -> str:
    """
    Cells are often a string like "{'role': 'user', 'content': '...'}" or a dict.
    """
    if isinstance(cell, dict):
        return str(cell.get("content", "")).strip()
    text = str(cell).strip()
    if not text:
        return ""
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, dict) and "content" in obj:
            return str(obj["content"]).strip()
    except (SyntaxError, ValueError):
        pass
    m = re.search(r"['\"]content['\"]\s*:\s*['\"](.*)['\"]\s*\}", text, re.DOTALL)
    if m:
        return m.group(1).replace("\\n", "\n").strip()
    return text


def normalize_incorrect_answers(cell: Any) -> List[str]:
    if cell is None:
        return []
    if isinstance(cell, list):
        return [str(x).strip() for x in cell if str(x).strip()]
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(s)
            except (SyntaxError, ValueError):
                return [s]
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
        return [str(parsed).strip()]
    return [str(cell).strip()]


def build_answer_options(
    correct_answer: Any,
    incorrect_answers: Sequence[str],
    rng: random.Random,
) -> Dict[str, Any]:
    answer_items = [(str(correct_answer).strip(), True)]
    answer_items.extend((str(x).strip(), False) for x in incorrect_answers if str(x).strip())
    if len(answer_items) > len(OPTION_LABELS):
        raise ValueError(f"Too many answer options: {len(answer_items)}")

    rng.shuffle(answer_items)

    options: Dict[str, str] = {}
    option_is_correct: Dict[str, bool] = {}
    correct_option = ""
    for i, (answer_text, is_correct) in enumerate(answer_items):
        letter = OPTION_LABELS[i]
        options[letter] = answer_text
        option_is_correct[letter] = is_correct
        if is_correct:
            correct_option = letter

    return {
        "options": options,
        "correct_option": correct_option,
        "option_is_correct": option_is_correct,
    }


def format_options_block(options: Mapping[str, str]) -> str:
    return "\n".join(f"{letter}: {text}" for letter, text in options.items()) + "\n"


def role_inference(tokenizer: Any, question: str, options: str, persona: str, axioms: str) -> str:
    standard_prompt = f"""Please read the following multiple-choice question carefully. From the perspective of the given persona and guided by the provided axioms, select ONE of the listed options and explain your reasoning.
    
    Structure your response as:
    ### Question: [Insert question here]
    ### Options: [Insert options here]
    ### Persona: [Insert persona description here]
    ### Axioms: [List of axioms guiding the persona's reasoning]
    ### Selected Answer: [A/B/C/D...]
    ### Reasoning: [Explain your answer based on the persona and axioms]
    
    Question: {question}
    Options: {options}
    Persona: {persona}
    Axioms: {axioms}
    """
    messages = [{"role": "user", "content": standard_prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_record(row: Mapping[str, Any], row_index: int, seed: int) -> Dict[str, Any]:
    expanded = parse_expanded_persona(row["expanded_persona"])
    flat_persona = build_flat_persona(expanded)

    question = parse_user_query_cell(row.get("user_query", ""))
    correct_answer = row.get("correct_answer")
    incorrect_answers = normalize_incorrect_answers(row.get("incorrect_answers"))
    answer_payload = build_answer_options(
        correct_answer=correct_answer,
        incorrect_answers=incorrect_answers,
        rng=random.Random(seed + row_index),
    )
    options = answer_payload["options"]
    correct_option = answer_payload["correct_option"]
    option_is_correct = answer_payload["option_is_correct"]

    persona = format_persona_block(flat_persona)
    general_axioms = collect_axiom_strings(expanded)
    row_preferences = collect_row_preference_strings(row)

    return {
        "persona_id": row.get("persona_id"),
        "question": question,
        "options": options,
        "options_text": format_options_block(options),
        "answer": correct_option,
        "correct_option": correct_option,
        "answer_text": correct_answer,
        "option_is_correct": option_is_correct,
        "persona": persona,
        "axioms": format_axioms_block(general_axioms + row_preferences),
        "correct_answer": correct_answer,
        "incorrect_answers": incorrect_answers,
        "persona_fields": flat_persona,
    }


def generate_responses(
    records: List[Dict[str, Any]],
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    llm = LLM(
        model=model_name_or_path,
        tokenizer=tokenizer_name_or_path,
        dtype="bfloat16",
        load_format="auto",
        seed=42,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    prompts = [
        role_inference(
            tokenizer=tokenizer,
            question=rec["question"],
            options=rec["options_text"],
            persona=rec["persona"],
            axioms=rec["axioms"],
        )
        for rec in records
    ]
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    for rec, output in zip(records, outputs):
        rec["Response"] = output.outputs[0].text




def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PersonaMem-v2 to prompt-ready persona/axioms + QA fields.")
    parser.add_argument(
        "--output",
        "-o",
        default="personamem_benchmark_formatted.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--split",
        default="benchmark_text",
        help="Dataset config / split name (default: benchmark_text).",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="If set, only export the first N rows (for smoke tests).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to shuffle answer options deterministically per row.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="If set, run vLLM generation and save a Response field for each record.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="HF id or checkpoint directory (with --generate).")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature used when --generate is set.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Sampling top_p used when --generate is set.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16000,
        help="Max generation tokens used when --generate is set.",
    )
    args = parser.parse_args()

    ds = load_dataset("bowen-upenn/PersonaMem-v2", split=args.split)
    n = len(ds) if args.max_rows is None else min(args.max_rows, len(ds))

    records = [build_record(ds[i], row_index=i, seed=args.seed) for i in range(n)]

    if args.generate:
        generate_responses(
            records=records,
            model_name_or_path=args.model,
            tokenizer_name_or_path=args.tokenizer,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Wrote {n} records to {args.output}")


if __name__ == "__main__":
    main()
