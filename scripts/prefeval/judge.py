#!/usr/bin/env python3
"""PrefEval judge (paper App. I): PrefEval's four checks, run verbatim, with GPT-5 or Claude.

Each response gets PrefEval's binary checks (PrefEval/error_type/*.txt): acknowledges the preference,
violates it, hallucinates it, is helpful; they are combined with PrefEval's boolean rules into
inconsistent / hallucinated / unaware / unhelpful errors and a preference-following accuracy.
PrefEval's own judge (Claude 3 Sonnet) is retired, so we use GPT-5 (reasoning effort minimal) and
claude-sonnet-5; the paper's consensus score counts a response as accurate only if both judges agree.

  python scripts/prefeval/judge.py untrained_reminder_10turns.json curriculum_reminder_10turns.json --model gpt-5
  python scripts/prefeval/judge.py ... --provider anthropic --model claude-sonnet-5

Writes <input>_<model>judged.json with an `evaluation_error_analysis` field per item; reruns resume.
With two input files it also prints the paired McNemar test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ERROR_TYPE_DIR = Path("PrefEval/error_type")
FILE_DICT = {  # their file_dict, order preserved: acknow must run before hallucinate
    "acknow": "check_acknowledge.txt",
    "violate": "check_violation.txt",
    "hallucinate": "check_hallucination.txt",
    "helpful": "check_helpful.txt",
}
SYSTEM_PROMPT = """You are a helpful assistant in evaluating an AI assistant's reponse. You should be fair and strict and follow the user's instruction"""  # verbatim, incl. their typo


def parse_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def build_prompt(metric: str, template: str, task: dict, error_check: dict) -> str:
    """Their placeholder substitution, verbatim."""
    text = template
    if metric == "acknow":
        text = text.replace("{end_generation}", task["response"])
        text = text.replace("{question}", task["question"])
    elif metric in ("violate", "helpful"):
        text = text.replace("{preference}", task["preference"])
        text = text.replace("{question}", task["question"])
        text = text.replace("{end_generation}", task["response"])
    elif metric == "hallucinate":
        text = text.replace("{preference}", task["preference"])
        text = text.replace("{assistant_restatement}", error_check["acknow"]["extract_pref"])
    return text


def make_client(args):
    if args.provider == "anthropic":
        import anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY")
        return anthropic.Anthropic(timeout=90.0, max_retries=0)
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY")
    return OpenAI(api_key=api_key, timeout=90.0, max_retries=0)


def gpt_call(client, prompt: str, model: str, effort: str, max_retries: int = 5,
             provider: str = "openai") -> str:
    if provider == "anthropic":
        # claude-sonnet-5: adaptive thinking by default; low effort suits binary checks
        eff = "low" if effort in ("", "minimal") else effort
        kwargs = dict(model=model, max_tokens=2000, system=SYSTEM_PROMPT,
                      messages=[{"role": "user", "content": prompt}],
                      extra_body={"output_config": {"effort": eff}})
        for attempt in range(max_retries):
            try:
                r = client.messages.create(**kwargs)
                return "".join(b.text for b in r.content if b.type == "text").strip()
            except TypeError:
                kwargs.pop("extra_body", None)
            except Exception as e:
                msg = str(e)
                if "output_config" in msg or "invalid_request" in msg and "effort" in msg:
                    kwargs.pop("extra_body", None)
                    continue
                if attempt == max_retries - 1:
                    print(f"[claude_call] giving up after {max_retries}: {e}")
                    return ""
                time.sleep(2 ** attempt)
        return ""
    kwargs = dict(model=model,
                  messages=[{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}])
    if effort:
        kwargs["reasoning_effort"] = effort
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(**kwargs)
            return (r.choices[0].message.content or "").strip()
        except TypeError:
            kwargs.pop("reasoning_effort", None)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[gpt_call] giving up after {max_retries}: {e}")
                return ""
            time.sleep(2 ** attempt)
    return ""


def judge_item(client, args, templates, task: dict) -> dict:
    """Run the four checks for one item; returns evaluation_error_analysis dict."""
    error_check = dict(task.get("evaluation_error_analysis", {}))
    for metric, template in templates:
        if metric in error_check and error_check[metric].get("answer"):
            continue
        if metric == "hallucinate" and not error_check.get("acknow", {}).get("answer"):
            continue  # dependency not ready (acknow call failed)
        prompt = build_prompt(metric, template, task, error_check)
        resp = gpt_call(client, prompt, args.model, args.effort, provider=args.provider)
        if not resp:
            continue  # leave missing; rerun fills it in
        if metric == "acknow":
            error_check[metric] = {"answer": parse_tag(resp, "answer"),
                                   "extract_pref": parse_tag(resp, "preference")}
        else:
            error_check[metric] = {"answer": parse_tag(resp, "answer"),
                                   "explanation": parse_tag(resp, "explanation")}
    return error_check


def classify(error_types: dict) -> dict:
    is_ack = "yes" in error_types.get("acknow", {}).get("answer", "").lower()
    is_hal = is_ack and "yes" in error_types.get("hallucinate", {}).get("answer", "").lower()
    is_vio = "yes" in error_types.get("violate", {}).get("answer", "").lower()
    is_unhelpful = "no" in error_types.get("helpful", {}).get("answer", "").lower()
    is_inconsistent = is_ack and not is_hal and is_vio and not is_unhelpful
    is_hal_vio = is_ack and is_hal and is_vio and not is_unhelpful
    is_unaware_vio = not is_ack and is_vio and not is_unhelpful
    acc = not any([is_inconsistent, is_hal_vio, is_unaware_vio, is_unhelpful])
    return {"unhelpful": is_unhelpful, "inconsistent": is_inconsistent,
            "hallucination_violation": is_hal_vio, "unaware_violation": is_unaware_vio,
            "accurate": acc}


def summarize(data: list, label: str) -> list:
    rows = [classify(t["evaluation_error_analysis"]) for t in data]
    n = len(rows)
    print(f"\n=== {label} (n={n}) ===")
    for k in ("unhelpful", "inconsistent", "hallucination_violation", "unaware_violation"):
        c = sum(r[k] for r in rows)
        print(f"  {k:25s}: {c:4d} ({c/n:.3f})")
    acc = sum(r["accurate"] for r in rows)
    print(f"  preference-following acc : {acc:4d} ({acc/n:.3f})")
    return rows


def mcnemar(rows_a: list, rows_b: list, la: str, lb: str) -> None:
    b = sum(1 for x, y in zip(rows_a, rows_b) if x["accurate"] and not y["accurate"])
    c = sum(1 for x, y in zip(rows_a, rows_b) if not x["accurate"] and y["accurate"])
    n = b + c
    if n == 0:
        print("\nMcNemar: no discordant pairs")
        return
    p = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n * 2
    print(f"\nPaired McNemar ({la} vs {lb}): {la}-only correct = {b}, {lb}-only correct = {c}, "
          f"exact p = {min(p,1.0):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="PrefEval 4-check judge with GPT-5.")
    ap.add_argument("files", nargs="+", help="scripts/prefeval/generate.py outputs")
    ap.add_argument("--model", default=os.environ.get("OPENAI_JUDGE_MODEL", "gpt-5"),
                    help="gpt-5 | gpt-5-mini | claude-sonnet-5 | ...")
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic"],
                    help="anthropic = Claude judge via ANTHROPIC_API_KEY (use with --model claude-sonnet-5)")
    ap.add_argument("--effort", default="minimal",
                    help="gpt-5 reasoning_effort: minimal|low|medium|high")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N items (smoke test)")
    args = ap.parse_args()

    templates = [(m, (ERROR_TYPE_DIR / f).read_text()) for m, f in FILE_DICT.items()]
    client = make_client(args)
    tag = re.sub(r"[^a-z0-9]", "", args.model.lower())

    all_rows = {}
    for path in args.files:
        in_path = Path(path)
        out_path = in_path.with_name(f"{in_path.stem}_{tag}judged.json")
        src = json.loads(out_path.read_text()) if out_path.exists() else json.loads(in_path.read_text())
        data = src[: args.limit] if args.limit else src
        print(f"\nJudging {in_path.name} -> {out_path.name} ({len(data)} items, model={args.model})")

        lock = threading.Lock()
        done = 0

        def work(i_task):
            i, task = i_task
            task["evaluation_error_analysis"] = judge_item(client, args, templates, task)
            return i

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(work, (i, t)) for i, t in enumerate(data)]
            for fut in as_completed(futs):
                fut.result()
                with lock:
                    done += 1
                    if done % 50 == 0:
                        print(f"  {done}/{len(data)}")
                        out_path.write_text(json.dumps(src, indent=2, ensure_ascii=False))
        out_path.write_text(json.dumps(src, indent=2, ensure_ascii=False))

        missing = sum(1 for t in data
                      if any(m not in t["evaluation_error_analysis"]
                             or not t["evaluation_error_analysis"][m].get("answer")
                             for m in FILE_DICT))
        if missing:
            print(f"  WARNING: {missing} items have incomplete checks — rerun to fill in.")
        all_rows[in_path.name] = summarize(data, in_path.name)

    if len(args.files) == 2:
        (na, ra), (nb, rb) = all_rows.items()
        if len(ra) == len(rb):
            mcnemar(ra, rb, na.split("_")[1], nb.split("_")[1])


if __name__ == "__main__":
    main()
