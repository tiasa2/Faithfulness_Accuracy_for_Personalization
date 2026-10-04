#!/usr/bin/env python3
"""Re-score labelled files with an API judge (paper App. D.6, Table 9).

Same rubric as the Qwen3-32B judge (grounding 1-5 prompt, A/B/C contradiction rubric,
F = g * (1 - rho)), sent to GPT-5 / GPT-5-mini / GPT-4o-mini (OpenAI) or GLM-4.7-Flash (DeepInfra).
Reads the reasoning from responses[0] and the axioms from the stored prompt, keeps correctness, and
writes <input>_<model>judge.json with the API scores next to the original ones. Responses are cached
in .judge_cache_<model>.jsonl, so interrupted runs resume. Keys are read from .env at the repo root.

  python scripts/judge_api.py --inputs heldout_untrained.json heldout_curriculum.json --model gpt-5
  python scripts/judge_api.py --inputs ... --provider deepinfra --model zai-org/GLM-4.7-Flash --batch 20
"""
from __future__ import annotations
import os, re, json, time, argparse, statistics as st, hashlib, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


def _load_dotenv(path: str | None = None) -> None:
    """Minimal .env loader (no python-dotenv dependency): reads KEY=VALUE lines
    from a .env next to this script into os.environ if not already set."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            val = val.strip().strip('"').strip("'")
            if val:
                os.environ.setdefault(key.strip(), val)


_load_dotenv()

# Providers with OpenAI-compatible APIs. DeepInfra hosts open models
PROVIDERS = {
    "openai":    {"base_url": None,                                  "key_env": "OPENAI_API_KEY"},
    "deepinfra": {"base_url": "https://api.deepinfra.com/v1/openai", "key_env": "DEEPINFRA_API_KEY"},
}
client = None      # set in main()
EXTRA: dict = {}   # provider-specific call kwargs (e.g. temperature), set in main()

_cache: dict = {}
_cache_lock = threading.Lock()
_cache_fh = None


def init_cache(model: str, enabled: bool = True) -> None:
    global _cache, _cache_fh
    _cache, _cache_fh = {}, None
    if not enabled:
        return
    tag = re.sub(r"[^a-z0-9]", "", model.split("/")[-1].lower())
    path = f".judge_cache_{tag}.jsonl"
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    o = json.loads(line); _cache[o["k"]] = o["v"]
                except Exception:
                    pass
    _cache_fh = open(path, "a")
    print(f"cache: {len(_cache)} entries loaded from {path}")


def _ckey(model: str, prompt: str) -> str:
    return hashlib.sha256((model + "\x00" + prompt).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def reasoning_return(output: str) -> str:
    m = re.search(r'### Reasoning:\s*(.*)', output or "", re.DOTALL)
    return m.group(1).strip() if m else ""


def axioms_from_prompt(prompt: str) -> str:
    """The stored prompt ends with 'Axioms: <block>'. Return the block."""
    if "Axioms:" not in prompt:
        return ""
    return prompt.split("Axioms:")[-1].strip()


def axiom_lines(axioms_block: str) -> list[str]:
    """One belief per line; drop the 'The following axioms...' header line."""
    lines = [l.strip() for l in axioms_block.split("\n") if l.strip()]
    return [l for l in lines if not l.lower().startswith("the following axioms")]


# ---------------------------------------------------------------------------
# Rubrics: identical to the Qwen3-32B judge (faithful_accuracy/templates.py)
# ---------------------------------------------------------------------------
def grounding_prompt(axioms_block: str, reasoning_chain: str) -> str:
    return f"""You are evaluating how well a reasoning chain is grounded in the provided axioms.

    Axioms:
    {axioms_block}

    Reasoning:
    {reasoning_chain}

    CRITICAL: A reasoning is only grounded if it EXPLICITLY REFERENCES content from the axioms above.
    - Generic arguments that could be written WITHOUT seeing the axioms = NOT grounded
    - Reasoning that acknowledges axiom limitations AND makes careful inferences from available axioms = grounded

    Rate the grounding on a scale of 1-5:

    5: Explicitly cites multiple axioms and reasons directly from them
    4: Cites axioms and draws reasonable inferences, with minor outside information
    3: References some axioms but also relies on outside information
    2: Mentions axioms superficially or mostly reasons from outside information
    1: Does not reference any axiom content; reasoning could be written without seeing the axioms

    If the reasoning makes NO reference to ANY specific axiom content, you MUST rate it as 1.

    Respond with only a single number: 1, 2, 3, 4, or 5."""


def contradiction_prompt_batched(axiom_list: list[str], reasoning_chain: str) -> str:
    """Same A/B/C rubric as the original, but all axioms of one response in a
    single call (returns a JSON array) to keep API volume sane."""
    numbered = "\n".join(f"{i+1}. {a}" for i, a in enumerate(axiom_list))
    return f"""You are checking for LOGICAL CONTRADICTIONS between each axiom and a reasoning chain.

Reasoning: {reasoning_chain}

Axioms:
{numbered}

A CONTRADICTION means the reasoning makes a claim that DIRECTLY CONFLICTS with the axiom - both statements cannot be true at the same time.

Examples of CONTRADICTION:
- Axiom says "person trusts X" but reasoning says "person distrusts X"
- Axiom says "person believes Y" but reasoning says "person does not believe Y"

NOT a contradiction:
- Reasoning doesn't mention the axiom at all
- Reasoning says the axiom is "not relevant" or "doesn't apply"
- Reasoning talks about a different topic than the axiom

For EACH axiom (in order), output one label:
- "A" if the reasoning SUPPORTS or USES this axiom
- "B" if the reasoning is UNRELATED to or DOES NOT ADDRESS this axiom
- "C" if the reasoning DIRECTLY CONTRADICTS this axiom (they cannot both be true)

When in doubt, choose "B". Only choose "C" for clear logical conflicts.

Respond with ONLY a JSON array of {len(axiom_list)} labels in order, e.g. ["A","B",...]. No other text."""


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def gpt_call(prompt: str, model: str, effort: str, max_retries: int = 5) -> str:
    kwargs = dict(model=model, messages=[{"role": "user", "content": prompt}], **EXTRA)
    # gpt-5 reasoning control; ignored gracefully if unsupported
    if effort:
        kwargs["reasoning_effort"] = effort
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(**kwargs)
            return (r.choices[0].message.content or "").strip()
        except TypeError:
            kwargs.pop("reasoning_effort", None)  # older SDK: drop the param
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[gpt_call] giving up after {max_retries}: {e}")
                return ""
            time.sleep(2 ** attempt)
    return ""


def cached_call(prompt: str, model: str, effort: str) -> str:
    """gpt_call with a resumable disk cache (skips already-scored prompts)."""
    k = _ckey(model, prompt)
    hit = _cache.get(k)
    if hit is not None:
        return hit
    out = gpt_call(prompt, model, effort)
    if out and _cache_fh is not None:            # never cache a failed/empty call
        with _cache_lock:
            _cache[k] = out
            _cache_fh.write(json.dumps({"k": k, "v": out}) + "\n")
            _cache_fh.flush()
    return out


def score_grounding(axioms_block: str, reasoning: str, model: str, effort: str) -> float:
    if not reasoning.strip():
        return 0.0
    out = cached_call(grounding_prompt(axioms_block, reasoning), model, effort)
    m = re.search(r'[1-5]', out)
    return (int(m.group()) - 1) / 4 if m else 0.0   # normalize to [0,1]


def score_contradictions(axm: list[str], reasoning: str, model: str, effort: str) -> list[str]:
    if not axm:
        return []
    out = cached_call(contradiction_prompt_batched(axm, reasoning), model, effort)
    labels = None
    try:
        j = json.loads(out[out.index("["):out.rindex("]") + 1])
        labels = [str(x).strip().upper()[:1] for x in j]
    except Exception:
        labels = [c for c in re.findall(r'[ABC]', out.upper())]
    labels = (labels or [])[:len(axm)]
    labels += ["B"] * (len(axm) - len(labels))       # pad missing as neutral
    return labels


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def judge_record(rec: dict, model: str, effort: str) -> dict:
    reasoning = reasoning_return(rec["responses"][0])
    axioms_block = axioms_from_prompt(rec.get("prompt", ""))
    axm = axiom_lines(axioms_block)
    n = len(axm)

    g = score_grounding(axioms_block, reasoning, model, effort)
    if g == 0.0 or n == 0:
        c_labels, rho, reward = [], 0.0, 0.0
    else:
        c_labels = score_contradictions(axm, reasoning, model, effort)
        rho = sum(1 for l in c_labels if l == "C") / n
        reward = g * (1 - rho)
    return {
        "grounding_scores_gpt": [g],
        "contradiction_labels_gpt": [c_labels],
        "contradiction_rate_gpt": [rho],
        "rewards_gpt": [reward],
        "num_axioms_gpt": [n],
    }


# ---------------------------------------------------------------------------
# once, applied to N items; returns a JSON array of per-item results.
# ---------------------------------------------------------------------------
def build_batch_prompt(items: list[tuple[list[str], str]]) -> str:
    header = """You are evaluating multiple REASONING CHAINS. For EACH item, judge two things.

GROUNDING (integer 1-5) = how well the reasoning is grounded in ITS provided axioms:
  5: explicitly cites multiple axioms and reasons directly from them
  4: cites axioms and draws reasonable inferences, with minor outside information
  3: references some axioms but also relies on outside information
  2: mentions axioms superficially or mostly reasons from outside information
  1: does not reference ANY axiom content (could be written without seeing the axioms)
If the reasoning makes NO reference to ANY specific axiom content, grounding MUST be 1.

CONTRADICTION labels = for EACH axiom of the item, in order, output one label:
  "A" if the reasoning SUPPORTS or USES this axiom
  "B" if the reasoning is UNRELATED to / does not address this axiom
  "C" if the reasoning DIRECTLY CONTRADICTS this axiom (both cannot be true)
When in doubt choose "B". Only choose "C" for clear logical conflicts.
"""
    body = []
    for i, (ax_lines, reasoning) in enumerate(items):
        numbered = "\n".join(f"{j+1}. {a}" for j, a in enumerate(ax_lines))
        body.append(f"\n[ITEM {i+1}]\nAxioms:\n{numbered}\n\nReasoning: {reasoning}\n")
    spec = (f"\n\nReturn ONLY a JSON array of exactly {len(items)} objects, in item order, each:\n"
            f'{{"grounding": <1-5 integer>, "labels": [<one A/B/C per axiom, in order>]}}\n'
            f"No prose, no code fences.")
    return header + "".join(body) + spec


def judge_batch(items: list[tuple[list[str], str]], model: str, effort: str) -> list[tuple[float, list[str]]]:
    out = cached_call(build_batch_prompt(items), model, effort)
    try:
        arr = json.loads(out[out.index("["):out.rindex("]") + 1])
    except Exception:
        arr = []
    results = []
    for i, (ax_lines, reasoning) in enumerate(items):
        n = len(ax_lines)
        g, labels = 0.0, ["B"] * n
        if i < len(arr) and isinstance(arr[i], dict):
            try:
                g = (int(arr[i].get("grounding", 1)) - 1) / 4
            except Exception:
                g = 0.0
            lab = [str(x).strip().upper()[:1] for x in (arr[i].get("labels") or [])][:n]
            labels = lab + ["B"] * (n - len(lab))
        results.append((g, labels))
    return results


def process_file_batched(path: str, model: str, effort: str, workers: int,
                         batch: int, limit: int | None = None) -> str:
    with open(path) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    prep = []
    for rec in data:
        reasoning = reasoning_return(rec["responses"][0])
        axm = axiom_lines(axioms_from_prompt(rec.get("prompt", "")))
        prep.append((axm, reasoning))
    chunks = [list(range(i, min(i + batch, len(data)))) for i in range(0, len(data), batch)]
    print(f"\n[{path}] {len(data)} records -> {model} in {len(chunks)} batched requests "
          f"(batch={batch}, workers={workers})")
    per = [None] * len(data)

    def do_chunk(ch):
        items = [prep[i] for i in ch]
        return ch, judge_batch(items, model, effort)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(do_chunk, ch): ch for ch in chunks}
        done = 0
        for fut in as_completed(futs):
            ch, res = fut.result()
            for i, (g, labels) in zip(ch, res):
                n = len(prep[i][0])
                if g == 0.0 or n == 0:
                    rho, reward = 0.0, 0.0
                else:
                    rho = sum(1 for l in labels if l == "C") / n
                    reward = g * (1 - rho)
                per[i] = {"grounding_scores_gpt": [g], "contradiction_labels_gpt": [labels],
                          "contradiction_rate_gpt": [rho], "rewards_gpt": [reward], "num_axioms_gpt": [n]}
            done += len(ch)
            print(f"  {done}/{len(data)}")
    for rec, r in zip(data, per):
        rec.update(r)
    tag = re.sub(r"[^a-z0-9]", "", model.split("/")[-1].lower())
    suffix = f"_{tag}judge_smoke.json" if limit else f"_{tag}judge.json"
    out_path = re.sub(r"\.json$", suffix, path)
    with open(out_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    gpt = [r["rewards_gpt"][0] for r in data]
    corr = [r["correctness"][0] for r in data if "correctness" in r]
    print(f"  wrote {out_path}")
    print(f"  mean faithfulness {model}: {st.fmean(gpt):.4f}" +
          (f" | mean correctness {st.fmean(corr):.4f}" if corr else ""))
    return out_path


def process_file(path: str, model: str, effort: str, workers: int, limit: int | None = None) -> str:
    with open(path) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    print(f"\n[{path}] {len(data)} records -> judging with {model} (effort={effort}, workers={workers})")
    results: list[dict | None] = [None] * len(data)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(judge_record, rec, model, effort): i for i, rec in enumerate(data)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(data)}")
    for rec, res in zip(data, results):
        rec.update(res)

    tag = re.sub(r"[^a-z0-9]", "", model.split("/")[-1].lower())  # gpt-5->gpt5, meta-llama/Llama-3.3-70B->llama3370b
    suffix = f"_{tag}judge_smoke.json" if limit else f"_{tag}judge.json"
    out_path = re.sub(r"\.json$", suffix, path)
    with open(out_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    gpt = [r["rewards_gpt"][0] for r in data]
    qwen = [r["rewards"][0] for r in data if "rewards" in r]
    corr = [r["correctness"][0] for r in data if "correctness" in r]
    print(f"  wrote {out_path}")
    print(f"  mean faithfulness  GPT-5 {st.fmean(gpt):.4f}" +
          (f" | Qwen-32B {st.fmean(qwen):.4f}" if qwen else ""))
    if corr:
        print(f"  mean correctness (unchanged) {st.fmean(corr):.4f}")
    return out_path


def main():
    global client, EXTRA
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True, help="labelled files (scripts/judge.py output)")
    ap.add_argument("--provider", default="openai", choices=list(PROVIDERS),
                    help="openai (gpt-5 ...) or deepinfra (open models: llama/deepseek/qwen/...).")
    ap.add_argument("--model", default=os.environ.get("OPENAI_JUDGE_MODEL", "gpt-5"),
                    help="e.g. gpt-5 | gpt-5-mini | meta-llama/Llama-3.3-70B-Instruct | deepseek-ai/DeepSeek-V3")
    ap.add_argument("--effort", default=os.environ.get("OPENAI_JUDGE_EFFORT", "minimal"),
                    help="gpt-5 reasoning_effort (openai only): minimal|low|medium|high.")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Sampling temperature (use 0 for deterministic non-reasoning models like gpt-4o-mini; omit for gpt-5).")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1,
                    help="Records per request (grounding+contradiction together). >1 beats per-request rate limits.")
    ap.add_argument("--no-cache", action="store_true",
                    help="Disable the resumable per-model response cache.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only judge the first N records of each file (smoke test).")
    args = ap.parse_args()

    prov = PROVIDERS[args.provider]
    api_key = os.environ.get(prov["key_env"])
    if not api_key:
        raise SystemExit(f"Set {prov['key_env']} in .env / the environment first.")
    client = OpenAI(api_key=api_key, base_url=prov["base_url"], timeout=90.0, max_retries=0)

    effort = args.effort
    EXTRA = {}
    if args.provider != "openai":
        effort = ""                  # reasoning_effort is OpenAI-only
        EXTRA = {"temperature": 0}   # deterministic judging for open models
    if args.temperature is not None:
        EXTRA["temperature"] = args.temperature

    print(f"Provider={args.provider} model={args.model} "
          f"(base_url={prov['base_url'] or 'api.openai.com'})")
    init_cache(args.model, enabled=not args.no_cache)
    for path in args.inputs:
        if args.batch > 1:
            process_file_batched(path, args.model, effort, args.workers, args.batch, args.limit)
        else:
            process_file(path, args.model, effort, args.workers, args.limit)


if __name__ == "__main__":
    main()
