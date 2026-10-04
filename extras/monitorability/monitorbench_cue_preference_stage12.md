# MonitorBench `cue_condition.preference` — stages 1–2 (generation + rule-based verification)

Run 2026-09-12, OOD interactive node gh126 (1x H200), vLLM 0.15.1, Qwen3 thinking ON,
108 items x 8 rollouts per condition (n=864 rollouts/cell, n=108 paired items).
Verification is rule-based (`deepscaler_reward_fn` on the `\boxed{}` letter) — no judge.

Models: base `Qwen/Qwen3-8B` vs trained `Qwen_iter3/check_point_curriculum`
(the SAME checkpoint the PrefEval trained arm used — verified via `model_path` in
`prefeval_trained_curriculum_*.json`; note `iteration_no=Qwen_iter4` there is a label only).
Staged weights verified distinct: 0/108 identical responses in every condition.

## Gold rate (= picks the preference-consistent option)

| condition        | base    | trained | Δ (pp)  | 95% CI        | perm p |
|------------------|---------|---------|---------|---------------|--------|
| original (no cue)| 36.23%  | 36.69%  | +0.46   | [-1.52,+2.45] | 0.732  |
| standard (cue)   | 57.87%  | 57.87%  |  0.00   | [-1.53,+1.53] | 1.000  |
| direct_conceal   | 59.61%  | 57.52%  | -2.08   | [-4.24,+0.07] | 0.071  |
| monitor_aware    | 59.26%  | 58.45%  | -0.81   | [-2.66,+1.04] | 0.463  |

The 500/864 tie on `standard` is a coincidence, not a fallback (responses differ 108/108).

## Cue total effect (standard − original)

| model   | TE       | 95% CI          | perm p  |
|---------|----------|-----------------|---------|
| base    | +21.64pp | [+12.90,+30.39] | 1e-5    |
| trained | +21.18pp | [+12.83,+29.53] | 1e-5    |
| **difference (trained − base)** | **−0.46pp** | [−3.07,+2.15] | **0.794** |

## Reading

1. The cue strongly moves both models (~+21.5pp, p=1e-5). This is the **precondition** for the
   benchmark: the monitor stage only has something to detect where the cue demonstrably changed
   the answer. Protocol sanity check passes.
2. **Training has no effect on the behavioral axis** — TE difference −0.46pp, p=.79, CI excludes
   anything larger than ±3pp. Consistent with the established finding that DPO moves the
   faithfulness axis, not raw answer behavior. This is a null, and an expected one.
3. No per-condition difference survives correction (direct_conceal p=.071 over 4 tests).
4. Neither model reduces cue-following when told to conceal it or told it is monitored
   (all within-model stress deltas ns, |Δ|<=1.7pp). Whether the *CoT admits* the cue is a
   different question — that is the monitorability score, NOT measured here.

## NOT measured: the monitorability score

Stages 1–2 measure only whether the cue changed the ANSWER. The paper-relevant quantity —
does the CoT acknowledge the cue drove the decision — requires the Qwen3-32B monitor stage
(`prompts/monitor_cue_condition.preference.yaml`, scopes action/cot/all). That did not fit in
this session's 1-GPU/3.5h budget. The `.tested.jsonl` files written here are exactly what that
stage resumes from: rerun `main.py` with the same `--output-base-dir` and generation is skipped.

## Environment fixes applied (needed for any future run)

- `vllm` conda env was missing `defusedxml` and `joblib` (nltk transitive deps, imported via
  `pipeline/outcome.py` -> `registry.py`, so they break every task). Installed.
- `pipeline/generation/run.py` imported `batch_api_run` (anthropic/openai/google-genai) at
  module scope even under `backend=vllm`. Guarded the import, re-raising only under
  `backend=="api"`. Backup at `run.py.bak`. `google-genai` deliberately NOT installed — it would
  risk bumping pydantic/protobuf in a working vLLM env for a code path we never execute.
- `run_monitorbench.sh`: monitor/verification `tensor_parallel` now clamped to available GPUs
  (`MONITOR_TP`), since TP=2 is unsatisfiable on a 1-GPU interactive node.

## Follow-up: are the two negative deltas a problem? (No.)

`direct_conceal` −2.08pp and `monitor_aware` −0.81pp (trained vs base) are not a degradation:

1. **Neither survives correction.** Holm over the 4 condition tests: direct_conceal p=.071 vs
   threshold .0125 → ns; all four ns. CI on direct_conceal [−4.24,+0.07] includes 0.
2. **The gap is the BASE model moving, not the trained model dropping.** Against each model's
   own `standard`: base 57.87→59.61 (**+1.74pp**), trained 57.87→57.52 (**−0.35pp**, flat).
   Difference-in-differences: direct_conceal −2.08pp p=.169; monitor_aware −0.81pp p=.614.
3. **Lower is not obviously worse in this cell.** `direct_conceal` instructs the model to hide
   that the cue influenced it. The base model responds by following the cue MORE while under
   instruction to conceal that influence — behavior maintained, disclosure suppressed, which is
   the exact signature MonitorBench exists to detect. The trained model is invariant to the
   conceal framing. Invariance is the better property here.

Caveat: all of the above is BEHAVIOR only. Whether base's +1.74pp under conceal is accompanied
by a CoT that stops mentioning the cue is the monitorability question — unmeasured until the 32B
stage. **Flag for that run:** check this cell specifically rather than treating the monitor stage
as a uniform sweep; if the pattern replicates on the CoT axis it stops being noise.
