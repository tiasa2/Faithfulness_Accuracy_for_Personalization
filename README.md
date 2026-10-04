# Faithfulness and Predictive Accuracy in LLM Reasoning for Personalization

Code, data and results for the paper *Investigating Faithfulness and Predictive Accuracy in LLM
Reasoning for Personalization* (anonymous submission).

A model reasoning on someone's behalf should predict their answer (**predictive accuracy**), but its
reasoning should also follow from what that person believes (**faithfulness**). We build personalized
multiple-choice tasks from OpinionQA, where each respondent is described by a demographic **persona**
and by **axioms**: their answers to the other survey questions, written as belief statements. We
measure both axes and train Qwen3-8B with **curriculum DPO**, which teaches grounding first and
accuracy second.

**Faithfulness** `F = g * (1 - rho)`: `g` is how much the reasoning draws on the axioms (1-5, rescaled
to 0-1) and `rho` the share of axioms it contradicts, both judged by Qwen3-32B. `F = 0` when `g = 0`.

## Repository

```
faithful_accuracy/   library: personas and axioms, prompt templates, faithfulness judge, curriculum pairs, DPO trainer
scripts/             one script per pipeline stage (generate, judge, train, evaluate) + PersonaMem, PrefEval
analysis/            reproduce every table and figure from results/
data/                OpinionQA respondents, axiom relevance maps, human ratings, PrefEval items
results/             all model outputs behind the paper (see results/README.md), tables/, figures/
extras/              analyses that are not in the paper
```

## Reproduce the paper's tables and figures

CPU only, about a minute:

```bash
pip install pandas matplotlib
python analysis/reproduce_tables.py      # Tables 3, 4, 5, 9, 10, 11, 12, 13, 14
python analysis/human_eval.py            # Tables 1, 2, 7, 8 and the human-agreement column of Table 9
python analysis/make_figures.py          # Figures 2-5
```

Each value is printed next to the paper's, and tables are saved to `results/tables/`.

## Run the pipeline

GPU, vLLM. Run from the repository root.

```bash
pip install -r requirements.txt

# held-out evaluation of any model: generate -> judge -> summary
bash scripts/evaluate.sh Qwen/Qwen3-8B runs/untrained

# curriculum DPO, 3 iterations (rollouts -> judge -> DPO -> evaluation each iteration)
bash scripts/run_curriculum.sh           # trained model: runs/curriculum/iter3/check_point_curriculum
```

| Stage | Script | Notes |
|---|---|---|
| Generation | `scripts/generate.py` | `--mode setups` (five prompting setups; training rollouts and Table 3) or `--mode eval` (persona + axioms; held-out evaluation). `--axioms minus_relevant\|only_relevant\|minus_random` runs the Sec. 5.2 ablation. |
| Judging | `scripts/judge.py` | correctness + F with Qwen3-32B |
| Training | `scripts/train_dpo.py` | one DPO iteration; iteration 1 rejects correct-but-unfaithful responses, later iterations reject incorrect ones (`faithful_accuracy/pairs.py`) |
| Second judges | `scripts/judge_api.py` | GPT-5, GPT-5-mini, GPT-4o-mini, GLM-4.7-Flash (Table 9) |
| NLI baseline | `scripts/nli_baseline.py` | DeBERTa-v3 NLI faithfulness (Tables 1, 2, 8) |
| PersonaMem-v2 | `scripts/personamem/` | `generate.py` then `judge.py` |
| PrefEval | `scripts/prefeval/run.sh <checkpoint>` | needs `git clone https://github.com/amazon-science/PrefEval` (commit `5079505`) and API keys in `.env` |

Settings used in the paper:
- **Rollouts:** temperature 0.6 with Qwen3 thinking off.
- **Evaluation:** greedy decoding with the default chat template.
- **DPO:** lr 5e-7 (cosine), beta 0.1, effective batch 4, one epoch per iteration, minimum reward gap
  0.2, on one H200.

## Acknowledgements

`faithful_accuracy/dpo_trainer.py` is adapted from [RLHFlow Online-DPO-R1](https://github.com/RLHFlow/Online-DPO-R1)
(MIT). Data: OpinionQA (Santurkar et al., 2023), PersonaMem-v2 (Jiang et al., 2025), PrefEval (Zhao et al., 2025).

## License

MIT
