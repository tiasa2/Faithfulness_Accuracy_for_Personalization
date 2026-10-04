# Extras (not reported in the paper)

Further analyses from the project, kept for reference while we decide what to keep. They are not
needed to reproduce the paper, and some read the original working-folder file names
(e.g. `responses_reward_eval_full_Qwen_iter1_1.json` = `results/opinionqa/heldout_untrained.json.gz`,
`best_responses_reward_for_best_model.json` = `results/opinionqa/heldout_curriculum.json.gz`).

| Folder | Question | Summary of the finding |
|---|---|---|
| `survey_simulation/` | Do axioms or personas better preserve the real survey answer distribution (variance, regression coefficients, belief-answer associations) and counter-stereotypical respondents? | `results/survey_sim/summary.txt`; tables in `results/` |
| `monitorability/` | Does training change how often the reasoning discloses a preference cue (MonitorBench) or a stored user attribute (OpenAI memory_advice)? | `RESULTS.md`: no change in behaviour; disclosure unchanged at ceiling |
| `sft_baseline/` | Does SFT on the base model's own high-reward outputs match DPO? | `RESULTS.md`: SFT is flat or negative on faithfulness |
