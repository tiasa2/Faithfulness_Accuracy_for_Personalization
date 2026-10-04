# SFT baseline (trained on the untrained base model's own outputs) — OpinionQA eval

Source: `sft_monitorbench/analysis_summary_sft*_*.json` (1500 items each, our judge).
Comparison rows computed from `responses_reward_eval_full_Qwen_iter1_1.json` (base) and
`best_responses_reward_for_best_model.json` (DPO curriculum, Qwen_iter4 best).

Setup (as described, not documented in the repo): sample responses from the untrained base
model `Qwen/Qwen3-8B` (`Qwen_iter1`) on the training prompts, keep the high-reward ones, and
fine-tune with a plain likelihood loss. No preference pairs, no on-policy re-generation. Two
variants, `sft` and `sftpipe`; the distinction is not recorded anywhere in the repo — define it
before either row appears in a table. One seed per configuration.

## Headline

| model | mean F | fully-faithful share (F=1) | correctness | C&F count | C&F rate |
|---|---|---|---|---|---|
| base (Qwen_iter1) | .922 | .723 | .511 | 535 | .357 |
| **DPO curriculum (iter4 best)** | **.942** | **.771** | **.521** | **601** | **.401** |
| sft lr1e-5 | .921 | .705 | .506 | 548 | .365 |
| sft lr2e-6 | .913 | .708 | .513 | 550 | .367 |
| sftpipe lr1e-5 | .898 | .654 | .495 | 501 | .334 |
| sftpipe lr2e-6 | .906 | .646 | .519 | 489 | .326 |
| sftpipe lr5e-7 | .902 | .620 | .511 | 473 | .315 |

Mean F = mean of rewards[0] = grounding × (1 − contradiction_rate), partial credit included.
Fully-faithful share = fraction of items with reward exactly 1.

## Quadrants (counts; Δ = change vs base)

| model | C&F | C&U | I&F | I&U | ΔC&F | ΔC&U | ΔI&F | ΔI&U | F given correct | F given incorrect |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 535 | 232 | 550 | 183 | — | — | — | — | .698 | .750 |
| DPO iter4 | 601 | 181 | 556 | 162 | +66 | −51 | +6 | −21 | .769 | .774 |
| sft lr1e-5 | 548 | 211 | 509 | 232 | +13 | −21 | −41 | +49 | .722 | .687 |
| sft lr2e-6 | 550 | 220 | 512 | 218 | +15 | −12 | −38 | +35 | .714 | .701 |
| sftpipe lr1e-5 | 501 | 242 | 480 | 277 | −34 | +10 | −70 | +94 | .674 | .634 |
| sftpipe lr2e-6 | 489 | 290 | 480 | 241 | −46 | +58 | −70 | +58 | .628 | .666 |
| sftpipe lr5e-7 | 473 | 293 | 457 | 277 | −62 | +61 | −93 | +94 | .617 | .623 |

## Reading

1. **Correctness is flat in every arm** (within ±1.6pp of base). Same ceiling as everywhere
   else in the project; the comparison is entirely on the faithfulness axis.
2. **No SFT configuration reaches the base model's faithfulness, let alone DPO's.** Best SFT is
   −1.5pp on fully-faithful share vs base; DPO is +4.8pp. Gap DPO vs best SFT ≈ 6pp on
   fully-faithful share, 51 items on C&F.
3. **DPO improves faithfulness unconditionally on correctness**: F|correct .70→.77 and
   F|incorrect .75→.77. Both unfaithful quadrants shrink (C&U −51, I&U −21).
4. **Plain `sft` trades the incorrect side for the correct side.** C&U drops 12–21 (small gain
   in grounding of correct answers) but I&F drops ~40 and I&U rises 35–49; F|incorrect falls
   .75→~.69. Consistent with the filtered training set being dominated by faithful-AND-correct
   responses: the model learns faithful reasoning as what correct answers look like, and has
   no template for staying grounded when its answer is wrong.
5. **`sftpipe` erodes faithfulness across the board**: both unfaithful quadrants grow by 60–90
   items at every learning rate, and the damage is largest at the *smallest* LR (5e-7). A
   monotone-in-the-wrong-direction LR sweep points at the data/format, not the optimizer.

## What it supports

The preference-pair design, not extra exposure to faithful reasoning, is the active ingredient:
same base model, same prompts, same reward function, same eval — imitating the base model's own
faithful outputs does not raise faithfulness (and lowers it on incorrect items), while contrasting
faithful against unfaithful outputs raises it on both sides at matched correctness.

## Caveats before it goes in the paper

- Single seed per configuration; the ±15 differences among SFT arms are within noise. DPO's +66
  C&F is not.
- Our judge only. Check direction with the GPT second judge (absolute F levels are judge-dependent,
  κ=0.08 on OpinionQA; only the trained>untrained direction has replicated across judges).
- Loss and data freshness are confounded: SFT here is one-shot off-policy on base-model data,
  DPO is iterative on-policy. The tighter control (SFT on the curriculum's chosen responses at
  each iteration) isolates the loss and is the planned "F and C separately vs curriculum" run.
- `sft` vs `sftpipe`: undocumented; needs a one-line definition.
