"""One canonical table for the modal / non-modal ("stereotype") accuracy numbers.

Untrained Qwen3-8B, 4 prompt conditions (final_trained_results/responses_reward_0to500_{0,1,2}.json).
A cell = (respondent, question). "Modal" = the respondent's TRUE answer equals their group's modal
answer (non-modal = counter-stereotypical). Ties excluded.

Variants (every number is labelled with all of them):
  sample   : all500 (3 runs pooled; modal leave-one-out from the other 499)
             test100 (the DPO held-out respondents; modal from the 400 train respondents)
  grouping : POLPARTY, POLIDEOLOGY, PARTYxIDEO
  metric   : acc_nonmodal  = P(correct | non-modal cell)          (conditional accuracy)
             acc_modal     = P(correct | modal cell)
             C&NM_share    = P(correct AND non-modal) over ALL cells (unconditional share)
             stereo_follow = P(predicts group mode | non-modal cell)
Contrasts with respondent-clustered bootstrap CIs (B=1000): persona-base, axioms-base,
P+A-persona, P+A-axioms. Also mean contradiction rate per condition (judge vs the person's own
axioms, which are scored for every condition even when not shown in the prompt).

Output: canonical_nonmodal_table.csv, canonical_contradiction_by_condition.csv
"""
import numpy as np
import pandas as pd

import survey_simulation_analysis as S

COND = S.COND4
B = 1000


def main():
    rng = np.random.default_rng(0)
    D = S.load_real()
    N = len(D["pers"]); all_idx = np.arange(N); test_idx = D["test_idx"]
    train_idx = np.setdiff1d(all_idx, test_idx)
    runs = []
    for k in range(3):
        pr, F, G, CR, C = S.from_reward_file(D, f"final_trained_results/responses_reward_0to500_{k}.json", 4)
        runs.append((pr, F, G, CR, C))
    real = D["real"]
    rows, crow = [], []
    for sample, idx, modal_from in (("all500", all_idx, all_idx), ("test100", test_idx, train_idx)):
        for grp in ("POLPARTY", "POLIDEOLOGY", "PARTYxIDEO"):
            gm = S.baseline_group_modal(D, modal_from, idx, grp, min_pool=5)
            # per-respondent sums so the bootstrap resamples respondents
            stats = {}
            for c_i, c in enumerate(COND):
                acc = []
                for pr, F, G, CR, C in runs:
                    t, m, p = real[idx], gm[idx], pr[c_i][idx]
                    ok = (t >= 0) & (m >= 0) & (p >= 0)
                    nm, mo = ok & (t != m), ok & (t == m)
                    cor = (p == t)
                    acc.append(np.stack([(cor & nm).sum(1), nm.sum(1), (cor & mo).sum(1), mo.sum(1),
                                         ((p == m) & nm).sum(1), ok.sum(1)], 1))
                stats[c] = np.sum(acc, 0)          # pool runs: (n_resp x 6)
            def metrics(a):
                return dict(acc_nonmodal=a[0] / a[1], acc_modal=a[2] / a[3], **{"C&NM_share": a[0] / a[5]},
                            stereo_follow=a[4] / a[1], nonmodal_share=a[1] / a[5])
            draws = [rng.integers(0, len(idx), len(idx)) for _ in range(B)]
            for c in COND:
                pt = metrics(stats[c].sum(0))
                for k_, v in pt.items():
                    rows.append(dict(sample=sample, grouping=grp, kind="level", comparison=c, metric=k_, value=v))
            for a_, b_ in (("persona", "baseline"), ("axioms", "baseline"), ("persona+axioms", "persona"),
                           ("persona+axioms", "axioms"), ("axioms", "persona")):
                pa, pb = metrics(stats[a_].sum(0)), metrics(stats[b_].sum(0))
                bs = {k_: [] for k_ in pa}
                for d in draws:
                    ma, mb = metrics(stats[a_][d].sum(0)), metrics(stats[b_][d].sum(0))
                    for k_ in pa:
                        bs[k_].append(ma[k_] - mb[k_])
                for k_ in pa:
                    v = np.array(bs[k_])
                    rows.append(dict(sample=sample, grouping=grp, kind="contrast", comparison=f"{a_} - {b_}",
                                     metric=k_, value=pa[k_] - pb[k_], lo=np.percentile(v, 2.5),
                                     hi=np.percentile(v, 97.5)))
        for c_i, c in enumerate(COND):
            cr = np.concatenate([r[3][c_i][idx].ravel() for r in runs])
            g = np.concatenate([r[2][c_i][idx].ravel() for r in runs])
            f = np.concatenate([r[1][c_i][idx].ravel() for r in runs])
            crow.append(dict(sample=sample, condition=c, mean_F=np.nanmean(f), mean_grounding=np.nanmean(g),
                             mean_contradiction_rate=np.nanmean(cr), share_any_contradiction=np.nanmean(cr > 0)))
    pd.DataFrame(rows).to_csv("extras/survey_simulation/results/canonical_nonmodal_table.csv", index=False)
    pd.DataFrame(crow).to_csv("extras/survey_simulation/results/canonical_contradiction_by_condition.csv", index=False)
    print("wrote canonical_nonmodal_table.csv, canonical_contradiction_by_condition.csv")


if __name__ == "__main__":
    main()
