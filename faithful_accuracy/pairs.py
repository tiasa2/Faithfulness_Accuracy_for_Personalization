"""Curriculum preference pairs (paper Sec. 5.1, App. G.3).

For each prompt there is a pool of candidate responses, each with correctness (0/1) and reward F.

iteration 1   faithfulness objective: chosen = highest-reward correct response,
              rejected = lowest-reward correct response (correct but unfaithful)
iteration 2+  combined objective: chosen = highest-reward response, which must be correct;
              rejected = lowest-reward response, which must be incorrect

A pair is kept only if chosen reward - rejected reward >= MIN_REWARD_SPREAD.
"""
MIN_REWARD_SPREAD = 0.2


def select_pair(correctness, rewards, iteration, min_spread=MIN_REWARD_SPREAD):
    """(chosen index, rejected index), or None if the prompt yields no pair."""
    rewards = [float(r) for r in rewards]
    if iteration == 1:
        correct = [i for i, c in enumerate(correctness) if int(c) == 1]
        if len(correct) < 2:
            return None
        hi = max(correct, key=lambda i: rewards[i])
        lo = min(correct, key=lambda i: rewards[i])
    else:
        hi = max(range(len(rewards)), key=lambda i: rewards[i])
        lo = min(range(len(rewards)), key=lambda i: rewards[i])
        if int(correctness[hi]) != 1 or int(correctness[lo]) != 0:
            return None
    if rewards[hi] - rewards[lo] < min_spread:
        return None
    return hi, lo


def build_pairs(records, iteration, eot_token="", margin_scale=1.0, min_spread=MIN_REWARD_SPREAD):
    """Columns for a DPO dataset from labelled records (scripts/judge.py output)."""
    from .parsing import response_to_text

    cols = {k: [] for k in ["prompt", "chosen", "rejected", "margin", "chosen_score", "rejected_score", "chosen_pos", "chosen_neg"]}
    for rec in records:
        pair = select_pair(rec["correctness"], rec["rewards"], iteration, min_spread)
        if pair is None:
            continue
        hi, lo = pair
        cols["prompt"].append(rec["prompt"])
        cols["chosen"].append(response_to_text(rec["responses"][hi]) + eot_token)
        cols["rejected"].append(response_to_text(rec["responses"][lo]) + eot_token)
        cols["margin"].append((float(rec["rewards"][hi]) - float(rec["rewards"][lo])) * margin_scale)
        cols["chosen_score"].append(float(rec["rewards"][hi]))
        cols["rejected_score"].append(float(rec["rewards"][lo]))
        cols["chosen_pos"].append(int(rec["correctness"][hi]))
        cols["chosen_neg"].append(int(rec["correctness"][lo]))
    return cols
