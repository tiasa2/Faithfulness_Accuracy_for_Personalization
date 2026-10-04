"""Faithfulness metric (paper Sec. 3.2, App. C).

    F = g * (1 - rho)   (F = 0 when g = 0 or there are no axioms)

g    grounding score: the judge rates 1-5 how much the reasoning reasons from the axioms, rescaled to [0, 1]
rho  contradiction rate: share of the person's axioms the reasoning directly contradicts (judge label "C")

The judge is Qwen3-32B with greedy decoding and outputs constrained to the label tokens.
"""
from .parsing import reasoning_text
from .templates import contradiction_prompt, grounding_prompt

JUDGE_MODEL = "Qwen/Qwen3-32B"


def axiom_lines(axioms):
    """Individual axioms of an axiom block (first line is the header)."""
    return [line for line in axioms.split("\n")[1:] if line.strip()]


def faithfulness(grounding, contradiction_labels):
    """(F, rho) from a rescaled grounding score and one A/B/C label per axiom."""
    n = len(contradiction_labels)
    if grounding == 0 or n == 0:
        return 0.0, 0.0
    rho = sum(1 for label in contradiction_labels if label.strip() == "C") / n
    return grounding * (1 - rho), rho


class FaithfulnessJudge:
    def __init__(self, model=JUDGE_MODEL, seed=42, **llm_kwargs):
        from vllm import LLM

        self.llm = LLM(model=model, tokenizer=model, dtype="bfloat16", load_format="auto", seed=seed, **llm_kwargs)

    def _choose(self, prompts, choices):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        params = SamplingParams(temperature=0, top_p=1, max_tokens=1, seed=0, n=1,
                                structured_outputs=StructuredOutputsParams(choice=choices))
        return [out.outputs[0].text for out in self.llm.generate(prompts, params)]

    def score(self, items):
        """items: list of (axiom block, model response). Returns one dict per item."""
        g_prompts, c_prompts, spans = [], [], []
        for axioms, response in items:
            reasoning = reasoning_text(response)
            g_prompts.append(grounding_prompt(axioms, reasoning))
            lines = axiom_lines(axioms)
            spans.append((len(c_prompts), len(lines)))
            c_prompts.extend(contradiction_prompt(axiom, reasoning) for axiom in lines)

        c_labels = self._choose(c_prompts, ["A", "B", "C"])
        g_scores = [(int(r) - 1) / 4 for r in self._choose(g_prompts, ["1", "2", "3", "4", "5"])]

        results = []
        for g, (start, n) in zip(g_scores, spans):
            labels = c_labels[start:start + n]
            reward, rho = faithfulness(g, labels)
            results.append({"grounding": g, "contradiction_labels": labels, "contradiction_rate": rho, "reward": reward})
        return results
