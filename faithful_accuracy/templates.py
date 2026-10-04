"""Prompt texts used in the paper's runs, copied verbatim from the original scripts.

Do not reformat: the indentation and trailing spaces inside these strings are part of the prompts
the models saw. `setup_prompts` (training rollouts and Table 3) and `eval_prompt` (held-out
evaluation, App. B) differ only in whitespace.
"""

SETUPS = ["Base", "Persona", "Axioms", "Persona_and_Axioms", "Relevant_Axioms"]


def setup_prompts(question, options, persona, axioms, relevant_axioms):
  """The five prompting setups (paper Sec. 4.1; Relevant_Axioms is used only as a training candidate)."""
  p_base = f"""Please read the following multiple-choice question carefully, select ONE of the listed options and explain your reasoning.

  Structure your response as:
  ### Question: [Insert question here]
  ### Options: [Insert options here]
  ### Selected Answer: [A/B/C/D...]
  ### Reasoning: [Provide reasoning for your answer]

  Question: {question}
  Options: {options}
    """
  p_persona = f"""Please read the following multiple-choice question carefully. From the perspective of the given persona, select ONE of the listed options and explain your reasoning.

  Structure your response as:
  ### Question: [Insert question here]
  ### Options: [Insert options here]
  ### Persona: [Insert persona description here]
  ### Selected Answer: [A/B/C/D...]
  ### Reasoning: [Explain your answer based on the persona]

  Question: {question}
  Options: {options}
  Persona: {persona}
  """
  p_axioms = f"""Please read the following multiple-choice question carefully. Guided by the provided axioms, select ONE of the listed options and explain your reasoning.

  Structure your response as:
  ### Question: [Insert question here]
  ### Options: [Insert options here]
  ### Axioms: [List of axioms guiding the persona’s reasoning]
  ### Selected Answer: [A/B/C/D...]
  ### Reasoning: [Explain your answer based on the axioms]

  Question: {question}
  Options: {options}
  Axioms: {axioms}
  """
  p_persona_and_axioms = f"""Please read the following multiple-choice question carefully. From the perspective of the given persona and guided by the provided axioms, select ONE of the listed options and explain your reasoning.

  Structure your response as:
  ### Question: [Insert question here]
  ### Options: [Insert options here]
  ### Persona: [Insert persona description here]
  ### Axioms: [List of axioms guiding the persona’s reasoning]
  ### Selected Answer: [A/B/C/D...]
  ### Reasoning: [Explain your answer based on the persona and axioms]

  Question: {question}
  Options: {options}
  Persona: {persona}
  Axioms: {axioms}
  """
  p_relevant_axioms = f"""Please read the following multiple-choice question carefully. From the perspective of the given persona and guided by the provided axioms, select ONE of the listed options and explain your reasoning.

  Structure your response as:
  ### Question: [Insert question here]
  ### Options: [Insert options here]
  ### Persona: [Insert persona description here]
  ### Axioms: [List of axioms guiding the persona’s reasoning]
  ### Selected Answer: [A/B/C/D...]
  ### Reasoning: [Explain your answer based on the persona and axioms]

  Question: {question}
  Options: {options}
  Persona: {persona}
  Axioms: {relevant_axioms}
  """
  return {"Base": p_base, "Persona": p_persona, "Axioms": p_axioms, "Persona_and_Axioms": p_persona_and_axioms, "Relevant_Axioms": p_relevant_axioms}


def eval_prompt(question, options, persona, axioms):
    """Persona + axioms prompt used for held-out evaluation (App. B)."""
    return f"""Please read the following multiple-choice question carefully. From the perspective of the given persona and guided by the provided axioms, select ONE of the listed options and explain your reasoning.
    
    Structure your response as:
    ### Question: [Insert question here]
    ### Options: [Insert options here]
    ### Persona: [Insert persona description here]
    ### Axioms: [List of axioms guiding the persona’s reasoning]
    ### Selected Answer: [A/B/C/D...]
    ### Reasoning: [Explain your answer based on the persona and axioms]
    
    Question: {question}
    Options: {options}
    Persona: {persona}
    Axioms: {axioms}
    """


def grounding_prompt(axioms, reasoning_chain):
    """Faithfulness judge, grounding score 1-5 (App. C.1)."""
    return f"""You are evaluating how well a reasoning chain is grounded in the provided axioms.

    Axioms:
    {axioms}
    
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


def contradiction_prompt(axiom, reasoning):
    """Faithfulness judge, one axiom vs the reasoning: A support / B unrelated / C contradiction (App. C.2)."""
    return f"""You are checking for LOGICAL CONTRADICTIONS between an axiom and a reasoning chain.
        Axiom: {axiom}
        
        Reasoning: {reasoning}
        
        A CONTRADICTION means the reasoning makes a claim that DIRECTLY CONFLICTS with the axiom — both statements cannot be true at the same time.
        
        Examples of CONTRADICTION:
        - Axiom says "person trusts X" but reasoning says "person distrusts X"
        - Axiom says "person believes Y" but reasoning says "person does not believe Y"
        
        NOT a contradiction:
        - Reasoning doesn't mention the axiom at all
        - Reasoning says the axiom is "not relevant" or "doesn't apply"
        - Reasoning talks about a different topic than the axiom
        
        Your answer:
        - "A" if the reasoning SUPPORTS or USES this axiom
        - "B" if the reasoning is UNRELATED to or DOES NOT ADDRESS this axiom
        - "C" if the reasoning DIRECTLY CONTRADICTS this axiom (they cannot both be true)
        
        When in doubt, choose "B". Only choose "C" for clear logical conflicts.
        
        Respond with only "A", "B", or "C"."""


def record_prompt(question, options, persona, axioms):
    """Prompt stored with each labelled record and used as the DPO prompt."""
    return f"""Please read the following multiple-choice question carefully. From the perspective of the given persona and guided by the provided axioms, select ONE of the listed options and explain your reasoning.
    
    Structure your response as:
    ### Question: [Insert question here]
    ### Options: [Insert options here]
    ### Persona: [Insert persona description here]
    ### Axioms: [List of axioms guiding the persona’s reasoning]
    ### Selected Answer: [A/B/C/D...]
    ### Reasoning: [Explain your answer based on the persona and axioms]
    
    Question: {question}
    Options: {options}
    Persona: {persona}
    Axioms: {axioms}
    """
