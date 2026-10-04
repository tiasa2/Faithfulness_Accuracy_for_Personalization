"""OpinionQA respondents -> options, persona text and axiom text (paper Sec. 3.3, App. A).

Each respondent row holds 15 survey answers (one column per Pew question ID) and a demographic
profile. For a target question, the persona is the demographic profile and the axioms are the
respondent's answers to the other 14 questions, recast as belief statements.
"""
import ast
import hashlib
import json
import random

import pandas as pd

OPTION_LETTERS = ["A", "B", "C", "D", "E"]
PERSONA_HEADER = "Persona description :\n"
AXIOM_HEADER = "The following axioms represent the persona’s beliefs or preferences:\n"
# demographic field -> label shown in the persona (RELIGATTEND is not shown)
PERSONA_FIELDS = {
    "CREGION": "Region",
    "AGE": "Age",
    "SEX": "sex",
    "EDUCATION": "Education",
    "CITIZEN": "Are they a citizen?",
    "MARITAL": "Marital Status",
    "RELIG": "Religion",
    "POLPARTY": "Political Affiliation",
    "INCOME": "Income",
    "POLIDEOLOGY": "Strength of political ideology",
    "RACE": "Race",
}


def load_respondents(path):
    """Respondent table (data/opinionqa/*.csv), re-indexed 0..n-1."""
    return pd.read_csv(path, index_col=0).reset_index(drop=True)


def question_keys(df):
    return [c for c in df.columns if c != "Persona"]


def cell(df, key, i):
    """{'question', 'option_mapping', 'response'} for respondent i and question key."""
    return ast.literal_eval(df[key][i])


def options_text(option_mapping):
    """'A: ...\nB: ...\n' from the option list, skipping 'Refused'."""
    if isinstance(option_mapping, str):
        option_mapping = ast.literal_eval(option_mapping)
    labels = [o for o in option_mapping if o != "Refused"]
    return "".join(f"{OPTION_LETTERS[i]}: {o}\n" for i, o in enumerate(labels))


def options_dict(text):
    """{'A': 'Very confident', ...} from options_text output."""
    out = {}
    for line in text.splitlines():
        if len(line) > 2 and line[0] in OPTION_LETTERS and line[1] in ":)":
            out[line[0]] = line[2:].strip()
    return out


def persona_text(df, i):
    persona = PERSONA_HEADER
    for field, value in ast.literal_eval(df["Persona"][i]).items():
        if field in PERSONA_FIELDS and value == value:  # value == value drops NaN
            persona += f"{PERSONA_FIELDS[field]}: {value}\n"
    return persona


def axiom_sentence(question, response):
    """Recast one survey answer as a belief statement (Table 6)."""
    table = _axiom_table(response)
    if question not in table:
        raise KeyError(f"no axiom template for question: {question}")
    templates = table[question]
    return templates[response] if response in templates else templates["other"]


def axioms_text(df, i, target, keep=None, source_keys=None):
    """Axiom block for respondent i when predicting `target`.

    By default every other answered question becomes an axiom (column order). `keep(key)` filters
    them (ablations); `source_keys` gives an explicit ordered list instead (Relevant_Axioms setup).
    """
    keys = source_keys if source_keys is not None else question_keys(df)
    axioms = AXIOM_HEADER
    for key in keys:
        if key == target or key == "Persona" or key not in df.columns:
            continue
        if keep is not None and not keep(key):
            continue
        c = cell(df, key, i)
        if c["response"] != "Refused":
            axioms += axiom_sentence(c["question"], c["response"]) + "\n"
    return axioms


def load_relevance_map(path):
    """target key -> ordered list of relevant source keys (data/opinionqa/relevant_axioms_*.json)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for entry in data["keys"]:
        seen = []
        for item in entry.get("relevant_axioms", []):
            if item.get("source_key") and item["source_key"] not in seen:
                seen.append(item["source_key"])
        out[entry["key"]] = seen
    return out


def ablation_filter(mode, relevance, df, i, target, seed=42):
    """keep(key) for the axiom-relevance ablation (Sec. 5.2, Table 11).

    all            every axiom
    minus_relevant drop the axioms relevant to `target`
    only_relevant  keep only the relevant axioms
    minus_random   drop as many randomly chosen *irrelevant* answered axioms (seeded per respondent/target)
    """
    relevant = set(relevance.get(target, []))
    if mode == "all":
        return None
    if mode == "minus_relevant":
        return lambda key: key not in relevant
    if mode == "only_relevant":
        return lambda key: key in relevant
    if mode == "minus_random":
        candidates = [k for k in question_keys(df)
                      if k != target and k not in relevant and cell(df, k, i)["response"] != "Refused"]
        h = int(hashlib.md5(f"{seed}-{target}-{i}".encode()).hexdigest(), 16)
        drop = set(random.Random(h).sample(candidates, min(len(relevant), len(candidates))))
        return lambda key: key not in drop
    raise ValueError(f"unknown ablation mode: {mode}")


def _axiom_table(response):
    # question -> {answer: sentence, 'other': template using the answer}; verbatim from the paper's runs
    return {'How confident are you, if at all, that companies will follow what their privacy policies say they will do with your personal information?': {'other': f"The person is {response.lower()} that companies will follow what their privacy policies say they will do with personal information"},
 'How comfortable are you, if at all, with companies using your personal data to help them develop new products?': {'other': f"The person is {response.lower()} with companies using their personal data to help them develop new products"},
 'Have you recently decided not to use a product or service because you were worried about how much personal information would be collected about you?': {'Yes, have done this': "The person has recently decided not to use a product or service because they were worried about how much personal information would be collected about them", 'No, have not done this': "The person has never not used a product or service because they were worried about how much personal information would be collected about them"},
 'Still thinking ahead 30 years, which do you think is more likely to happen in the U.S.? The public education system will improve/get worse': {'other': f"The person thinks that {response.lower()} in the next 30 years in the U.S."},
 "Still thinking about the future of our country, how worried are you, if at all, about the ability of political leaders to solve the country's biggest problems?": {'other': f"The person is {response.lower()} about the ability of political leaders to solve the country's biggest problems in the future."},
 'In order to maintain the strength of the U.S. economy over the next 30 years, do you think that legal immigration will need to be': {'other': f"The person thinks that in order to maintain the strength of the U.S. economy over the next 30 years, legal immigration will need to be {response.lower()}"},
 'How much do you trust the accuracy of the news and information that you get from political leaders and public officials': {'A great deal': "The person trusts the accuracy of the news and information that they get from political leaders and public officials", 'Some': "The person somewhat trusts the accuracy of the news and information that they get from political leaders and public officials", 'Not much': "The person does not much trust the accuracy of the news and information that they get from political leaders and public officials", 'Not at all': "The person does not trust at all the accuracy of the news and information that they get from political leaders and public officials"},
 'How do you feel about your own ability to recognize the following type of news and information? A video or image that is altered or made up to mislead the public': {'I find it easy to recognize it': "The person trusts their own ability to recognize a video or image that is altered or made up to mislead the public", 'It find it hard to recognize it': "The person does not trust their own ability to recognize a video or image that is altered or made up to mislead the public"},
 'Some people think that by criticizing leaders, news organizations keep political leaders from doing their job. Others think that such criticism is worth it because it keeps political leaders from doing things that should not be done. Which position is closer to your opinion?': {'Keep political leaders from doing their job': "The person thinks that by criticizing leaders, news organizations keep political leaders from doing their job.", "Keep political leaders from doing things that shouldn't be done": "The person thinks that by criticizing leaders, news organizations keep political leaders from doing things that should not be done"},
 "In general, would you say the following statements describes most research scientists well? Don't pay attention to the moral values of society": {'Describes research scientists well': "The person believes that saying research scientists do not pay attention to the moral values of society is accurate", 'Does not describe research scientists well': "The person believes that saying research scientists do not pay attention to the moral values of society is not accurate"},
 'In general, do you think most of the important policy decisions in this country are made with': {'A great deal of concern for how the policies affect people like you': "The person thinks most of the important policy decisions in this country are made with a great deal of concern for how the policies affect people like them", 'Some concern for how the policies affect people like you': "The person thinks most of the important policy decisions in this country are made with some concern for how the policies affect people like them", 'Not too much concern for how the policies affect people like you': "The person thinks most of the important policy decisions in this country are made with not too much concern for how the policies affect people like them", 'No concern at all for how the policies affect people like you': "The person thinks most of the important policy decisions in this country are made with no concern at all for how the policies affect people like them"},
 "Which of these do you think is the better way to solve the country's most pressing problems, even if neither is exactly right?": {'other': f"The person thinks the better way to solve the country's most pressing problems, even if neither is exactly right, is to {response.lower()}."},
 'How often, if ever, does race or race relations come up in your conversations with family and friends?': {'other': f"In this person's life, race or race relations have {response.lower()} come up in their conversations with family and friends"},
 'In general, do you think there is too much, too little, or about the right amount of attention paid to race and racial issues in our country these days?': {'other': f"The person thinks that {response.lower()} is being paid to race and racial issues in their country these days"},
 'When it comes to racial discrimination, which do you think is the bigger problem for the country today?': {'other': f"The person thinks that when it comes to racial discrimination, the bigger problem for the country today is {response.lower()}."}
 }
