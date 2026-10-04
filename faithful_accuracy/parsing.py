"""Parsing model responses of the form `### Selected Answer: ...` / `### Reasoning: ...`."""
import re


def response_to_text(response):
    """Some generations were saved as parsed dicts; turn them back into the raw text format."""
    if isinstance(response, dict):
        answer = response.get("Selected Answer", "").strip()
        reasoning = response.get("Reasoning", "").strip()
        return f"### Selected Answer: {answer}\n### Reasoning: {reasoning}"
    return response or ""


def answer_text(response):
    match = re.search(r"### Selected Answer:\s*(.*)", response_to_text(response), re.DOTALL)
    return match.group(1).strip() if match else ""


def reasoning_text(response):
    match = re.search(r"### Reasoning:\s*(.*)", response_to_text(response), re.DOTALL)
    return match.group(1).strip() if match else ""


def first_capital(text):
    for char in text or "":
        if char.isupper():
            return char
    return None


def selected_letter(response):
    """The option letter the model chose: first capital letter after `### Selected Answer:`."""
    return first_capital(answer_text(response))
