import spacy
import re

# Load spaCy model lazily with fallback to a no-op parser if unavailable
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:  # model missing in some environments
    nlp = None
    _spacy_load_error = str(e)

def parse_query(query):
    doc = nlp(query) if nlp is not None else None
    age = None
    gender = None
    location = None
    procedure = None
    duration = None

    # Age
    age_match = re.search(r"\b(\d{1,3})\b", query)
    if age_match:
        age = int(age_match.group(1))

    # Gender (anywhere in the string)
    gender_match = re.search(r"\b(male|female|m|f)\b", query, re.I)
    if gender_match:
        g = gender_match.group(1).lower()
        if g in ["m", "male"]:
            gender = "male"
        elif g in ["f", "female"]:
            gender = "female"

    # Location
    if doc is not None:
        for ent in doc.ents:
            if ent.label_ == "GPE":
                location = ent.text
    # Fallback location from comma-separated tokens if spaCy missed it
    if location is None:
        # Try to identify a token (comma-separated) that isn't age/gender/procedure/duration
        tokens = [t.strip() for t in query.split(',') if t.strip()]
        for tok in tokens:
            if re.fullmatch(r"\d{1,3}", tok):
                continue
            if re.search(r"\b(male|female|m|f)\b", tok, re.I):
                continue
            if re.search(r"\b(months?|years?)\b", tok, re.I):
                continue
            if any(p in tok.lower() for p in ["surgery", "treatment", "operation", "therapy"]):
                continue
            # Accept alphabetic tokens (including spaces and hyphens); normalize casing
            if re.fullmatch(r"[A-Za-z][A-Za-z\-\s]*", tok):
                location = tok.strip().title()
                break

    # Procedure (simple heuristic)
    procedures = ["surgery", "treatment", "operation", "therapy"]
    for proc in procedures:
        if proc in query.lower():
            procedure = proc

    # Policy duration
    dur_match = re.search(r"\b(\d+)\s*[- ]?\s*(month|months|year|years)\b", query, re.I)
    if dur_match:
        num = dur_match.group(1)
        unit = dur_match.group(2).lower()
        # normalize to plural
        if unit.startswith("month"):
            unit = "months" if int(num) != 1 else "month"
        elif unit.startswith("year"):
            unit = "years" if int(num) != 1 else "year"
        duration = f"{num} {unit}"

    return {
        "age": age,
        "gender": gender,
        "location": location,
        "procedure": procedure,
        "duration": duration
    }