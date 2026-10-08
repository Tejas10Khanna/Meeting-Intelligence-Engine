"""Deterministic safety checks on LLM edits: numbers, negations, names, edit plausibility."""
import difflib
import re

from rapidfuzz import fuzz

NEG = {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "cannot", "without"}
NUMWORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
            "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
            "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
            "thousand", "million", "billion"}


def norm(tok):
    t = tok.lower().replace("’", "'")
    return re.sub(r"[^\w%$.'-]", "", t).strip(".'-")


def numbers_in(tokens):
    return sorted(norm(t) for t in tokens if re.search(r"\d", norm(t)) or norm(t) in NUMWORDS)


def negations_in(tokens):
    return sorted(norm(t) for t in tokens if norm(t) in NEG or norm(t).endswith("n't"))


def meaning_violations(raw, refined):
    """Whole-text check used for evaluation: which protected tokens differ?"""
    r, f = raw.split(), refined.split()
    out = []
    if numbers_in(r) != numbers_in(f):
        out.append("number_changed")
    if negations_in(r) != negations_in(f):
        out.append("negation_changed")
    return out


def _accept(r, f, glossary, protected, support, min_sim):
    if numbers_in(r) != numbers_in(f):
        return False, "number"
    if negations_in(r) != negations_in(f):
        return False, "negation"
    rj, fj = " ".join(r).lower(), " ".join(f).lower()
    if protected and any(p.lower() in rj and p.lower() not in fj for p in protected):
        return False, "protected_name"
    if not r:
        return (len(f) <= 2, "insertion" if len(f) > 2 else "ok")
    if not f:
        return (len(r) <= 2, "deletion" if len(r) > 2 else "ok")
    if len(f) > 2 * len(r) + 2:
        return False, "too_long"
    gl = [g.lower() for g in glossary]
    if any(g in fj for g in gl) and len(r) <= 4:
        return True, "glossary"
    if support and all(norm(t) in support for t in f):
        return True, "supported_by_other_asr"
    if fuzz.ratio(rj, fj) / 100.0 >= min_sim:
        return True, "similar"
    return False, "low_similarity"


def apply_guarded(raw, refined, glossary=(), protected=(), support=None, min_sim=0.5):
    """Apply an LLM rewrite hunk by hunk, keeping only edits that pass the checks.
    Returns (final_text, hunk_log)."""
    rt, ft = raw.split(), refined.split()
    sm = difflib.SequenceMatcher(None, [norm(t) for t in rt], [norm(t) for t in ft], autojunk=False)
    out, log = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        r, f = rt[i1:i2], ft[j1:j2]
        if tag == "equal":
            out += f
            continue
        ok, why = _accept(r, f, glossary, protected, support, min_sim)
        out += f if ok else r
        log.append({"raw": " ".join(r), "new": " ".join(f), "accepted": ok, "reason": why})
    return " ".join(out), log
