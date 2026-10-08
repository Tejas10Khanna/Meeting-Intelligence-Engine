"""Cross-system agreement: where two ASR systems disagree, the audio is likely hard (Path 5)."""
import difflib
import re


def norm_tokens(text):
    return re.sub(r"[^\w\s']", " ", text.lower().replace("’", "'")).split()


def words_in_range(words, start, end, pad=0.3):
    return [w for w in words if start - pad <= (w.start + w.end) / 2 <= end + pad]


def disagreement(a, b):
    ta, tb = norm_tokens(a), norm_tokens(b)
    if not ta and not tb:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()


def annotate(units, alt_words, name="parakeet"):
    """Attach the other system's reading of each unit and its disagreement score."""
    for u in units:
        alt = " ".join(w.text for w in words_in_range(alt_words, u.start, u.end))
        u.alt_texts[name] = alt
        u.disagreement = max(u.disagreement, disagreement(u.text, alt))
    return units


def flag_units(units, cfg):
    """Mark units that deserve LLM attention. Thresholds are tuned on the dev set."""
    lp, dis = cfg.get("logprob_thr", -0.45), cfg.get("disagree_thr", 0.12)
    low_word = cfg.get("low_word_prob", 0.4)
    for u in units:
        r = []
        if u.avg_logprob and u.avg_logprob < lp:
            r.append("low_logprob")
        if u.words and sum(w.prob < low_word for w in u.words) >= max(1, int(0.1 * len(u.words))):
            r.append("low_word_conf")
        if u.disagreement > dis:
            r.append("asr_disagreement")
        if u.overlap:
            r.append("overlap")
        u.flag_reasons, u.flagged = r, bool(r)
    has_conf = any(u.avg_logprob != 0 for u in units) or any(w.prob < 1.0 for u in units for w in u.words) \
        or any(u.disagreement > 0 for u in units)
    if units and not has_conf and not any(u.flagged for u in units):
        # e.g. Hugging Face Whisper backend or Parakeet: no confidence values exist, so the gate must not
        # silently skip every unit. Fall back to refining everything (the edit guard still protects the text).
        for u in units:
            u.flagged, u.flag_reasons = True, ["no_confidence_info"]
    return units
