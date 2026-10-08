"""Evaluation metrics for every rubric area. Pure functions (unit-tested)."""
import re

import jiwer
from rapidfuzz import fuzz

from mtg.guardrails import meaning_violations  # noqa: F401  (re-exported)

FILLERS = {"uh", "um", "uhm", "mm", "hmm", "erm", "ah", "eh"}
U = "unspecified"


def normalize(t):
    t = re.sub(r"[^\w\s']", " ", t.lower().replace("’", "'"))
    return " ".join(w for w in t.split() if w not in FILLERS)


def wer(ref, hyp):
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    if not h:
        return 1.0
    return jiwer.wer(r, h)


def term_recall(terms, text):
    t = " " + normalize(text) + " "
    hits = sum(1 for x in terms if " " + normalize(x) + " " in t)
    return hits / len(terms) if terms else None


def _hit_indices(ref, hyp):
    out = jiwer.process_words(ref, hyp)
    hits = set()
    for c in out.alignments[0]:
        if c.type == "equal":
            hits.update(range(c.ref_start_idx, c.ref_end_idx))
    return hits


def refinement_effect(ref, raw, refined):
    """Fix rate (wrong -> right) and regression rate (right -> wrong) against the reference transcript."""
    r, a, b = normalize(ref), normalize(raw), normalize(refined)
    ha, hb = _hit_indices(r, a), _hit_indices(r, b)
    n = len(r.split())
    wrong_before, right_before = n - len(ha), len(ha)
    return {"fix_rate": len(hb - ha) / wrong_before if wrong_before else 0.0,
            "regression_rate": len(ha - hb) / right_before if right_before else 0.0}


class Sim:
    """Semantic similarity (sentence-transformers) with a lexical fallback."""

    def __init__(self, use_embeddings=True):
        self.model = None
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            except Exception:
                self.model = None

    def matrix(self, a, b):
        if not a or not b:
            return []
        if self.model is not None:
            from sentence_transformers import util
            return util.cos_sim(self.model.encode(a), self.model.encode(b)).tolist()
        return [[fuzz.token_set_ratio(normalize(x), normalize(y)) / 100 for y in b] for x in a]


def match(pred, gold, sim, thr=0.6):
    """Greedy one-to-one matching. Returns list of (pred_idx, gold_idx)."""
    m = sim.matrix(pred, gold)
    cands = sorted(((m[i][j], i, j) for i in range(len(pred)) for j in range(len(gold))), reverse=True)
    used_p, used_g, pairs = set(), set(), []
    for s, i, j in cands:
        if s >= thr and i not in used_p and j not in used_g:
            pairs.append((i, j))
            used_p.add(i)
            used_g.add(j)
    return pairs


def prf(n_match, n_pred, n_gold):
    p = n_match / n_pred if n_pred else (1.0 if not n_gold else 0.0)
    r = n_match / n_gold if n_gold else 1.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def _same_person(a, b):
    a, b = (a or U).strip().lower(), (b or U).strip().lower()
    if a == U or b == U:
        return a == b
    return a in b or b in a


def eval_documentation(pred_decisions, pred_actions, gold, sim, thr=0.6):
    """gold: {decisions:[str], proposals:[str], actions:[{task,owner,deadline}]}."""
    res = {}
    gd, gp, ga = gold.get("decisions", []), gold.get("proposals", []), gold.get("actions", [])
    pairs = match(pred_decisions, gd, sim, thr)
    res["decision"] = prf(len(pairs), len(pred_decisions), len(gd))
    if gp:   # proposals wrongly reported as decisions
        wrong = match(pred_decisions, gp, sim, thr)
        res["proposal_as_decision_rate"] = len(wrong) / len(gp)
    pt = [a["task"] for a in pred_actions]
    pairs = match(pt, [a["task"] for a in ga], sim, thr)
    res["action"] = prf(len(pairs), len(pt), len(ga))
    if pairs:
        res["owner_acc"] = sum(_same_person(pred_actions[i]["owner"], ga[j].get("owner")) for i, j in pairs) / len(pairs)
        res["deadline_acc"] = sum(_same_person(pred_actions[i]["deadline"], ga[j].get("deadline")) for i, j in pairs) / len(pairs)
        res["invented_owner_rate"] = sum(
            (pred_actions[i]["owner"] or U).lower() != U and (ga[j].get("owner") or U).lower() == U
            for i, j in pairs) / len(pairs)
    return res


def _clip(x):
    return max(0.0, min(1.0, x))


def _num(v):
    """None for missing/NaN so that 'not measured' is never confused with 'measured as zero'."""
    if v is None or (isinstance(v, float) and v != v):
        return None
    return v


def _wavg(pairs):
    """Weighted mean (0-100) over the available (value, weight) pairs; None if nothing was measured."""
    have = [(v, w) for v, w in pairs if v is not None]
    return None if not have else 100 * sum(v * w for v, w in have) / sum(w for _, w in have)


def scorecard(m):
    """Map metrics to 0-100 area scores mirroring the judges' weights (20/20/25/15/15).
    An area whose inputs were not measured (no reference transcript, no annotated gold) is None and is EXCLUDED
    from the total (the weights are renormalised); `coverage` says how much of the rubric was actually measured."""
    g = lambda k: _num(m.get(k))
    inv = lambda v: None if v is None else 1 - v
    wer = g("wer_raw")
    stt = None if wer is None else 100 * _clip(1 - wer)
    ref = None
    if g("term_recall_refined") is not None or g("regression_rate") is not None:
        viol = g("violations_per_1k_words")
        ref = _wavg([(g("term_recall_refined"), 0.4), (inv(g("regression_rate")), 0.3),
                     (None if viol is None else 1 - min(1.0, viol / 5), 0.3)])
    mins = None
    if g("decision_f1") is not None:
        mins = _wavg([(g("decision_f1"), 0.5), (inv(g("proposal_as_decision_rate")), 0.3),
                      (g("judge_summary"), 0.2)])
    act = None
    if g("action_f1") is not None:
        act = _wavg([(g("action_f1"), 0.5), (g("owner_acc"), 0.25), (g("deadline_acc"), 0.25)])
    rtf = g("rtf")
    e2e = _wavg([(g("success_rate"), 0.5), (g("json_valid"), 0.25),
                 (None if rtf is None else _clip(1 - (rtf - 0.3) / 1.2), 0.25)])
    areas = {"stt": (stt, 20), "refine": (ref, 20), "minutes": (mins, 25), "actions": (act, 15), "e2e": (e2e, 15)}
    have = {k: v for k, v in areas.items() if v[0] is not None}
    total = sum(v * w for v, w in have.values()) / sum(w for _, w in have.values()) if have else None
    out = {k: (None if v[0] is None else round(v[0], 1)) for k, v in areas.items()}
    out["total_100"] = None if total is None else round(total, 1)
    out["coverage"] = round(sum(w for _, w in have.values()) / 95, 2)       # share of the rubric that was measured
    return out
