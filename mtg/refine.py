"""Stage 2: transcript refinement with LLM-A. Modes differ in how much structure and safety they add."""
from .guardrails import apply_guarded, norm
from .llm import chat_json
from .prompts import GLOSSARY_SYS, REFINE_BASIC_SYS, REFINE_GUARDED_SYS


def infer_glossary(raw_text, llm, user_terms=(), context=""):
    words = raw_text.split()
    sample = " ".join(words[:2500] + (["..."] + words[-1500:] if len(words) > 4000 else words[2500:4000]))
    try:
        out = chat_json(llm, GLOSSARY_SYS, f"Meeting context: {context or 'none'}\n\nTranscript excerpt:\n{sample}")
        terms = [t for t in out.get("terms", []) if isinstance(t, str) and 1 < len(t) < 40]
    except Exception:
        terms = []
    seen, merged = set(), []
    for t in list(user_terms) + terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            merged.append(t)
    return merged


def _batches(idx, size):
    for i in range(0, len(idx), size):
        yield idx[i:i + size]


def refine(units, llm, cfg, glossary=(), protected=(), context=""):
    """Returns (refined_text_by_unit_id, change_log). cfg keys: guard, gate, window, ctx_units, use_alts."""
    guard, gate = cfg.get("guard", False), cfg.get("gate", False)
    refined = {u.id: u.text for u in units}
    by_id = {u.id: u for u in units}
    targets = [i for i, u in enumerate(units) if (not gate) or u.flagged]
    system = REFINE_GUARDED_SYS if guard else REFINE_BASIC_SYS
    log, errors = [], 0
    for batch in _batches(targets, cfg.get("window", 12)):
        first = batch[0]
        ctx = units[max(0, first - cfg.get("ctx_units", 3)):first] if guard else []
        payload = []
        for i in batch:
            u = units[i]
            item = {"id": u.id, "speaker": u.speaker, "text": u.text}
            if cfg.get("use_alts") and u.alt_texts:
                item["alternatives"] = {k: v for k, v in u.alt_texts.items() if v}
            payload.append(item)
        user = ""
        if guard:
            user += f"Meeting context: {context or 'none'}\nGlossary: {', '.join(glossary) or 'none'}\n"
            user += "Context units (read-only):\n" + "\n".join(f"- {c.speaker or ''}: {c.text}" for c in ctx) + "\n"
        user += "Units to edit (JSON):\n" + str(payload).replace("'", '"')
        try:
            out = chat_json(llm, system, user, max_new_tokens=cfg.get("max_new_tokens", 2500))
            items = out.get("units", [])
        except Exception as e:                                # keep raw text for this batch
            errors += 1
            log.append({"id": None, "error": str(e)[:200]})
            continue
        for it in items:
            uid, new = it.get("id"), it.get("text")
            if uid not in by_id or not isinstance(new, str) or not new.strip():
                continue
            raw = by_id[uid].text
            if guard:
                support = {norm(t) for alt in by_id[uid].alt_texts.values() for t in alt.split()}
                final, hunks = apply_guarded(raw, new, glossary, protected, support, cfg.get("min_sim", 0.5))
            else:
                final, hunks = new, [{"raw": raw, "new": new, "accepted": True, "reason": "unguarded"}]
            refined[uid] = final
            if final != raw:
                log.append({"id": uid, "raw": raw, "proposed": new, "final": final, "hunks": hunks,
                            "flags": by_id[uid].flag_reasons})
    return refined, log
