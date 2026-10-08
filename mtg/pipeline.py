"""End-to-end orchestration. A 'path' is just a YAML config selecting which modules run."""
import json
import os
import re
import tempfile
import time

import yaml

from . import audio, consensus, diarize as dz, document as doc, merge, refine as rf, stt
from .llm import get_llm


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def rename_speakers(text, mapping):
    for old, new in mapping.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    return text


def parse_mapping(s):
    """'Speaker 1=Alice, Speaker 2=Bob' -> dict"""
    return {a.strip(): b.strip() for a, b in (p.split("=", 1) for p in s.split(",") if "=" in p)}


def _peak_vram():
    try:
        import torch
        return round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None
    except Exception:
        return None


def run(audio_path, cfg, out_dir, context="", user_terms=(), attendees=(), progress=print):
    os.makedirs(out_dir, exist_ok=True)
    T, rep = {}, {"config": cfg["name"]}
    t0 = time.time()

    def stage(name):
        progress(f"[{cfg['name']}] {name} ...")
        return time.time()

    t = stage("validating audio")
    wav, dur = audio.validate_and_convert(audio_path, out_dir)
    rep["audio_seconds"] = round(dur, 1)
    T["validate"] = time.time() - t

    # ---- Stage 1: speech-to-text -------------------------------------------------------------
    t = stage("transcribing")
    sc = cfg["stt"]
    prompt = (context + " " + ", ".join(user_terms))[:600] if sc.get("prompt") and (context or user_terms) else None
    if sc["engine"] == "parakeet":
        segs = stt.words_to_segments(stt.transcribe_parakeet(wav, sc))
    else:
        segs = stt.transcribe_whisper(wav, sc, prompt)
    if not segs:
        raise audio.AudioError("No speech was detected in the recording.")
    raw_plain = "\n".join(s.text for s in segs)
    T["stt"] = time.time() - t

    # ---- Optional: diarization (never allowed to break the pipeline) ----------------------------
    units = segs
    dc = cfg.get("diarization", {})
    if dc.get("enabled"):
        t = stage("identifying speakers")
        try:
            turns = dz.diarize(wav, dc)
            words = [w for s in segs for w in s.words]
            if not words:
                raise RuntimeError("no word timestamps available")
            labels = merge.smooth(merge.assign_speakers(words, turns))
            units = merge.build_turns(words, labels, dz.overlap_regions(turns))
            rep["speakers"] = len({u.speaker for u in units})
        except Exception as e:
            rep["diarization_error"] = str(e)[:300]
            progress(f"Speaker identification unavailable ({str(e)[:80]}); continuing without speaker labels.")
            units = segs
        T["diarization"] = time.time() - t
    raw_spk = "\n".join(f"{u.speaker or ''}: {u.text}" if u.speaker else u.text for u in units)

    # ---- Optional: second ASR system for consensus (Path 5) -----------------------------------
    cc = cfg.get("consensus", {})
    if cc.get("enabled"):
        t = stage("second recogniser (consensus)")
        try:
            consensus.annotate(units, stt.transcribe_parakeet(wav, cc))
        except Exception as e:
            rep["consensus_error"] = str(e)[:300]
        T["consensus"] = time.time() - t
    rc = cfg["refine"]
    if rc.get("gate"):
        consensus.flag_units(units, rc)
    rep["flagged_units"], rep["units"] = sum(u.flagged for u in units), len(units)

    # ---- Stage 2: refinement (LLM-A) --------------------------------------------------------
    stt.free(whisper=True, nemo=True)
    t = stage("refining transcript")
    llm_a = get_llm(rc["llm"], cfg["llms"])
    glossary = rf.infer_glossary(raw_plain, llm_a, user_terms, context) if rc.get("glossary") else list(user_terms)
    rep["glossary"] = glossary
    if rc.get("redecode") and glossary:                       # acoustic re-check of flagged units, biased by glossary
        with tempfile.TemporaryDirectory() as td:
            for u in [u for u in units if u.flagged][: rc.get("redecode_max", 60)]:
                p = audio.slice_audio(wav, u.start - 0.4, u.end + 0.4, os.path.join(td, "s.wav"))
                s2 = stt.transcribe_whisper(p, sc, "Glossary: " + ", ".join(glossary)[:500])
                if s2:
                    u.alt_texts["whisper_biased"] = " ".join(x.text for x in s2)
    refined, changes = rf.refine(units, llm_a, rc, glossary, attendees, context)
    if cfg.get("unload_between", True):
        llm_a.unload()
    T["refine"] = time.time() - t

    # ---- Stage 3: documentation (LLM-B) ---------------------------------------------------------
    t = stage("writing minutes, decisions and tasks")
    llm_b = get_llm(cfg["document"]["llm"], cfg["llms"])
    record, drep = doc.build_record(units, refined, llm_b, cfg["document"], context)
    T["document"] = time.time() - t
    rep["document_report"] = drep

    # ---- Save outputs --------------------------------------------------------------------------
    refined_text = "\n".join((f"{u.speaker}: " if u.speaker else "") + refined[u.id] for u in units)
    files = {
        "raw_transcript.txt": raw_plain,
        "raw_transcript_speakers.txt": raw_spk if any(u.speaker for u in units) else None,
        "refined_transcript.txt": refined_text,
        "refine_changes.json": json.dumps(changes, indent=2),
        "meeting_record.json": record.model_dump_json(indent=2),
        "meeting_record.md": doc.to_markdown(record),
    }
    for name, content in files.items():
        if content is not None:
            with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
                f.write(content)
    doc.to_docx(record, os.path.join(out_dir, "meeting_record.docx"))
    rep["timings_s"] = {k: round(v, 1) for k, v in T.items()}
    rep["total_s"] = round(time.time() - t0, 1)
    rep["rtf"] = round(rep["total_s"] / max(dur, 1), 3)
    rep["peak_vram_gb"] = _peak_vram()
    with open(os.path.join(out_dir, "run_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    return {"raw": raw_plain, "raw_speakers": raw_spk, "refined": refined_text, "changes": changes,
            "record": record, "report": rep, "out_dir": out_dir, "units": units, "refined_map": refined}
