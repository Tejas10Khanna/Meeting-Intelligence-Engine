"""Run every path on every sample and write a comparison table.
Layout:  data/<sample_id>/audio.(wav|mp3|...)   ref_transcript.txt (optional)   gold.json (optional)   ref.rttm (optional)
Folders starting with '_' or without an audio file are ignored. A metric that cannot be measured is shown as n/a
(never as 0) and excluded from the total; 'coverage' shows how much of the rubric was measured.
gold.json: {"annotated": true, "terms": [], "decisions": [], "proposals": [], "actions": [{"task","owner","deadline"}]}
Usage:  python -m eval.run_eval --data data --paths configs/p2_guarded.yaml configs/p3_speaker.yaml --out results [--only ES2008a]"""
import argparse
import glob
import json
import os

import pandas as pd

from eval import metrics as M
from mtg.audio import AudioError, duration
from mtg.pipeline import load_config, run

try:
    from mtg.pipeline import free_all
except ImportError:                                  # older checkout without the patch
    def free_all():
        pass

AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".mp4", ".webm", ".mkv")


def discover_samples(data, only=None):
    out = []
    for d in sorted(glob.glob(os.path.join(data, "*"))):
        sid = os.path.basename(d)
        if not os.path.isdir(d) or sid.startswith("_") or (only and sid not in only):
            continue
        audio = next((p for p in sorted(glob.glob(os.path.join(d, "audio.*"))) if p.lower().endswith(AUDIO_EXT)), None)
        if audio is None:
            continue
        opt = lambda n: os.path.join(d, n) if os.path.exists(os.path.join(d, n)) else None
        out.append({"id": sid, "dir": d, "audio": audio, "ref": opt("ref_transcript.txt"),
                    "gold": opt("gold.json"), "rttm": opt("ref.rttm")})
    return out


def is_annotated(gold):
    return bool(gold.get("annotated", bool(gold.get("decisions") or gold.get("proposals") or gold.get("actions"))))


def manifest(samples):
    rows = []
    for s in samples:
        try:
            mins = round(duration(s["audio"]) / 60, 1)
        except Exception:
            mins = None
        gold = json.load(open(s["gold"], encoding="utf-8")) if s["gold"] else {}
        rows.append({"sample": s["id"], "minutes": mins, "ref_transcript": bool(s["ref"]),
                     "gold_annotated": is_annotated(gold) if s["gold"] else False, "ref_rttm": bool(s["rttm"])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--paths", nargs="+", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", nargs="*", help="evaluate only these sample ids")
    ap.add_argument("--keep-models", action="store_true",
                    help="keep LLMs loaded between samples (faster; needs ~11 GB free for both 4-bit LLMs)")
    ap.add_argument("--cache", default=None, help="stage cache dir (default <out>/_cache); 'off' disables")
    ap.add_argument("--no-embed", action="store_true", help="lexical matching only")
    a = ap.parse_args()
    os.environ.setdefault("MTG_CACHE", a.cache or os.path.join(a.out, "_cache"))
    samples = discover_samples(a.data, a.only)
    if not samples:
        raise SystemExit(f"No samples found in '{a.data}'. Expected data/<id>/audio.wav (+ ref_transcript.txt, gold.json). "
                         "Folders starting with '_' are ignored.")
    print("Samples that will be evaluated:\n" + manifest(samples).to_string(index=False) + "\n")
    sim = None
    rows = []
    for cfgp in a.paths:
        cfg = load_config(cfgp)
        if a.keep_models:
            cfg["unload_between"] = False
        for s in samples:
            out = f"{a.out}/{cfg['name']}/{s['id']}"
            row = {"path": cfg["name"], "sample": s["id"], "success": 1}
            try:
                r = run(s["audio"], cfg, out)
            except Exception as e:                               # AudioError or any failure counts as a failed run
                row.update(success=0, error=f"{type(e).__name__}: {str(e)[:200]}")
                rows.append(row)
                print(f"  FAILED {cfg['name']} / {s['id']}: {row['error']}")
                continue
            raw_t, ref_t = r["raw"], "\n".join(r["refined_map"][u.id] for u in r["units"])
            row.update(rtf=r["report"]["rtf"], vram=r["report"]["peak_vram_gb"], json_valid=1,
                       speakers=r["report"].get("speakers"))
            row["violations_per_1k_words"] = 1000 * sum(
                bool(M.meaning_violations(u.text, r["refined_map"][u.id])) for u in r["units"]) / max(1, len(raw_t.split()))
            gold = json.load(open(s["gold"], encoding="utf-8")) if s["gold"] else {}
            if s["ref"]:
                ref = open(s["ref"], encoding="utf-8").read()
                row["wer_raw"], row["wer_refined"] = M.wer(ref, raw_t), M.wer(ref, ref_t)
                row.update(M.refinement_effect(ref, raw_t, ref_t))
            if gold.get("terms"):
                row["term_recall_raw"] = M.term_recall(gold["terms"], raw_t)
                row["term_recall_refined"] = M.term_recall(gold["terms"], ref_t)
            if s["gold"] and is_annotated(gold):
                sim = sim or M.Sim(not a.no_embed)
                rec = r["record"]
                d = M.eval_documentation([x.text for x in rec.decisions], [x.model_dump() for x in rec.action_items], gold, sim)
                row["decision_f1"], row["action_f1"] = d["decision"]["f1"], d["action"]["f1"]
                for k in ("proposal_as_decision_rate", "owner_acc", "deadline_acc", "invented_owner_rate"):
                    row[k] = d.get(k)
            if s["rttm"]:
                try:
                    from eval.der import der_from_turns, read_rttm
                    row["speaker_der"] = der_from_turns(read_rttm(s["rttm"]),
                                                        [(u.start, u.end, u.speaker) for u in r["units"] if u.speaker])
                except Exception as e:
                    row["speaker_der_error"] = str(e)[:100]
            rows.append(row)
        free_all()
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/per_sample.csv", index=False)
    summ = []
    for p, g in df.groupby("path"):
        m = {c: g[c].mean() for c in g.columns if c not in ("path", "sample", "error", "speaker_der_error")
             and pd.api.types.is_numeric_dtype(g[c])}
        m["success_rate"] = g["success"].mean()
        summ.append({"path": p, **m, **M.scorecard(m)})
    sm = pd.DataFrame(summ)
    sm.to_csv(f"{a.out}/summary.csv", index=False)
    cols = [c for c in ("path", "stt", "refine", "minutes", "actions", "e2e", "total_100", "coverage") if c in sm]
    print("\nScorecard (n/a = not measured, excluded from the total):")
    print(sm[cols].to_string(index=False, na_rep="n/a"))
    show = [c for c in ("path", "sample", "success", "wer_raw", "wer_refined", "term_recall_refined", "regression_rate",
                        "decision_f1", "action_f1", "speaker_der", "speakers", "rtf", "error") if c in df]
    print("\nPer sample:")
    print(df[show].to_string(index=False, na_rep="n/a"))


if __name__ == "__main__":
    main()
