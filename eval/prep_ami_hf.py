"""Rebuild ONE full-length meeting from the Hugging Face AMI dataset (UNTESTED here - check the first output by ear).
The HF dataset stores 1-3 second utterances, so evaluating one folder per utterance gives meaningless 1-second 'meetings'.
Use config 'sdm' (every utterance is a crop of the SAME far-field channel): each crop is placed back at its original
begin_time, which reproduces the real recording in speech regions. It also writes reference text and a reference RTTM.
Usage: python -m eval.prep_ami_hf --meeting ES2008a --split test --out data/ES2008a --max_minutes 15
Assumes the split lists a meeting's rows contiguously (it stops at the first different meeting after seeing yours)."""
import argparse
import io
import json
import os

import numpy as np
import soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meeting", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf_id", default="edinburghcstr/ami")
    ap.add_argument("--config", default="sdm")
    ap.add_argument("--max_minutes", type=float, default=15.0)
    a = ap.parse_args()
    from datasets import Audio, load_dataset
    ds = load_dataset(a.hf_id, a.config, split=a.split, streaming=True).cast_column("audio", Audio(decode=False))
    limit, rows, seen = a.max_minutes * 60, [], False
    for r in ds:
        if r["meeting_id"] == a.meeting:
            seen = True
            if r["begin_time"] < limit:
                rows.append(r)
        elif seen:
            break
    if not rows:
        raise SystemExit(f"No rows found for meeting {a.meeting} in split {a.split}.")
    sr = 16000
    buf = np.zeros(int((max(r["end_time"] for r in rows) + 1) * sr), dtype="float32")
    for r in rows:
        x, s = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        i = int(r["begin_time"] * sr)
        buf[i:i + len(x)] = x[:max(0, len(buf) - i)]
    os.makedirs(a.out, exist_ok=True)
    sf.write(os.path.join(a.out, "audio.wav"), buf, sr, subtype="PCM_16")
    rows.sort(key=lambda r: r["begin_time"])
    open(os.path.join(a.out, "ref_transcript.txt"), "w", encoding="utf-8").write(" ".join(r["text"].lower() for r in rows))
    with open(os.path.join(a.out, "ref.rttm"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"SPEAKER {a.meeting} 1 {r['begin_time']:.3f} {r['end_time'] - r['begin_time']:.3f} <NA> <NA> {r['speaker_id']} <NA> <NA>\n")
    gold = os.path.join(a.out, "gold.json")
    if not os.path.exists(gold):
        json.dump({"annotated": False, "terms": [], "decisions": [], "proposals": [], "actions": []},
                  open(gold, "w", encoding="utf-8"), indent=2)
    print(f"{a.meeting}: {len(rows)} utterances, {len(buf) / sr / 60:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
