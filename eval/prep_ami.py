"""Build one evaluation sample from the AMI corpus (UNTESTED against the real download - verify on first use).
You need: (1) the meeting audio, e.g. <meeting>.Mix-Headset.wav, and (2) the manual annotation package (NXT XML),
whose per-speaker word files look like  words/<meeting>.<A|B|C|D>.words.xml  with <w starttime=".." endtime="..">word</w>.
Usage: python -m eval.prep_ami --audio ES2008a.Mix-Headset.wav --words_glob "ami_manual/words/ES2008a.*.words.xml" --out data/ES2008a
Creates data/<id>/audio.wav, ref_transcript.txt and an empty gold.json to fill in (terms, decisions, proposals, actions)."""
import argparse
import glob
import json
import os
import shutil
import xml.etree.ElementTree as ET


def read_words(path):
    out = []
    for el in ET.parse(path).getroot().iter():
        txt = (el.text or "").strip()
        if el.tag.split("}")[-1] == "w" and el.get("starttime") and txt and el.get("punc") != "true":
            out.append((float(el.get("starttime")), txt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--words_glob", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    words = sorted(w for p in glob.glob(a.words_glob) for w in read_words(p))
    open(os.path.join(a.out, "ref_transcript.txt"), "w", encoding="utf-8").write(" ".join(t for _, t in words))
    shutil.copy(a.audio, os.path.join(a.out, "audio" + os.path.splitext(a.audio)[1]))
    gold = os.path.join(a.out, "gold.json")
    if not os.path.exists(gold):
        json.dump({"terms": [], "decisions": [], "proposals": [], "actions": []}, open(gold, "w"), indent=2)
    print(f"{len(words)} words written to {a.out}")


if __name__ == "__main__":
    main()
