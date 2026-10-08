"""Whole-pipeline smoke test with stubbed models: checks wiring, outputs, error handling."""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest

from mtg import pipeline, stt
from mtg.audio import AudioError
from mtg.llm import LLM
from mtg.types import Segment, Word


class Stub(LLM):
    def chat(self, system, user, max_new_tokens=0, temperature=0.0):
        if "Output JSON only: {\"domain\"" in system or "domain-specific terms" in system:
            return '{"domain":"infra","terms":["Postgres"]}'
        if "units" in system and "Return JSON only" in system:
            return '{"units":[{"id":0,"text":"Let us go with Postgres for the new service. Agreed."}]}'
        if "audit" in system:
            return '{"verdicts":[]}'
        return json.dumps({"summary": "Chose Postgres.", "minutes": [{"title": "DB", "points": ["Postgres"]}],
                           "decisions": [{"text": "Use Postgres", "evidence": "go with Postgres for the new service"}],
                           "open_proposals": [], "action_items": []})


def test_run_end_to_end_stub(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=3", str(wav)])
    seg = Segment(0, 0, 3, "Let's go with Postgres for the new service. Agreed.", avg_logprob=-0.9,
                  words=[Word("Let's", 0, .3, .9)])
    monkeypatch.setattr(stt, "transcribe_whisper", lambda *a, **k: [seg])
    monkeypatch.setattr(pipeline, "get_llm", lambda name, cfgs: Stub())
    cfg = pipeline.load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "p2_guarded.yaml"))
    out = pipeline.run(str(wav), cfg, str(tmp_path / "out"), context="infra sync", user_terms=["Postgres"])
    for f in ("raw_transcript.txt", "refined_transcript.txt", "meeting_record.json", "meeting_record.md",
              "meeting_record.docx", "run_report.json", "refine_changes.json"):
        assert os.path.exists(tmp_path / "out" / f), f
    rec = json.load(open(tmp_path / "out" / "meeting_record.json"))
    assert rec["decisions"][0]["text"] == "Use Postgres"
    assert out["report"]["flagged_units"] == 1


def test_run_rejects_bad_file(tmp_path):
    bad = tmp_path / "x.mp3"
    bad.write_bytes(b"junk" * 50)
    cfg = pipeline.load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "p1_lean.yaml"))
    with pytest.raises(AudioError):
        pipeline.run(str(bad), cfg, str(tmp_path / "o"))
