import json
import math
import os
import subprocess
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest

from eval import metrics as M
from eval.der import read_rttm
from eval.run_eval import discover_samples, is_annotated
from mtg import consensus, stt
from mtg.types import Segment, Word


def _wav(path, seconds=1, ch=1):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\x10\x00" * 16000 * seconds * ch)


def test_load_wav_array(tmp_path):
    p = tmp_path / "a.wav"; _wav(p, 2)
    x, sr = stt.load_wav_array(str(p))
    assert sr == 16000 and x.dtype.name == "float32" and len(x) == 32000
    _wav(p, 1, ch=2)
    assert len(stt.load_wav_array(str(p))[0]) == 16000          # stereo is averaged to mono
    assert stt.load_wav_array(str(tmp_path / "missing.wav")) is None


def test_backend_falls_back_when_ct2_probe_fails(monkeypatch):
    monkeypatch.setattr(stt, "_cuda", lambda: True)
    monkeypatch.setattr(stt, "ct2_ok", lambda force=False: (False, "Could not load libcudnn_ops.so.9"))
    assert stt._pick_backend({}) == "hf"
    monkeypatch.setattr(stt, "ct2_ok", lambda force=False: (True, ""))
    assert stt._pick_backend({}) == "ct2"
    assert stt._pick_backend({"backend": "hf"}) == "hf"          # explicit choice skips the probe


def test_transcribe_dispatch(monkeypatch):
    monkeypatch.setattr(stt, "_pick_backend", lambda cfg: "hf")
    monkeypatch.setattr(stt, "_transcribe_hf", lambda *a: ["hf"])
    monkeypatch.setattr(stt, "_transcribe_ct2", lambda *a: ["ct2"])
    assert stt.transcribe_whisper("x.wav", {}) == ["hf"]


def test_ct2_probe_survives_subprocess_crash(monkeypatch):
    class R:
        returncode, stdout, stderr = -6, "", "Aborted (core dumped) Could not load library libcudnn"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    stt._PROBE.clear()
    ok, msg = stt.ct2_ok()
    assert ok is False and "cudnn" in msg
    stt._PROBE.clear()


def test_gate_does_not_switch_off_without_confidence():
    units = [Segment(0, 0, 2, "a b c", words=[Word("a", 0, 1, 1.0)]), Segment(1, 2, 4, "d e f")]
    consensus.flag_units(units, {})
    assert all(u.flagged and u.flag_reasons == ["no_confidence_info"] for u in units)
    ok = [Segment(0, 0, 2, "a b c", avg_logprob=-0.1, words=[Word("a", 0, 1, 0.99)])]
    consensus.flag_units(ok, {})
    assert not ok[0].flagged                                      # real confidence present -> normal gating


def test_scorecard_not_measured_is_not_zero():
    s = M.scorecard({"success_rate": 1.0, "json_valid": 1, "rtf": 0.3, "violations_per_1k_words": 0.0})
    assert s["stt"] is None and s["refine"] is None and s["minutes"] is None and s["actions"] is None
    assert s["e2e"] == 100.0 and s["total_100"] == 100.0 and s["coverage"] == pytest.approx(15 / 95, abs=0.01)
    s2 = M.scorecard({"wer_raw": 0.2, "success_rate": 1.0, "json_valid": 1, "rtf": 0.3,
                      "decision_f1": float("nan"), "action_f1": float("nan")})
    assert s2["minutes"] is None and s2["actions"] is None and s2["stt"] == 80.0
    full = M.scorecard({"wer_raw": 0.1, "term_recall_refined": 1, "regression_rate": 0, "violations_per_1k_words": 0,
                        "decision_f1": 1, "action_f1": 1, "owner_acc": 1, "deadline_acc": 1,
                        "success_rate": 1, "json_valid": 1, "rtf": 0.3})
    assert full["coverage"] == 1.0 and 0 < full["total_100"] <= 100


def test_discover_samples_skips_junk(tmp_path):
    for name, audio in (("good", True), ("_junk", True), ("noaudio", False), ("dummy2", True)):
        d = tmp_path / name; d.mkdir()
        if audio:
            _wav(d / "audio.wav")
    (tmp_path / "good" / "ref_transcript.txt").write_text("hello")
    (tmp_path / "good" / "ref.rttm").write_text("SPEAKER m 1 0.5 2.0 <NA> <NA> A <NA> <NA>\n")
    ids = [s["id"] for s in discover_samples(str(tmp_path))]
    assert ids == ["dummy2", "good"]
    assert [s["id"] for s in discover_samples(str(tmp_path), only=["good"])] == ["good"]
    g = [s for s in discover_samples(str(tmp_path)) if s["id"] == "good"][0]
    assert g["ref"] and g["rttm"] and g["gold"] is None
    assert read_rttm(g["rttm"]) == [(0.5, 2.5, "A")]


def test_is_annotated():
    assert not is_annotated({"terms": [], "decisions": [], "proposals": [], "actions": []})
    assert is_annotated({"decisions": ["x"]})
    assert is_annotated({"annotated": True, "decisions": []})
