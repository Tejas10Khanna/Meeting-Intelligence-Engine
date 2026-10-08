import json
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from eval import metrics as M
from mtg import consensus, document as doc, guardrails as G, merge, refine as rf
from mtg.audio import AudioError, validate_and_convert
from mtg.diarize import overlap_regions
from mtg.llm import LLM, chat_json, parse_json
from mtg.types import Segment, Word


class FakeLLM(LLM):
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def chat(self, system, user, max_new_tokens=0, temperature=0.0):
        self.calls += 1
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


# ---------------- guardrails ----------------
def test_guard_blocks_number_and_negation_changes():
    text, log = G.apply_guarded("we will not ship 5 units", "we will ship 6 units")
    assert text == "we will not ship 5 units"
    assert {h["reason"] for h in log} == {"negation", "number"}


def test_guard_accepts_glossary_fix_and_rejects_wild_rewrite():
    t, _ = G.apply_guarded("deploy on the sea cure server", "deploy on the SQL server", glossary=["SQL"])
    assert "SQL" in t
    t2, log = G.apply_guarded("the budget looks fine", "the weather is lovely today", min_sim=0.5)
    assert t2 == "the budget looks fine" and not log[0]["accepted"]


def test_guard_accepts_support_from_other_asr():
    t, log = G.apply_guarded("we use cooper netties", "we use kubernetes", support={"kubernetes"}, min_sim=0.9)
    assert t == "we use kubernetes" and log[0]["reason"] == "supported_by_other_asr"


def test_protected_name_cannot_change():
    t, _ = G.apply_guarded("thanks priya for that", "thanks maria for that", protected=["Priya"])
    assert "priya" in t


def test_meaning_violations():
    assert G.meaning_violations("do not go", "go") == ["negation_changed"]
    assert G.meaning_violations("pay $200 now", "pay $20 now") == ["number_changed"]
    assert G.meaning_violations("hello there", "hello there!") == []


# ---------------- merge / diarization ----------------
def W(t, s, e, p=0.9):
    return Word(t, s, e, p)


def test_assign_smooth_and_turns():
    words = [W("hi", 0, .4), W("all", .5, .9), W("ok", 1.0, 1.2), W("sure", 3.0, 3.4), W("yes", 3.5, 3.8)]
    turns = [(0, 1.3, "A"), (2.9, 4.0, "B")]
    labels = merge.assign_speakers(words, turns)
    assert labels == ["A", "A", "A", "B", "B"]
    assert merge.smooth(["A", "B", "A", "A"]) == ["A", "A", "A", "A"]
    ts = merge.build_turns(words, labels)
    assert [t.speaker for t in ts] == ["Speaker 1", "Speaker 2"]
    assert ts[0].text == "hi all ok"


def test_overlap_regions():
    assert overlap_regions([(0, 5, "A"), (3, 8, "B")]) == [(3, 5)]
    assert overlap_regions([(0, 2, "A"), (3, 5, "B")]) == []


def test_overlap_flag_on_turn():
    words = [W("x", 3.2, 3.4)]
    t = merge.build_turns(words, ["A"], overlaps=[(3, 5)])
    assert t[0].overlap


# ---------------- consensus ----------------
def test_disagreement_and_flagging():
    assert consensus.disagreement("the cat sat", "the cat sat") == 0
    assert consensus.disagreement("the cat sat", "a dog ran") > 0.5
    u = Segment(0, 0, 2, "we use cooper netties", avg_logprob=-0.2, words=[W("a", 0, 1)])
    consensus.annotate([u], [W("we", 0, .3), W("use", .4, .6), W("kubernetes", .7, 1.5)])
    consensus.flag_units([u], {"disagree_thr": 0.12})
    assert u.flagged and "asr_disagreement" in u.flag_reasons


# ---------------- llm json ----------------
def test_parse_json_with_fences_and_retry():
    assert parse_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    llm = FakeLLM(["not json", '{"ok": true}'])
    assert chat_json(llm, "s", "u") == {"ok": True} and llm.calls == 2
    with pytest.raises(ValueError):
        chat_json(FakeLLM(["nope"]), "s", "u", retries=1)


# ---------------- refine (fake LLM) ----------------
def test_refine_guarded_gate_and_log():
    units = [Segment(0, 0, 2, "we deploy on the sea cure server", flagged=True),
             Segment(1, 2, 4, "this one is fine")]
    reply = json.dumps({"units": [{"id": 0, "text": "we deploy on the SQL server"}]})
    out, log = rf.refine(units, FakeLLM([reply]), {"guard": True, "gate": True}, glossary=["SQL"])
    assert out[0].endswith("SQL server") and out[1] == "this one is fine" and len(log) == 1


def test_refine_survives_bad_llm_output():
    units = [Segment(0, 0, 2, "hello world")]
    out, log = rf.refine(units, FakeLLM(["garbage"]), {"guard": True})
    assert out[0] == "hello world" and "error" in log[0]


# ---------------- documentation (fake LLM) ----------------
TRANSCRIPT_UNITS = [Segment(0, 0, 5, "Let's go with Postgres for the new service. Agreed.", speaker="Speaker 1"),
                    Segment(1, 5, 9, "Priya will send the migration plan by Friday.", speaker="Speaker 2"),
                    Segment(2, 9, 12, "Maybe we should also consider Redis.", speaker="Speaker 1")]
REFINED = {u.id: u.text for u in TRANSCRIPT_UNITS}


def _reply():
    return json.dumps({
        "summary": "Chose Postgres.", "minutes": [{"title": "Database", "points": ["Postgres chosen"]}],
        "decisions": [{"text": "Use Postgres", "evidence": "Let's go with Postgres for the new service"},
                      {"text": "Use Redis", "evidence": "we should also move everything to Redis"}],
        "open_proposals": ["Consider Redis"],
        "action_items": [
            {"task": "Send migration plan", "owner": "Priya", "owner_basis": "named", "deadline": "Friday",
             "evidence": "Priya will send the migration plan by Friday"},
            {"task": "Write docs", "owner": "Rahul", "owner_basis": "named", "deadline": "next week",
             "evidence": "Priya will send the migration plan by Friday"}]})


def test_document_grounding_drops_invented_items():
    rec, rep = doc.build_record(TRANSCRIPT_UNITS, REFINED, FakeLLM([_reply()]), {"grounded": True})
    assert [d.text for d in rec.decisions] == ["Use Postgres"]          # fabricated quote dropped
    a = {x.task: x for x in rec.action_items}
    assert a["Send migration plan"].owner == "Priya" and a["Send migration plan"].deadline == "Friday"
    assert a["Write docs"].owner == "unspecified" and a["Write docs"].deadline == "unspecified"  # not in transcript
    assert rec.open_proposals == ["Consider Redis"]
    md = doc.to_markdown(rec)
    assert "unspecified" in md and "Postgres" in md


def test_document_self_consistency_keeps_only_agreed():
    r2 = json.loads(_reply())
    r2["decisions"] = []
    rec, _ = doc.build_record(TRANSCRIPT_UNITS, REFINED, FakeLLM([_reply(), json.dumps(r2)]),
                              {"grounded": True, "consensus_runs": 2})
    assert rec.decisions == []


def test_basic_mode_does_not_require_evidence():
    basic = json.dumps({"summary": "s", "minutes": [], "decisions": ["Use Postgres"],
                        "action_items": [{"task": "t", "owner": "unspecified", "deadline": "unspecified"}]})
    rec, _ = doc.build_record(TRANSCRIPT_UNITS, REFINED, FakeLLM([basic]), {"grounded": False})
    assert len(rec.decisions) == 1 and len(rec.action_items) == 1


# ---------------- metrics ----------------
def test_wer_and_refinement_effect():
    assert M.wer("the cat sat", "the cat sat") == 0.0
    assert M.wer("the cat sat", "the dog sat") == pytest.approx(1 / 3)
    e = M.refinement_effect("we use kubernetes today", "we use cooper netties today", "we use kubernetes today")
    assert e["fix_rate"] == 1.0 and e["regression_rate"] == 0.0
    e = M.refinement_effect("we use kubernetes today", "we use kubernetes today", "we use cooper today")
    assert e["regression_rate"] > 0


def test_documentation_metrics():
    sim = M.Sim(use_embeddings=False)
    gold = {"decisions": ["Use Postgres for the new service"], "proposals": ["Consider Redis"],
            "actions": [{"task": "Send the migration plan", "owner": "Priya", "deadline": "Friday"},
                        {"task": "Update the wiki", "owner": "unspecified", "deadline": "unspecified"}]}
    res = M.eval_documentation(["Use Postgres for new service", "Consider Redis for caching"],
                               [{"task": "Send migration plan", "owner": "Priya", "deadline": "Friday"},
                                {"task": "Update wiki", "owner": "Rahul", "deadline": "unspecified"}], gold, sim)
    assert res["decision"]["recall"] == 1.0
    assert res["proposal_as_decision_rate"] == 1.0
    assert res["action"]["f1"] == 1.0 and res["owner_acc"] == 0.5 and res["invented_owner_rate"] == 0.5


def test_scorecard_range():
    s = M.scorecard({"wer_raw": 0.1, "term_recall_refined": 0.9, "regression_rate": 0.01, "decision_f1": 0.8,
                     "action_f1": 0.7, "owner_acc": 0.8, "deadline_acc": 0.8, "rtf": 0.3})
    assert 0 <= s["total_100"] <= 100


# ---------------- audio validation ----------------
def test_audio_validation(tmp_path):
    with pytest.raises(AudioError):
        validate_and_convert(str(tmp_path / "missing.wav"), str(tmp_path))
    empty = tmp_path / "e.wav"; empty.write_bytes(b"")
    with pytest.raises(AudioError, match="empty"):
        validate_and_convert(str(empty), str(tmp_path))
    bad = tmp_path / "b.mp3"; bad.write_bytes(b"not audio at all" * 10)
    with pytest.raises(AudioError, match="could not be read"):
        validate_and_convert(str(bad), str(tmp_path))
    txt = tmp_path / "t.txt"; txt.write_text("hi")
    with pytest.raises(AudioError, match="Unsupported"):
        validate_and_convert(str(txt), str(tmp_path))
    sil = tmp_path / "s.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "3", str(sil)])
    with pytest.raises(AudioError, match="silent"):
        validate_and_convert(str(sil), str(tmp_path))
    ok = tmp_path / "o.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=300:duration=3", str(ok)])
    wav, dur = validate_and_convert(str(ok), str(tmp_path / "out"))
    assert os.path.exists(wav) and dur == pytest.approx(3, abs=0.2)
