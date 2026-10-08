"""Speaker-attribution error against an RTTM reference (needs pyannote.metrics, installed with pyannote.audio)."""


def read_rttm(path):
    turns = []
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            turns.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
    return turns


def der_from_turns(ref_turns, hyp_turns, collar=0.25):
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate
    ref, hyp = Annotation(), Annotation()
    for s, e, k in ref_turns:
        ref[Segment(s, e)] = k
    for s, e, k in hyp_turns:
        hyp[Segment(s, e)] = k
    return float(DiarizationErrorRate(collar=collar, skip_overlap=False)(ref, hyp))
