"""Merge word-level ASR output with diarization turns into speaker-attributed turns."""
import math

from .types import Segment


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speakers(words, turns, max_snap=1.0):
    """Label each word with the speaker whose turn overlaps it most (nearest turn if in a gap)."""
    labels, last = [], None
    for w in words:
        best, best_ov = None, 0.0
        for s, e, spk in turns:
            ov = _overlap(w.start, w.end, s, e)
            if ov > best_ov:
                best, best_ov = spk, ov
        if best is None and turns:
            mid = (w.start + w.end) / 2
            d, spk = min((min(abs(mid - s), abs(mid - e)), spk) for s, e, spk in turns)
            best = spk if d <= max_snap else last
        labels.append(best or last)
        last = labels[-1]
    return labels


def smooth(labels, min_run=2):
    """Remove isolated 1-word speaker flips: A B A -> A A A (runs shorter than min_run)."""
    labels = list(labels)
    i = 0
    while i < len(labels):
        j = i
        while j < len(labels) and labels[j] == labels[i]:
            j += 1
        if (j - i) < min_run and 0 < i and j < len(labels) and labels[i - 1] == labels[j]:
            for k in range(i, j):
                labels[k] = labels[i - 1]
        i = j
    return labels


def build_turns(words, labels, overlaps=(), max_gap=1.5):
    """Group consecutive same-speaker words into turns; name speakers 'Speaker 1..N' by first appearance."""
    names, turns, cur = {}, [], []

    def flush():
        if not cur:
            return
        spk = cur[0][1]
        names.setdefault(spk, f"Speaker {len(names) + 1}")
        ws = [w for w, _ in cur]
        lp = sum(math.log(max(w.prob, 1e-4)) for w in ws) / len(ws)
        ov = any(_overlap(w.start, w.end, a, b) > 0 for w in ws for a, b in overlaps)
        turns.append(Segment(id=len(turns), start=ws[0].start, end=ws[-1].end,
                             text=" ".join(w.text for w in ws), avg_logprob=lp, words=ws,
                             speaker=names[spk], overlap=ov))
        cur.clear()

    for w, spk in zip(words, labels):
        if cur and (spk != cur[-1][1] or w.start - cur[-1][0].end > max_gap):
            flush()
        cur.append((w, spk))
    flush()
    return turns
