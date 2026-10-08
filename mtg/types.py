from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float = 1.0


@dataclass
class Segment:
    """A transcript unit: an ASR segment, or a speaker turn after diarization merge."""
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    words: list = field(default_factory=list)
    speaker: Optional[str] = None
    overlap: bool = False                              # overlapped speech detected inside this unit
    alt_texts: dict = field(default_factory=dict)      # other hypotheses {system_name: text}
    disagreement: float = 0.0                          # 0..1 disagreement between ASR systems
    flagged: bool = False                              # low-confidence -> candidate for correction
    flag_reasons: list = field(default_factory=list)
