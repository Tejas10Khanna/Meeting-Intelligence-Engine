import gc
import os
import wave
import torch
import torchaudio
import numpy as np

if not hasattr(torchaudio, 'AudioMetaData'):
    class AudioMetaData: pass
    torchaudio.AudioMetaData = AudioMetaData
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
if not hasattr(torchaudio, 'info'):
    torchaudio.info = lambda *args, **kwargs: torchaudio.AudioMetaData()

import huggingface_hub.file_download
if not hasattr(huggingface_hub.file_download, '_patched_for_pyannote'):
    _orig_hf_download = huggingface_hub.file_download.hf_hub_download
    def _patched_hf_download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return _orig_hf_download(*args, **kwargs)
    huggingface_hub.file_download.hf_hub_download = _patched_hf_download
    huggingface_hub.hf_hub_download = _patched_hf_download
    huggingface_hub.file_download._patched_for_pyannote = True

def _load_wav(path):
    try:
        with wave.open(path, "rb") as w:
            sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            raw = w.readframes(n)
        if sw != 2: return None
        x = np.frombuffer(raw, dtype=np.int16).astype("float32") / 32768.0
        x = x.reshape(-1, ch).mean(axis=1)
        return {"waveform": torch.from_numpy(x).unsqueeze(0), "sample_rate": sr}
    except Exception:
        return None

def _shrink_batches(pipe, bs=8):
    for attr in ("embedding_batch_size", "segmentation_batch_size"):
        if hasattr(pipe, attr): setattr(pipe, attr, bs)
    seg = getattr(pipe, "_segmentation", None)
    if seg is not None and hasattr(seg, "batch_size"):
        seg.batch_size = bs

def diarize(wav, cfg):
    from pyannote.audio import Pipeline
    model = cfg.get("model", "pyannote/speaker-diarization-3.1")
    tok = os.environ.get("HF_TOKEN")
    
    try:
        pipe = Pipeline.from_pretrained(model, use_auth_token=tok)
    except TypeError:
        pipe = Pipeline.from_pretrained(model, token=tok)
        
    if pipe is None:
        raise RuntimeError("Pyannote pipeline failed to load. Check HF_TOKEN.")
        
    kw = {k: cfg[k] for k in ("num_speakers", "min_speakers", "max_speakers") if cfg.get(k)}
    loaded_audio = _load_wav(wav)
    cuda = torch.cuda.is_available()
    pipe.to(torch.device("cuda" if cuda else "cpu"))
    _shrink_batches(pipe, bs=8)

    inp = loaded_audio or wav
    out = pipe(inp, **kw)
    ann = getattr(out, "exclusive_speaker_diarization", None) if cfg.get("exclusive", True) else None
    if ann is None: ann = getattr(out, "speaker_diarization", out)
    
    del pipe
    gc.collect()
    if cuda: torch.cuda.empty_cache()
    
    return [(t.start, t.end, s) for t, _, s in ann.itertracks(yield_label=True)]

# [CRITICAL FIX: Restored overlap_regions for Path 5 Baseline]
def overlap_regions(turns):
    ev = sorted([(s, 1) for s, e, _ in turns] + [(e, -1) for s, e, _ in turns])
    regions, active, start = [], 0, None
    for t, d in ev:
        prev = active
        active += d
        if prev < 2 <= active:
            start = t
        elif prev >= 2 > active and start is not None:
            if t > start: regions.append((start, t))
            start = None
    return regions
