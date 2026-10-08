"""Speech-to-text engines.
Whisper has two interchangeable backends:
  * ct2 : faster-whisper / CTranslate2 (fast, keeps per-word confidence)
  * hf  : Hugging Face Transformers Whisper (pure PyTorch; slower, no confidence) - automatic fallback.
Audio is passed as a NumPy array (our pipeline already converts to 16 kHz mono WAV), which bypasses PyAV completely,
so an outdated `av` package on Kaggle/Colab can no longer break transcription.
CTranslate2 aborts the WHOLE PROCESS (kills the notebook kernel) if cuDNN is missing, so it is probed in a subprocess first."""
import gc
import os
import subprocess
import sys
import tempfile
import wave

from .audio import duration, slice_audio
from .types import Segment, Word

_WH, _NEMO, _HF, _PROBE = {}, {}, {}, {}


def free(whisper=True, nemo=True):
    """Release GPU memory held by cached STT models (call before loading the LLMs)."""
    if whisper:
        _WH.clear()
        _HF.clear()
    if nemo:
        _NEMO.clear()
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def load_wav_array(path):
    """16-bit PCM WAV -> (float32 mono array, sample_rate). Returns None if the file is not a plain PCM wav."""
    try:
        import numpy as np
        with wave.open(path, "rb") as w:
            sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            raw = w.readframes(n)
        if sw != 2:
            return None
        x = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)
        return x, sr
    except Exception:
        return None


def ct2_ok(force=False):
    """Run a tiny GPU transcription in a SUBPROCESS. A broken cuDNN/CUDA setup makes CTranslate2 abort the process,
    which would otherwise kill the notebook kernel with no Python traceback. Returns (ok, message)."""
    if "ct2" in _PROBE and not force:
        return _PROBE["ct2"]
    code = ("import numpy as np\n"
            "from faster_whisper import WhisperModel\n"
            "m = WhisperModel('tiny.en', device='cuda', compute_type='float16')\n"
            "s, _ = m.transcribe(np.zeros(32000, dtype='float32'), language='en')\n"
            "list(s)\nprint('CT2_OK')\n")
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
        ok, msg = r.returncode == 0 and "CT2_OK" in r.stdout, (r.stderr or r.stdout).strip()[-500:]
    except Exception as e:                                          # timeout, no internet for the tiny model, ...
        ok, msg = False, str(e)[:300]
    _PROBE["ct2"] = (ok, msg)
    return _PROBE["ct2"]


def _pick_backend(cfg):
    b = cfg.get("backend", "auto")
    if b in ("ct2", "hf"):
        return b
    if not _cuda():
        return "ct2"                                                # CPU int8 works without cuDNN
    ok, msg = ct2_ok()
    if not ok:
        print(f"[stt] faster-whisper GPU probe failed, using Hugging Face Whisper instead. Reason: {msg[-250:]}")
    return "ct2" if ok else "hf"


def _ct2_model(cfg):
    dev = cfg.get("device") or ("cuda" if _cuda() else "cpu")
    ct = cfg.get("compute_type", "float16")
    key = (cfg["model"], dev, ct)
    if key not in _WH:
        from faster_whisper import WhisperModel
        last = None
        for c in ([ct, "int8_float16", "int8"] if dev == "cuda" else ["int8"]):
            try:
                _WH[key] = WhisperModel(cfg["model"], device=dev, compute_type=c)
                break
            except Exception as e:
                last = e
        else:
            raise RuntimeError(f"could not load Whisper model: {last}")
    return _WH[key]


def _transcribe_ct2(wav, cfg, initial_prompt, offset):
    model = _ct2_model(cfg)
    arr = load_wav_array(wav)
    audio = arr[0] if arr is not None and arr[1] == 16000 else wav
    segs, _ = model.transcribe(
        audio, language="en", beam_size=cfg.get("beam_size", 5), vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500), word_timestamps=True,
        condition_on_previous_text=False, initial_prompt=initial_prompt)
    out = []
    for s in segs:
        text = s.text.strip()
        if not text:
            continue
        if s.no_speech_prob > 0.8 and s.avg_logprob < -1.0:         # likely hallucination on noise
            continue
        words = [Word(w.word.strip(), w.start + offset, w.end + offset, w.probability) for w in (s.words or [])]
        out.append(Segment(id=len(out), start=s.start + offset, end=s.end + offset, text=text,
                           avg_logprob=s.avg_logprob, no_speech_prob=s.no_speech_prob, words=words))
    return out


def _transcribe_hf(wav, cfg, initial_prompt, offset):
    """Pure-PyTorch Whisper. No per-word confidence and no prompt support (simple, robust fallback)."""
    import torch
    from transformers import pipeline
    name = cfg.get("hf_model", "openai/whisper-large-v3")
    if name not in _HF:
        gpu = torch.cuda.is_available()
        _HF[name] = pipeline("automatic-speech-recognition", model=name, device=0 if gpu else -1,
                             torch_dtype=torch.float16 if gpu else torch.float32)
    arr = load_wav_array(wav)
    if arr is None:
        raise RuntimeError("the hf backend needs a 16-bit PCM wav (the pipeline converts uploads automatically)")
    out = _HF[name]({"raw": arr[0], "sampling_rate": arr[1]}, chunk_length_s=30, stride_length_s=5,
                    batch_size=cfg.get("hf_batch", 8), return_timestamps="word",
                    generate_kwargs={"language": "english", "task": "transcribe"})
    words = []
    for c in out.get("chunks", []):
        t = c["text"].strip()
        if not t:
            continue
        s, e = c["timestamp"]
        s = 0.0 if s is None else s
        words.append(Word(t, s + offset, (e if e is not None else s + 0.3) + offset, 1.0))
    return words_to_segments(words)


def transcribe_whisper(wav, cfg, initial_prompt=None, offset=0.0):
    """Return list[Segment] (word timestamps; confidence only with the ct2 backend)."""
    if _pick_backend(cfg) == "hf":
        return _transcribe_hf(wav, cfg, initial_prompt, offset)
    return _transcribe_ct2(wav, cfg, initial_prompt, offset)


def transcribe_parakeet(wav, cfg):
    """Return list[Word] from NVIDIA Parakeet-TDT. Long audio is processed in fixed chunks."""
    import nemo.collections.asr as nemo_asr
    name = cfg.get("model", "nvidia/parakeet-tdt-0.6b-v2")
    if name not in _NEMO:
        _NEMO[name] = nemo_asr.models.ASRModel.from_pretrained(model_name=name)
    model, chunk, words = _NEMO[name], cfg.get("chunk_s", 240), []
    total = duration(wav)
    with tempfile.TemporaryDirectory() as td:
        off = 0.0
        while off < total:
            piece = slice_audio(wav, off, min(off + chunk, total), os.path.join(td, "c.wav"))
            h = model.transcribe([piece], timestamps=True)[0]
            for w in h.timestamp["word"]:
                words.append(Word(w["word"], w["start"] + off, w["end"] + off))
            off += chunk
    return words


def words_to_segments(words, max_gap=0.8, max_words=35):
    """Group a flat word list into segments (used for Parakeet and the hf Whisper backend)."""
    segs, cur = [], []

    def flush():
        if cur:
            segs.append(Segment(id=len(segs), start=cur[0].start, end=cur[-1].end,
                                text=" ".join(w.text for w in cur), words=list(cur)))
            cur.clear()

    for w in words:
        if cur and (w.start - cur[-1].end > max_gap or len(cur) >= max_words):
            flush()
        cur.append(w)
    flush()
    return segs
