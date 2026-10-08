"""Input validation and conversion. Raises AudioError with a user-readable message."""
import json
import os
import re
import subprocess

ALLOWED = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".mp4", ".webm", ".mkv"}


class AudioError(Exception):
    pass


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def duration(path):
    p = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path])
    return float(json.loads(p.stdout)["format"]["duration"])


def validate_and_convert(path, out_dir, min_seconds=1.0, max_hours=4.0, silence_db=-60.0):
    """Validate the upload and convert to 16 kHz mono WAV. Returns (wav_path, duration_s)."""
    if not path or not os.path.exists(path):
        raise AudioError("No file was provided, or the file could not be found.")
    if os.path.getsize(path) == 0:
        raise AudioError("The uploaded file is empty (0 bytes).")
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED:
        raise AudioError(f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(ALLOWED))}.")
    try:
        dur = duration(path)
    except Exception:
        raise AudioError("The file could not be read as audio. It may be corrupt or not an audio/video file.")
    if dur < min_seconds:
        raise AudioError(f"The recording is too short ({dur:.1f}s) to contain a meeting.")
    if dur > max_hours * 3600:
        raise AudioError(f"The recording is longer than the supported maximum of {max_hours} hours.")
    os.makedirs(out_dir, exist_ok=True)
    wav = os.path.join(out_dir, "audio_16k.wav")
    r = _run(["ffmpeg", "-y", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", wav])
    if r.returncode != 0 or not os.path.exists(wav):
        raise AudioError("The audio could not be decoded: " + r.stderr.strip()[-200:])
    v = _run(["ffmpeg", "-i", wav, "-af", "volumedetect", "-f", "null", "-"])
    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", v.stderr)
    if m and float(m.group(1)) < silence_db:
        raise AudioError("The recording appears to be silent, so there is no speech to transcribe.")
    return wav, dur


def slice_audio(wav, start, end, out_path):
    _run(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0, start):.3f}", "-to", f"{end:.3f}", "-i", wav, out_path])
    return out_path
