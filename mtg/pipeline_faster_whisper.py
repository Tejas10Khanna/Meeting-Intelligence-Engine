"""
Optimized Pipeline variant utilizing CTranslate2 faster-whisper + Shifted Window Diarization
"""
import os
import torch
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel
from . import audio, diarize as dz, refine as rf, consensus

def run(audio_path, cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    wav_path, duration = audio.validate_and_convert(audio_path, out_dir)
    
    model_size = cfg.get("whisper_size", "large-v3")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    
    audio_data, sr = sf.read(wav_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    audio_data = audio_data.astype(np.float32)
    
    segments, _ = model.transcribe(
        audio_data, 
        language="en", 
        beam_size=2, 
        condition_on_previous_text=False, 
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400, speech_pad_ms=400)
    )
    
    raw_transcript = " ".join([seg.text for seg in segments])
    
    del model
    torch.cuda.empty_cache()
    
    # 3. Shifted-Window Diarization
    diarization_turns = dz.diarize(wav_path, cfg)
    
    # 4. Align Speakers & Generate Consensus 
    aligned_text = rf.align(raw_transcript, diarization_turns)
    result = consensus.generate(aligned_text, cfg)
    
    # 5. Pack result so the evaluator can extract raw_transcript for WER
    class OptRecord:
        def __init__(self, res, raw):
            if isinstance(res, dict):
                self.__dict__.update(res)
            else:
                self.__dict__.update(vars(res))
            self.raw_transcript = raw
            
    return OptRecord(result, raw_transcript)
