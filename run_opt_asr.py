import os
import sys
import time
import re
import jiwer
import torch
import soundfile as sf
import numpy as np

def clean(text):
    text = re.sub(r"\d{2}:\d{2}:\d{2}", " ", text)
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(text.split())

def main():
    work_dir = "/kaggle/working/meeting_ai"
    audio_path = os.path.join(work_dir, "data/billing_call/audio.wav")
    ref_path = os.path.join(work_dir, "data/billing_call/ref_transcript.txt")
    
    duration = sf.info(audio_path).duration
    
    with open(ref_path, "r", encoding="utf-8") as f:
        ref_clean = clean(f.read())
        
    print(" Running OPTIMIZED faster-whisper")
    start = time.time()
    
    audio_data, sr = sf.read(audio_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    audio_data = audio_data.astype(np.float32)
    
    from faster_whisper import WhisperModel
    fw_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    
    # --- THE OPTIMIZATIONS ---
    segs, info = fw_model.transcribe(
        audio_data, 
        language="en", # Skips 2-3 seconds of auto-detect processing
        beam_size=2, # Recovers dropped words with minimal speed penalty
        condition_on_previous_text=False, # Kills the 181 hallucinated insertions
        vad_filter=True, 
        vad_parameters=dict(min_silence_duration_ms=400, speech_pad_ms=400) # Prevents clipping
    )
    
    fw_text_list = []
    for seg in segs:
        fw_text_list.append(seg.text)
        print(f"\r Transcribing: {seg.end:.1f}s / {duration:.1f}s processed", end="", flush=True)
        
    print()
    fw_text = " ".join(fw_text_list)
    t_fw = time.time() - start
    
    fw_wer = jiwer.wer(ref_clean, clean(fw_text))
    fw_det = jiwer.process_words(ref_clean, clean(fw_text))
    
    del fw_model
    torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print(format("OPTIMIZED FASTER-WHISPER RESULTS", "^80"))
    print("="*80)
    print(f"| {'Engine':<20} | {'Time (s)':<8} | {'RTF':<6} | {'WER (%)':<7} | {'Hits':<6} | {'Sub':<5} | {'Del':<5} | {'Ins':<5} |")
    print(f"|{'-'*22}|{'-'*10}|{'-'*8}|{'-'*9}|{'-'*8}|{'-'*7}|{'-'*7}|{'-'*7}|")
    print(f"| {'faster-whisper (Opt)':<20} | {t_fw:<8.2f} | {t_fw/duration:<6.3f} | {fw_wer*100:<7.2f} | {fw_det.hits:<6} | {fw_det.substitutions:<5} | {fw_det.deletions:<5} | {fw_det.insertions:<5} |")
    print("="*80)

if __name__ == "__main__":
    main()
