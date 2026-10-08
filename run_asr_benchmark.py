import os
import sys
import time
import re
import yaml
import jiwer
import torch
import soundfile as sf
import numpy as np
import logging

# Ensure standard progress bars for downloads are not suppressed
logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.INFO)

sys.path.append("/kaggle/working/meeting_ai")
from mtg import stt

def clean(text):
    text = re.sub(r"\d{2}:\d{2}:\d{2}", " ", text)
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(text.split())

def main():
    work_dir = "/kaggle/working/meeting_ai"
    audio_path = os.path.join(work_dir, "data/billing_call/audio.wav")
    ref_path = os.path.join(work_dir, "data/billing_call/ref_transcript.txt")
    
    if not os.path.exists(audio_path):
        print(f" ERROR: Audio file not found at {audio_path}")
        sys.exit(1)
        
    duration = sf.info(audio_path).duration
    print(f"🎙️ Audio Duration: {duration:.2f}s")
    
    with open(ref_path, "r", encoding="utf-8") as f:
        ref_clean = clean(f.read())
        
    # TEST A: Faster-Whisper
    print("\n Running faster-whisper")
    start = time.time()
    
    # [FIX: Load raw array via soundfile to bypass PyAV metadata_errors bug]
    print(" Loading audio array into memory...")
    audio_data, sr = sf.read(audio_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    audio_data = audio_data.astype(np.float32)
    
    from faster_whisper import WhisperModel
    # Model download progress will automatically show here
    fw_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    
    segs, info = fw_model.transcribe(audio_data, beam_size=1)
    
    fw_text_list = []
    # [FEATURE: Real-time progress tracker on a single updating line]
    for seg in segs:
        fw_text_list.append(seg.text)
        print(f"\rTranscribing: {seg.end:.1f}s / {duration:.1f}s processed", end="", flush=True)
        
    print() # Clear line after progress finishes
    fw_text = " ".join(fw_text_list)
    t_fw = time.time() - start
    
    fw_wer = jiwer.wer(ref_clean, clean(fw_text))
    fw_det = jiwer.process_words(ref_clean, clean(fw_text))
    del fw_model
    torch.cuda.empty_cache()

    # TEST B: Baseline Path 5 (Parakeet/Whisper)
    print("\n Running Path 5")
    cfg_path = os.path.join(work_dir, "configs/p5_consensus_hybrid.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    start = time.time()
    sc = cfg["stt"]
    os.chdir(work_dir)
    
    if sc["engine"] == "parakeet":
        # Parakeet/NeMo handles its own download/loading progress bars
        baseline_segs = stt.words_to_segments(stt.transcribe_parakeet(audio_path, sc))
    else:
        baseline_segs = stt.transcribe_whisper(audio_path, sc, None)
        
    base_text = " ".join(s.text for s in baseline_segs)
    t_base = time.time() - start
    
    base_wer = jiwer.wer(ref_clean, clean(base_text))
    base_det = jiwer.process_words(ref_clean, clean(base_text))
    stt.free(whisper=True, nemo=True)
    torch.cuda.empty_cache()

    # Print Report
    print("\n" + "="*80)
    print(format("ASR ACCURACY & SPEED COMPARISON", "^80"))
    print("="*80)
    print(f"| {'Engine':<20} | {'Time (s)':<8} | {'RTF':<6} | {'WER (%)':<7} | {'Hits':<6} | {'Sub':<5} | {'Del':<5} | {'Ins':<5} |")
    print(f"|{'-'*22}|{'-'*10}|{'-'*8}|{'-'*9}|{'-'*8}|{'-'*7}|{'-'*7}|{'-'*7}|")
    print(f"| {'faster-whisper':<20} | {t_fw:<8.2f} | {t_fw/duration:<6.3f} | {fw_wer*100:<7.2f} | {fw_det.hits:<6} | {fw_det.substitutions:<5} | {fw_det.deletions:<5} | {fw_det.insertions:<5} |")
    print(f"| {'Baseline (Path 5)':<20} | {t_base:<8.2f} | {t_base/duration:<6.3f} | {base_wer*100:<7.2f} | {base_det.hits:<6} | {base_det.substitutions:<5} | {base_det.deletions:<5} | {base_det.insertions:<5} |")
    print("="*80)

if __name__ == "__main__":
    main()
