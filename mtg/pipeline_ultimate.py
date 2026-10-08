import os
import gc
import json
import re
import torch
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from . import audio, diarize as dz

def _safe_align(segments, turns):
    if not turns:
        return " ".join([getattr(s, 'text', getattr(s, 'word', '')).strip() for s in segments])
    lines, cur_spk, cur_txt = [], None, []
    for seg in segments:
        s_t = getattr(seg, 'start', getattr(seg, 'start_time', 0.0))
        e_t = getattr(seg, 'end', getattr(seg, 'end_time', 0.0))
        txt = getattr(seg, 'text', getattr(seg, 'word', '')).strip()
        
        b_spk, m_ovl = "UNKNOWN", 0.0
        for ds, de, spk in turns:
            ovl = max(0, min(e_t, de) - max(s_t, ds))
            if ovl > m_ovl:
                m_ovl, b_spk = ovl, spk
        
        if b_spk != cur_spk:
            if cur_txt:
                lines.append(f"[{cur_spk}] {' '.join(cur_txt)}")
            cur_spk, cur_txt = b_spk, [txt]
        else:
            cur_txt.append(txt)
    if cur_txt:
        lines.append(f"[{cur_spk}] {' '.join(cur_txt)}")
    return "\n".join(lines)

def run(audio_path, cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    wav_path, dur = audio.validate_and_convert(audio_path, out_dir)
    
    # 1. Faster-Whisper ASR
    print("   [Ultimate] Transcribing with faster-whisper...")
    model = WhisperModel(cfg.get("whisper_size", "large-v3"), device="cuda", compute_type="float16")
    audio_data, _ = sf.read(wav_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    
    segments, _ = model.transcribe(
        audio_data.astype(np.float32), language="en", beam_size=2,
        condition_on_previous_text=False, vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400, speech_pad_ms=400)
    )
    seg_list = list(segments)
    raw_transcript = " ".join([s.text.strip() for s in seg_list])
    
    # Free STT VRAM immediately
    del model
    del audio_data
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. Diarization
    print("   [Ultimate] Performing shifted-window diarization...")
    diarization_turns = dz.diarize(wav_path, cfg)
    aligned_text = _safe_align(seg_list, diarization_turns)
    
    # 3. Unified Generation with Qwen-AWQ
    print("   [Ultimate] Generating documentation with Qwen-2.5-AWQ...")
    m_id = cfg.get("llm_model", "Qwen/Qwen2.5-7B-Instruct-AWQ")
    tok = AutoTokenizer.from_pretrained(m_id)
    llm = AutoModelForCausalLM.from_pretrained(m_id, device_map="cuda", torch_dtype=torch.float16)
    
    prompt = f"""<|im_start|>system
You are an executive meeting assistant. Extract meeting records from the transcript and output ONLY valid JSON matching this schema:
{{
  "summary": "string",
  "minutes": ["string"],
  "key_decisions": ["string"],
  "action_items": [{{"task": "string", "owner": "string", "deadline": "string"}}]
}}
Do not write explanations, markdown backticks, or notes. Output JSON directly.<|im_end|>
<|im_start|>user
Transcript:
{aligned_text}
<|im_end|>
<|im_start|>assistant
"""
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = llm.generate(**inputs, max_new_tokens=1024, temperature=0.1, do_sample=False)
    response = tok.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # Robust JSON parsing
    parsed_data = {}
    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            parsed_data = json.loads(json_match.group(0))
    except Exception:
        pass
    
    if not parsed_data:
        parsed_data = {
            "summary": response[:500],
            "minutes": [line for line in response.split("\n") if line.strip()][:5],
            "key_decisions": [],
            "action_items": []
        }
        
    del llm, tok, inputs, outputs
    gc.collect()
    torch.cuda.empty_cache()
    
    class UltimateRecord:
        def __init__(self, d, raw):
            self.raw_transcript = raw
            self.summary = d.get("summary", "")
            self.minutes = d.get("minutes", [])
            self.key_decisions = d.get("key_decisions", d.get("decisions", []))
            self.action_items = d.get("action_items", d.get("actions", []))
            
    return UltimateRecord(parsed_data, raw_transcript)
