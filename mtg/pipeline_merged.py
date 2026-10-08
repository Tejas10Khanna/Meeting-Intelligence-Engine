import os
import gc
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from . import audio, diarize as dz, stt

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
    
    sc = cfg.get("stt", {})
    print("   [Merged] Running STT...")
    if sc.get("engine") == "parakeet":
        baseline_segs = stt.words_to_segments(stt.transcribe_parakeet(wav_path, sc))
    else:
        baseline_segs = stt.transcribe_whisper(wav_path, sc, None)
        
    raw_transcript = " ".join([getattr(s, 'text', getattr(s, 'word', '')).strip() for s in baseline_segs])
    stt.free(whisper=True, nemo=True)
    gc.collect()
    torch.cuda.empty_cache()
    
    print("   [Merged] Running diarization...")
    diarization_turns = dz.diarize(wav_path, cfg)
    aligned_text = _safe_align(baseline_segs, diarization_turns)
    
    print("   [Merged] Generating documentation with Qwen-2.5-AWQ...")
    m_id = cfg.get("llm_model", "Qwen/Qwen2.5-7B-Instruct-AWQ")
    tok = AutoTokenizer.from_pretrained(m_id)
    llm = AutoModelForCausalLM.from_pretrained(m_id, torch_dtype=torch.float16, device_map="cuda")
    
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
    
    class MergedRecord:
        def __init__(self, d, raw):
            self.raw_transcript = raw
            self.summary = d.get("summary", "")
            self.minutes = d.get("minutes", [])
            self.key_decisions = d.get("key_decisions", d.get("decisions", []))
            self.action_items = d.get("action_items", d.get("actions", []))

    return MergedRecord(parsed_data, raw_transcript)
