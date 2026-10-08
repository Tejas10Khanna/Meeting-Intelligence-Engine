import os
import sys
import time
import yaml
import json
import re
import subprocess
from rouge_score import rouge_scorer

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
hf_token = os.environ.get("HF_TOKEN")

def match_items(gold_list, gen_list):
    if not gen_list or not gold_list:
        return 0.0
    matches = 0
    for g in gold_list:
        g_task = g.get("task", "") if isinstance(g, dict) else str(g)
        g_words = set(re.findall(r"\w+", g_task.lower()))
        for p in gen_list:
            p_task = p.get("task", "") if isinstance(p, dict) else str(p)
            p_words = set(re.findall(r"\w+", p_task.lower()))
            overlap = len(g_words & p_words) / max(len(g_words), 1)
            if overlap >= 0.35:
                matches += 1
                break
    prec = matches / len(gen_list) if gen_list else 0
    rec = matches / len(gold_list) if gold_list else 0
    return (2 * (prec * rec) / (prec + rec) * 100) if (prec + rec) > 0 else 0.0

def run_isolated_variant(module_name):
    # Standalone script executed in a dedicated, isolated process
    worker_code = f"""
import os, sys, json, time, yaml
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
hf_token = os.environ.get("HF_TOKEN")
sys.path.append("/kaggle/working/meeting_ai")

audio_path = "data/billing_call/audio.wav"
cfg_path = "configs/p5_consensus_hybrid.yaml"
out_dir = "/tmp/{module_name}_eval"

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

t0 = time.time()
if "{module_name}" == "baseline":
    import mtg.pipeline as pipe
elif "{module_name}" == "merged":
    import mtg.pipeline_merged as pipe
elif "{module_name}" == "ultimate":
    import mtg.pipeline_ultimate as pipe

res = pipe.run(audio_path, cfg, out_dir)
elapsed = time.time() - t0

def get_field(obj, *keys):
    for k in keys:
        if isinstance(obj, dict) and k in obj: return obj[k]
        if hasattr(obj, k): return getattr(obj, k)
    return None

data = {{
    "elapsed": elapsed,
    "summary": get_field(res, "summary", "executive_summary") or "",
    "minutes": get_field(res, "minutes", "meeting_minutes") or [],
    "decisions": get_field(res, "key_decisions", "decisions") or [],
    "actions": get_field(res, "action_items", "actions", "tasks") or []
}}

with open("/tmp/result_{module_name}.json", "w", encoding="utf-8") as f:
    json.dump(data, f)
"""
    worker_path = f"/tmp/worker_{module_name}.py"
    with open(worker_path, "w", encoding="utf-8") as f:
        f.write(worker_code)
    
    # Process isolation: OS reclaims 100% of VRAM on completion
    ret = subprocess.run([sys.executable, "-u", worker_path], check=False)
    if ret.returncode != 0:
        return None
        
    result_path = f"/tmp/result_{module_name}.json"
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def main():
    gold_path = "data/billing_call/gold.json"
    with open(gold_path, "r", encoding="utf-8") as f:
        gold = json.load(f)
        
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    variants = [
        ("Baseline (Path 5)", "baseline"),
        ("Merged Prompt (Parakeet)", "merged"),
        ("Ultimate (FastWhisper+Merged)", "ultimate")
    ]
    
    results = []
    for display_name, mod in variants:
        print(f"\n=======================================================")
        print(f"▶️  Evaluating: {display_name}")
        print(f"=======================================================")
        sys.stdout.flush()
        
        data = run_isolated_variant(mod)
        if not data:
            print(f"❌ {display_name} failed execution.")
            results.append({"name": display_name, "time": "ERR", "sum_f1": "-", "min_f1": "-", "ai_f1": "-"})
            continue
            
        elapsed = data["elapsed"]
        gen_sum = data["summary"]
        gen_min = " ".join([str(m) for m in data["minutes"]]) if isinstance(data["minutes"], list) else str(data["minutes"])
        gen_act = data["actions"] if isinstance(data["actions"], list) else []
        
        gold_sum = gold.get("summary", "")
        gold_min = " ".join(gold.get("minutes", []))
        gold_act = gold.get("action_items", [])
        
        sum_f1 = scorer.score(gold_sum, gen_sum)['rougeL'].fmeasure * 100 if gen_sum else 0.0
        min_f1 = scorer.score(gold_min, gen_min)['rougeL'].fmeasure * 100 if gen_min else 0.0
        ai_f1 = match_items(gold_act, gen_act)
        
        print(f"✅ Finished in {elapsed:.1f}s | Summary F1: {sum_f1:.1f} | Minutes F1: {min_f1:.1f} | Tasks F1: {ai_f1:.1f}")
        results.append({
            "name": display_name,
            "time": f"{elapsed:.1f}s",
            "sum_f1": f"{sum_f1:.1f}",
            "min_f1": f"{min_f1:.1f}",
            "ai_f1": f"{ai_f1:.1f}"
        })
        
    print("\n" + "="*85)
    print(format("END-TO-END PIPELINE EVALUATION (QUALITY & SPEED)", "^85"))
    print("="*85)
    print(f"| {'Pipeline Variant':<30} | {'Total Time':<10} | {'Summary F1':<10} | {'Minutes F1':<10} | {'Tasks F1':<8} |")
    print(f"|{'-'*32}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*10}|")
    for r in results:
        print(f"| {r['name']:<30} | {r['time']:<10} | {r['sum_f1']:<10} | {r['min_f1']:<10} | {r['ai_f1']:<8} |")
    print("="*85)

if __name__ == "__main__":
    main()
