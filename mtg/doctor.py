"""Environment doctor: python -m mtg.doctor
Every risky check runs in a SUBPROCESS, so a native crash (e.g. missing cuDNN) is reported instead of killing your kernel."""
import importlib.metadata as md
import os
import shutil
import subprocess
import sys

OK, WARN, FAIL = "PASS", "WARN", "FAIL"


def sub(code, timeout=600):
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stdout + r.stderr).strip()[-400:], r.returncode
    except Exception as e:
        return False, str(e)[:300], -1


def ver(p):
    try:
        return md.version(p)
    except Exception:
        return "not installed"


def checks():
    out = []
    out.append(("ffmpeg + ffprobe", OK if shutil.which("ffmpeg") and shutil.which("ffprobe") else FAIL,
                "", "apt-get install -y ffmpeg"))
    vs = ", ".join(f"{p} {ver(p)}" for p in ("torch", "transformers", "faster-whisper", "ctranslate2", "av",
                                           "pyannote.audio", "bitsandbytes", "gradio"))
    out.append(("package versions", OK, vs, ""))
    ok, msg, _ = sub("import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0), "
                     "[round(x/1e9,1) for x in torch.cuda.mem_get_info()], 'cudnn', torch.backends.cudnn.version())")
    out.append(("GPU visible to PyTorch (free/total GB)", OK if ok and msg.startswith("True") else FAIL, msg,
                "Enable the GPU accelerator and restart the session"))
    wav = ("import wave,os;w=wave.open('/tmp/_d.wav','wb');w.setnchannels(1);w.setsampwidth(2);"
           "w.setframerate(16000);w.writeframes(b'\\0\\0'*16000);w.close();import av;"
           "av.open('/tmp/_d.wav', metadata_errors='ignore').close();print('av ok')")
    ok, msg, _ = sub(wav)
    out.append(("PyAV accepts metadata_errors (only matters if audio is passed as a file path)", OK if ok else WARN, msg,
                "Not needed any more: transcription passes NumPy arrays. If you still want it: pip install -U av "
                "AND restart the kernel (an already-imported old version stays in memory)"))
    ok, msg, rc = sub("from mtg.stt import ct2_ok;print(ct2_ok(True))")
    good = ok and msg.startswith("(True")
    out.append(("faster-whisper / CTranslate2 on GPU", OK if good else FAIL, msg,
                "The pipeline now falls back to Hugging Face Whisper automatically. To fix CT2: look for 'cudnn' in the "
                "message; try `pip install -U nvidia-cudnn-cu12` or pin ctranslate2 to the version matching your "
                "CUDA/cuDNN (4.4.x needs cuDNN 8, >=4.5 needs cuDNN 9). Or set `backend: hf` under stt: to skip the probe"))
    ok, msg, _ = sub("from transformers import pipeline;import numpy as np;"
                     "p=pipeline('automatic-speech-recognition','openai/whisper-tiny.en');"
                     "print(p({'raw':np.zeros(16000,dtype='float32'),'sampling_rate':16000}))")
    out.append(("Hugging Face Whisper (fallback engine)", OK if ok else FAIL, msg, "Check internet is ON and transformers is installed"))
    ok, msg, _ = sub("import os;from huggingface_hub import whoami;print(whoami(token=os.environ.get('HF_TOKEN'))['name'])")
    out.append(("HF_TOKEN valid", OK if ok else WARN, msg, "Create a read token; set HF_TOKEN (Kaggle: Add-ons > Secrets)"))
    for repo, f in (("pyannote/speaker-diarization-3.1", "config.yaml"), ("pyannote/segmentation-3.0", "config.yaml"),
                    ("meta-llama/Llama-3.1-8B-Instruct", "config.json")):
        ok, msg, _ = sub(f"import os;from huggingface_hub import hf_hub_download as h;"
                         f"h('{repo}','{f}',token=os.environ.get('HF_TOKEN'));print('ok')")
        out.append((f"access to {repo}", OK if ok else WARN, "" if ok else msg[-160:],
                    "Open the model page while logged in and accept the terms"))
    ok, msg, _ = sub("import bitsandbytes as b;print(b.__version__)")
    out.append(("bitsandbytes (4-bit LLMs)", OK if ok else FAIL, msg, "pip install -U bitsandbytes"))
    free = shutil.disk_usage(os.getcwd()).free / 1e9
    out.append(("free disk in working dir (GB)", OK if free > 25 else WARN, f"{free:.0f}",
                "Models need 20-35 GB; use pre-quantised 4-bit repos"))
    out.append(("torch.load workaround for pyannote 3.x",
                OK if os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") else WARN,
                os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "unset"),
                "export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 (needed with torch>=2.6 and pyannote.audio 3.x)"))
    return out


def main():
    rows = checks()
    for name, st, detail, hint in rows:
        print(f"[{st}] {name}" + (f"\n        {detail}" if detail else ""))
        if st != OK and hint:
            print(f"        fix: {hint}")
    bad = [r for r in rows if r[1] == FAIL]
    print("\nAll critical checks passed." if not bad else f"\n{len(bad)} critical check(s) failed - fix those first.")


if __name__ == "__main__":
    main()
