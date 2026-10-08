"""LLM backends behind one interface + robust JSON parsing."""
import gc
import json
import os
import re

_CACHE = {}


class LLM:
    def chat(self, system, user, max_new_tokens=2048, temperature=0.0):
        raise NotImplementedError

    def unload(self):
        pass


class HFLLM(LLM):
    """Local Hugging Face model, optionally 4-bit (bitsandbytes). Works on Kaggle/Colab T4."""

    def __init__(self, model_id, load_4bit=True):
        self.model_id, self.load_4bit = model_id, load_4bit
        self.tok = self.model = None

    def _load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        kw = dict(device_map="auto", torch_dtype=torch.float16)
        if self.load_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kw)

    def chat(self, system, user, max_new_tokens=2048, temperature=0.0):
        import torch
        self._load()
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        gen = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                   pad_token_id=self.tok.eos_token_id)
        if temperature > 0:
            gen["temperature"] = temperature
        with torch.no_grad():
            out = self.model.generate(**enc, **gen)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def unload(self):
        import torch
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()


class OpenAICompatLLM(LLM):
    """Any OpenAI-compatible server (vLLM, Ollama, llama.cpp server, hosted APIs)."""

    def __init__(self, model_id, base_url, api_key_env="OPENAI_API_KEY"):
        self.model_id, self.base_url = model_id, base_url.rstrip("/")
        self.key = os.environ.get(api_key_env, "none")

    def chat(self, system, user, max_new_tokens=2048, temperature=0.0):
        import requests
        r = requests.post(f"{self.base_url}/chat/completions",
                          headers={"Authorization": f"Bearer {self.key}"},
                          json={"model": self.model_id, "temperature": temperature, "max_tokens": max_new_tokens,
                                "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": user}]}, timeout=600)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


def get_llm(name, llm_cfgs):
    if name not in _CACHE:
        c = dict(llm_cfgs[name])
        backend = c.pop("backend", "hf")
        _CACHE[name] = HFLLM(**c) if backend == "hf" else OpenAICompatLLM(**c)
    return _CACHE[name]


def parse_json(text):
    """Extract the first JSON object/array from model output (handles ``` fences and chatter)."""
    t = re.sub(r"```(?:json)?", "", text).strip()
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = t.find(open_c), t.rfind(close_c)
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no valid JSON found in model output")


def chat_json(llm, system, user, retries=2, **kw):
    last = None
    for k in range(retries + 1):
        raw = llm.chat(system, user if k == 0 else user + "\n\nReturn ONLY valid JSON, nothing else.", **kw)
        try:
            return parse_json(raw)
        except ValueError as e:
            last = e
    raise ValueError(f"LLM did not return valid JSON after {retries + 1} attempts: {last}")
