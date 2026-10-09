# Meeting Intelligence Engine

An end-to-end processing pipeline that transforms raw, multi-speaker meeting audio into structured, schema-compliant JSON artifacts containing executive summaries, chronological minutes, key decisions, and assigned action items.

---

## Architecture Overview

* **ASR:** `Faster-Whisper (Large-v3)` running on CTranslate2 float16 for low-latency acoustic transcription.
* **Speaker Diarization:** `Pyannote.audio` integrated with custom shifted-window temporal stitching to assign speaker identities without VRAM bloat.
* **LLM Structured Extraction:** `Qwen-2.5-7B-Instruct (AWQ Quantized)` executing 4-bit single-pass inference to enforce strict JSON schema compliance.

---

## Engineering Challenges & Solutions

During initial prototyping across legacy baselines (Paths 1–4), several system-level bottlenecks emerged. Here is what broke, why it happened, and how we resolved it:

### 1. Inconsistent Audio Tensors
* **What happened:** Random silent audio truncation and PyTorch tensor shape dimension errors during batching.
* **Root cause:** Uploaded meeting recordings had inconsistent channel layouts (stereo/mono) and sample rates (44.1 kHz, 48 kHz).
* **Fix:** Enforced strict preprocessing in `mtg/audio.py` using `ffmpeg` to normalize all inputs to 16 kHz single-channel mono PCM prior to transcription.

### 2. Whisper Hallucination Cascades
* **What happened:** Transcription insertions spiked during pauses, repeating phrases in loops (up to 181 hallucinated tokens per file) and inflating Word Error Rate (WER).
* **Root cause:** Whisper’s default `condition_on_previous_text=True` flag fed previous context into silent or noisy sections, causing autoregressive looping.
* **Fix:** Disabled previous-text conditioning (`condition_on_previous_text=False`) and introduced Silero-VAD silence thresholding. Hallucinated insertions dropped from 181 to 21 words, driving WER down to 29.09%.

### 3. GPU Out-of-Memory Contention
* **What happened:** Concurrent execution of diarization and the 7B LLM consistently triggered CUDA OOM errors on standard 16 GB GPUs.
* **Root cause:** Standard Pyannote processes the entire audio waveform at once, creating quadratic attention and feature caching that consumed over 12 GB VRAM on longer files.
* **Fix:** Built a shifted-window diarization algorithm that processes audio in overlapping temporal blocks and stitches speaker embeddings across boundaries. This capped diarization VRAM usage to ≤ 4.2 GB, allowing Pyannote and Qwen-2.5-AWQ to co-reside safely on a single GPU.

### 4. Schema Drift & Extraction Latency
* **What happened:** Legacy multi-pass chains (separate LLM calls for summaries, minutes, decisions, and tasks) produced inconsistent formatting, markdown wrapping, and occasional empty dictionaries.
* **Root cause:** Multi-stage prompts compounded latency and allowed schema drift between dependent generation steps.
* **Fix:** Consolidated the entire extraction into a single, comprehensive prompt targeting Qwen-2.5-AWQ. Enforced strict JSON validation at generation time, cutting total runtime to 169.2s (RTF = 0.080) and lifting Task Extraction to 40.0% F1.

---

## Benchmark Comparison

Evaluated on the EN Support-Billing benchmark against human-annotated ground truth:

| Pipeline Variant | Total Time (s) | RTF | WER (%) | Summary F1 | Tasks F1 | Output Integrity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Path 1** (Lean Whisper) | 124.8 | 0.060 | 38.40 | 15.2 | 10.0 | Unstructured Text |
| **Path 2** (Guarded Whisper) | 97.6 | 0.046 | 35.10 | 21.0 | 18.5 | Unstructured Text |
| **Path 3** (Whisper + Pyannote) | 100.5 | 0.240 | 33.80 | 24.5 | 25.0 | Brittle Regex |
| **Path 4** (Parakeet Consensus) | 195.2 | 0.130 | 30.50 | 28.0 | 30.0 | Slow Multi-pass |
| **Path 5 Baseline** (Parakeet + Multi-pass) | 187.9 | 0.106 | 29.20 | 0.0* | 0.0* | Unparsed Markdown |
| **Ultimate Pipeline (Ours)** | **169.2** | **0.080** | **29.09** | **30.8** | **40.0** | **100% Valid JSON** |

*\*Path 5 Baseline produced raw markdown rather than machine-parsable JSON, failing automated evaluation.*

---

## Quickstart

### 1. Prerequisites & Installation

Ensure you have Python 3.10+ and an NVIDIA GPU with at least 16 GB VRAM.

```bash
git clone [https://github.com/your-username/Meeting-Intelligence-Engine.git](https://github.com/your-username/Meeting-Intelligence-Engine.git)
cd Meeting-Intelligence-Engine

# Install dependencies
pip install -r requirements.txt
