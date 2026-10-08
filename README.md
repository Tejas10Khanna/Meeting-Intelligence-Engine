# Meeting Assistant - five comparable pipeline paths

Audio -> speech-to-text -> LLM-A refinement -> LLM-B minutes/decisions/action items -> UI + downloads.
Each *path* is a YAML in `configs/`; the code is shared, so paths can be compared fairly.

| Config | Idea |
|---|---|
| p1_lean | Whisper + whole-chunk LLM prompts, no safeguards (baseline) |
| p2_guarded | + glossary, confidence gating, sliding window, hunk-level guardrails, evidence-grounded minutes |
| p3_speaker | p2 + pyannote diarization and speaker-aware owners |
| p4_parakeet_big | Parakeet STT + larger LLM-B (tests model family / size vs. design) |
| p5_consensus_hybrid | Whisper + Parakeet consensus, glossary-biased re-decode, constrained LLM selection, verified minutes |

## Run on Kaggle / Colab
```
pip install -r requirements.txt            # nemo_toolkit[asr] and pyannote are optional (paths 3-5)
export HF_TOKEN=...                        # pyannote + gated Llama; accept model terms on huggingface.co first
python app.py                              # UI (prints a public share link)
python -m eval.run_eval --data data --paths configs/p1_lean.yaml configs/p2_guarded.yaml --out results
python -m pytest tests -q                  # logic tests (no GPU needed)
```
Data layout for evaluation: `data/<id>/audio.wav`, `ref_transcript.txt`, `gold.json` (see `data/_example`).
Edit model ids in the `llms:` block of each config; any OpenAI-compatible server can replace the local models.
