# Hotfix v2 - drop-in files for your /kaggle/working/meeting_ai
Copy over your working copy (never edit /kaggle/input - it is read-only):
    unzip -o hotfix_v2.zip -d /kaggle/working/meeting_ai
Replaced: mtg/stt.py, mtg/consensus.py, eval/metrics.py, eval/run_eval.py. New: mtg/doctor.py, eval/der.py,
eval/prep_ami_hf.py, tests/test_hotfix.py. Compatible with the pyannote/stage-cache patch (uses its stt.free and free_all).
Does NOT touch mtg/diarize.py, so keep your own torch.load workaround or set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1.
Then:  python -m pytest tests -q   and   python -m mtg.doctor
