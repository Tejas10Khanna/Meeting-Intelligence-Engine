# app.py
import gradio as gr
import json
import os
import tempfile
import time

# Assuming pipeline_ultimate.py is renamed to pipeline.py for the final deployment
import mtg.pipeline_ultimate as pipe 

def process_meeting(audio_file, progress=gr.Progress()):
    if not audio_file:
        return "Please upload an audio file.", "", "", "", "", []
    
    out_dir = os.path.join(tempfile.gettempdir(), f"mtg_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    
    progress(0.1, desc="Transcribing audio with Faster-Whisper...")
    try:
        # Load your single ultimate config
        import yaml
        with open("configs/_models.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            
        progress(0.4, desc="Diarizing speakers & extracting JSON with Qwen-2.5...")
        result = pipe.run(audio_file, cfg, out_dir)
        
    except Exception as e:
        return f"Processing failed: {str(e)}", "", "", "", "", []
        
    progress(0.9, desc="Formatting outputs...")
    
    # Extract fields safely
    raw = getattr(result, "raw_transcript", "No transcript generated.")
    summary = getattr(result, "summary", "No summary generated.")
    
    # Handle lists
    minutes = getattr(result, "minutes", [])
    min_md = "\n".join([f"- {m}" for m in minutes]) if minutes else "No minutes recorded."
    
    decisions = getattr(result, "key_decisions", [])
    dec_md = "\n".join([f"- {d}" for d in decisions]) if decisions else "No decisions recorded."
    
    # Handle Dataframe tasks
    acts = getattr(result, "action_items", [])
    act_table = [[a.get("task", ""), a.get("owner", ""), a.get("deadline", "")] for a in acts]
    
    progress(1.0, desc="Done!")
    return "Complete", raw, summary, min_md, dec_md, act_table

with gr.Blocks(title="Meeting Intelligence Engine", css="#json-output { max-height: 600px; overflow-y: auto; }") as demo:
    gr.Markdown("# Meeting AI\nUpload a recording to extract automated minutes, decisions, and task assignments using Faster-Whisper and Qwen-2.5-AWQ.")
    
    with gr.Row():
        f = gr.Audio(type="filepath", label="Upload Call Center Audio")
        go = gr.Button("Process Meeting", variant="primary")
        
    status = gr.Markdown()
    
    with gr.Tabs():
        with gr.Tab("Summary"):
            summ = gr.Textbox(lines=6, show_copy_button=True, label="Executive Summary")
        with gr.Tab("Minutes & Decisions"):
            with gr.Row():
                md = gr.Markdown(label="Meeting Minutes")
                dec = gr.Markdown(label="Key Decisions")
        with gr.Tab("Action Items"):
            act = gr.Dataframe(headers=["Task", "Owner", "Deadline"], interactive=False)
        with gr.Tab("Raw Transcript"):
            raw = gr.Textbox(lines=15, show_copy_button=True)

    go.click(process_meeting, inputs=[f], outputs=[status, raw, summ, md, dec, act])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)