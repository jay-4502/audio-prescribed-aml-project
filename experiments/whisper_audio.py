"""
Whisper Fine-tuned Model Live Interface
=========================================
Provides a real-time Gradio interface for the fine-tuned Whisper model.
Loads a custom trained Whisper-Large-V3-Turbo model and uses it for continuous audio transcription.
Features live audio streaming, chunk-based processing for low latency, and transcript accumulation.
Supports GPU (CUDA), Apple Silicon (MPS), and CPU backends with appropriate precision handling.
"""

import torch
import numpy as np
import gradio as gr
from transformers import AutoProcessor, WhisperForConditionalGeneration
import librosa
import time
import os
from pathlib import Path

# --- Configuration ---
# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = str(PROJECT_ROOT / "models" / "whisper-large-v3-turbo-finetuned")
PROCESSOR_ID = "openai/whisper-large-v3-turbo" # Keep the base processor for vocab/audio parsing
# Force MPS if available, otherwise CPU (or CUDA if present)
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

SAMPLE_RATE = 16000
CHUNK_PROCESSING_INTERVAL = 1.0 # Process every 1 second of new audio (adjust for latency vs load)

print(f"🚀 Initializing Whisper-Live on {DEVICE}...")

# --- Model Loading ---
def load_model():
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(PROCESSOR_ID)
    
    print(f"Loading custom fine-tuned model from {MODEL_ID}...")
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32, # Force float32 for stability on MPS
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(DEVICE)
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
        
    return processor, model

# Load globally
try:
    PROCESSOR, MODEL = load_model()
except Exception as e:
    print(f"Failed to load model: {e}")
    PROCESSOR, MODEL = None, None

# --- State ---
class TranscriptionState:
    def __init__(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_process_time = 0
        self.full_transcript = ""
        self.sr = None

# --- Processing ---
def resample_audio(audio, orig_sr, target_sr=16000):
    if orig_sr != target_sr:
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    return audio

def perform_inference(audio_data):
    # Check for silence (simple energy threshold)
    # RMS amplitude
    rms = np.sqrt(np.mean(audio_data**2))
    print(f"DEBUG: Audio RMS: {rms:.4f}") # Debug print
    if rms < 0.01: # Increased threshold
        print("DEBUG: Silence detected.")
        return ""

    import re

    inputs_dict = PROCESSOR(
        audio_data, 
        sampling_rate=SAMPLE_RATE, 
        return_tensors="pt"
    ).to(DEVICE)
    
    input_features = inputs_dict.input_features
    # Create attention mask (1s for valid data) - though for raw audio it's all valid, 
    # but Whisper sometimes needs it or it defaults to strange behavior.
    # Actually, PROCESSOR doesn't always return attention_mask for audio if not padded.
    # We will trust the processor but ensure we handle the output.

    if DEVICE != "cpu":
        # Keep float32 for stability on MPS
        input_features = input_features.to(dtype=torch.float32)
    
    # Generate transcription
    try:
        # Force English
        forced_decoder_ids = PROCESSOR.get_decoder_prompt_ids(language="en", task="transcribe")
        
        predicted_ids = MODEL.generate(
            input_features, 
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=128,
            temperature=0.2, # Slight temp can help avoid loops compared to 0.0 sometimes
            condition_on_prev_tokens=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1, 
            logprob_threshold=-1.0, 
            no_speech_threshold=0.4, # Stricter speech detection
        )
        transcription = PROCESSOR.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Post-processing filter for hallucinations
        # Filter out "!!!!!!!!", ".......", or repeated small phrases
        if re.match(r'^[\W_]+$', transcription): # Only punctuation/symbols
             print(f"DEBUG: Filtered hallucination (punctuation): {transcription}")
             return ""
        
        if len(transcription) > 10 and len(set(transcription)) < 4: # e.g. "So So So"
             print(f"DEBUG: Filtered hallucination (repetitive): {transcription}")
             return ""

        return transcription
    except Exception as e:
        return f"[Error: {e}]"

def process_stream(audio_chunk, state):
    if state is None:
        state = TranscriptionState()
    
    if audio_chunk is None:
        return state, state.full_transcript
        
    sr, data = audio_chunk
    state.sr = sr
    
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
        
    # Normalize and Append
    if data.dtype != np.float32:
        data = data.astype(np.float32)
        if np.max(np.abs(data)) > 1.0:
            data = data / 32768.0

    state.audio_buffer = np.concatenate((state.audio_buffer, data))
    
    # Check if we should process (throttle to avoid overload)
    current_time = time.time()
    if current_time - state.last_process_time > CHUNK_PROCESSING_INTERVAL:
        # Resample the WHOLE buffer for better context (or could use a window)
        # For true "live" with Whisper (which is seq2seq), passing the growing buffer 
        # is the simplest way to get corrected context, though it gets slower.
        # For a demo, we'll cap buffer at say 30s? Or just let it grow.
        # Let's resample and transcribe.
        
        resampled_audio = resample_audio(state.audio_buffer, sr, SAMPLE_RATE)
        
        transcript = perform_inference(resampled_audio)
        state.full_transcript = transcript
        state.last_process_time = current_time
        
    return state, state.full_transcript

def on_clear(state):
    return TranscriptionState(), ""

# --- UI ---
with gr.Blocks(title="Whisper Live Transcription") as demo:
    gr.Markdown(f"# 🎙️ Live Transcription with {MODEL_ID}")
    
    state = gr.State(TranscriptionState())
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone"], 
                type="numpy", 
                streaming=True,
                label="Microphone"
            )
            clear_btn = gr.Button("Clear Transcript")
            
        with gr.Column(scale=2):
            transcript_output = gr.Textbox(
                label="Live Transcript", 
                lines=10,
                placeholder="Start speaking..."
            )

    # Event: Streaming
    audio_input.stream(
        fn=process_stream,
        inputs=[audio_input, state],
        outputs=[state, transcript_output],
        show_progress=False
    )
    
    # Event: Clear
    clear_btn.click(
        fn=on_clear,
        inputs=[state],
        outputs=[state, transcript_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
