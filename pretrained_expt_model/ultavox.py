import gradio as gr
import torch
import librosa
import numpy as np
import transformers.processing_utils
from transformers import AutoModel, AutoProcessor

# --- MONKEY PATCH START ---
# This bypasses the strict type check in Transformers 4.48+ that breaks Ultravox
# We check for both naming conventions (ProcessorMixin is the new name)
mixin_class = getattr(transformers.processing_utils, "ProcessorMixin", 
              getattr(transformers.processing_utils, "ProcessingMixin", None))

if mixin_class:
    _original_check = mixin_class.check_argument_for_proper_class

    def _patched_check(self, attribute_name, value):
        # Allow WhisperProcessor to pass as audio_processor
        if attribute_name == "audio_processor" and "WhisperProcessor" in str(type(value)):
            return
        return _original_check(self, attribute_name, value)

    mixin_class.check_argument_for_proper_class = _patched_check
    print(f"✅ Patch applied to {mixin_class.__name__}")
# --- MONKEY PATCH END ---

# 1. Setup Configuration
# Check for MPS (Apple Silicon) or CUDA, fallback to CPU
if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float16
elif torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

MODEL_ID = "fixie-ai/ultravox-v0_7-glm-4_6"

print(f"🚀 Loading model on {DEVICE} with {DTYPE}...")

# 2. Load Model & Processor
processor = AutoProcessor.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True
)

# Note: We use 'dtype' instead of 'torch_dtype' to fix the warning
model = AutoModel.from_pretrained(
    MODEL_ID, 
    trust_remote_code=True, 
    dtype=DTYPE
).to(DEVICE)

print("✅ Model loaded successfully!")

# 3. Define Inference Function
def run_inference(audio_filepath):
    if audio_filepath is None:
        return "Please record audio first."
    
    print(f"Processing audio: {audio_filepath}")
    
    # Load audio at 16kHz (Required by Ultravox)
    audio, sr = librosa.load(audio_filepath, sr=16000)
    
    # Prompt: Use "<|audio|>" for chat, or add instruction for transcription
    text_prompt = "<|audio|> Transcribe the spoken audio verbatim."
    
    # Preprocess
    inputs = processor(
        text=text_prompt,
        audios=audio,
        return_tensors="pt",
        sampling_rate=16000
    )
    
    # Move to Device
    inputs["input_ids"] = inputs["input_ids"].to(DEVICE)
    inputs["pixel_values"] = inputs["pixel_values"].to(DEVICE, dtype=DTYPE)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=256,
            do_sample=True,
            temperature=0.6,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id
        )
    
    # Decode
    transcription = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True
    )[0]
    
    return transcription

# 4. Build UI
with gr.Blocks(title="Ultravox Local") as demo:
    gr.Markdown("## 🎙️ Ultravox v0.7 GLM-4 (Local on Mac)")
    gr.Markdown(f"Running on **{DEVICE}**")
    
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record Voice")
        text_output = gr.Textbox(label="Transcription / Response")
    
    btn = gr.Button("Submit", variant="primary")
    btn.click(fn=run_inference, inputs=audio_input, outputs=text_output)

if __name__ == "__main__":
    demo.launch()
