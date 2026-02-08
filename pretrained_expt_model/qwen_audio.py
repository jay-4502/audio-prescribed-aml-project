import torch
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# Force the use of the Mac GPU
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"

def run_qwen_audio_mac(audio_path, prompt):
    print(f"🚀 Launching model on {DEVICE}...")
    
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    # Load model with MPS-friendly settings
    # device_map="auto" allows accelerate to handle memory management better
    try:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,  # Use float16 for better MPS compatibility
            device_map="auto",
            low_cpu_mem_usage=True
        )
    except RuntimeError as e:
        if "Invalid buffer size" in str(e):
            print(f"\n❌ Error: The model {MODEL_ID} (approx 14GB) is too large for your Mac's current memory configuration.")
            print("Try closing other applications to free up RAM, or use a smaller model.")
            raise e
        else:
            raise e

    # Standardize audio for Qwen (16kHz, Mono)
    audio, sr = librosa.load(audio_path, sr=16000)

    # Format for Qwen2-Audio
    messages = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": prompt}
        ]}
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    inputs = processor(
        text=text,
        audio=audio,
        return_tensors="pt",
        sampling_rate=sr
    ).to(DEVICE)

    print("🧠 Inference in progress...")
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=128)
    
    # Clean output
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

    return response

if __name__ == "__main__":
    # Ensure your wav file is in the same folder
    print(run_qwen_audio_mac("prescription_1.wav", "Describe this audio."))