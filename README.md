# 🩺 AI Doctor Prescription Assistant (POC Phase 1)

**Current Status:** *Model Evaluation & Feasibility Study with Interactive Prototype*

This repository hosts a **Live Audio Interface** for an AI-powered assistant designed to transcribe doctor prescriptions. It uses a local Gradio web interface to capture microphone input and uses **Qwen2-Audio-7B-Instruct** to transcribe medical dictation into text.

## 🚀 Key Features

* **Interactive Web UI:** Simple "Record" and "Stop" interface powered by Gradio.
* **Local Inference:** Runs entirely on your machine (Mac MPS / CUDA), ensuring **patient data privacy** by never sending audio to the cloud.
* **Auto-Logging:** Automatically saves all transcriptions with timestamps to a local `transcripts.txt` file.
* **Model:** Uses `Qwen/Qwen2-Audio-7B-Instruct`, a state-of-the-art multimodal model capable of understanding direct audio inputs.

## 🎯 Project Goals & Roadmap

The ultimate goal is to build a tool that doctors can use to dictate prescriptions, which the AI will transcribe and format accurately, specifically handling complex medicine names and dosage instructions.

* **Phase 1 (Current):**
    * Evaluate open-source multimodal models (Qwen2-Audio, Ultravox, etc.).
    * Test baseline performance on medical terminology.
    * Setup local inference pipeline on consumer hardware (Mac MPS / NVIDIA CUDA).
* **Phase 2:**
    * Select the best performing 1-2 models.
    * **Fine-tune** models using a specialized dataset of medicine names, dosages, and medical instructions.
* **Phase 3:**
    * Develop a user-friendly interface for clinical settings.
    * Integrate structured output (JSON/EHR compatible formats).

## 🛠️ Setup & Installation

Follow these steps to run the evaluation environment locally.

### 1. Prerequisites
* **Python 3.10+**
* **FFmpeg** (Required for audio processing):
    * *Mac:* `brew install ffmpeg`
    * *Windows:* Download from [ffmpeg.org](https://ffmpeg.org/) and add to PATH.
* **Hardware:**
    * *Mac:* Apple Silicon (M1/M2/M3) with at least **16GB RAM** (Model uses ~14GB).
    * *Windows/Linux:* NVIDIA GPU with 16GB+ VRAM recommended.

### 2. Environment Setup

```bash
# Clone the repository
git clone [https://github.com/jay-4502/doctor-prescription-poc.git](https://github.com/jay-4502/doctor-prescription-poc.git)
cd doctor-prescription-poc

# Create virtual environment
python3 -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the models
# python pretrained_expt_model/ultavox.py
# python pretrained_expt_model/qwen_audio.py
python pretrained_expt_model/qwen_audio_live.py