# 🩺 AI Doctor Prescription Assistant (POC Phase 1)

**Current Status:** *Model Evaluation & Feasibility Study*

This repository contains the **Phase 1 Proof of Concept (POC)** for an AI-powered assistant designed to transcribe and structure doctor prescriptions from voice input.

Currently, we are evaluating the **Qwen2-Audio-7B-Instruct** model to assess its baseline capabilities in handling raw audio transcription and multimodal instruction following before moving to domain-specific fine-tuning.

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

## 🧪 Current Experiment: Qwen2-Audio-7B-Instruct

This specific script runs a local instance of `Qwen/Qwen2-Audio-7B-Instruct`. It is designed to test how well a state-of-the-art multimodal model handles immediate "listen-and-transcribe" tasks and chat-based audio interaction without intermediate ASR steps.

### Technical Highlights
* **Type:** Multimodal Audio-Text LLM (Direct audio understanding).
* **Infrastructure:** Runs locally to ensure data privacy (critical for medical data).
* **Hardware Support:** Optimized for Mac (MPS) using float16 for efficiency, with auto-fallback to CPU or CUDA.

## 🛠️ Setup & Installation

Follow these steps to run the evaluation environment locally.

### 1. Prerequisites
* **Python 3.10+**
* **FFmpeg** (Required for audio processing):
    * *Mac:* `brew install ffmpeg`
    * *Windows:* Download from [ffmpeg.org](https://ffmpeg.org/) and add to PATH.

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
python pretrained_expt_model/qwen_audio.py