# 🩺 AI Doctor Prescription Assistant (POC Phase 1)

**Current Status:** *Model Evaluation & Feasibility Study*

This repository contains the **Phase 1 Proof of Concept (POC)** for an AI-powered assistant designed to transcribe and structure doctor prescriptions from voice input.

Currently, we are evaluating the **Ultravox v0.7** (multimodal GLM-4) model to assess its baseline capabilities in handling raw audio transcription before moving to domain-specific fine-tuning.

## 🎯 Project Goals & Roadmap

The ultimate goal is to build a tool that doctors can use to dictate prescriptions, which the AI will transcribe and format accurately, specifically handling complex medicine names and dosage instructions.

* **Phase 1 (Current):**
    * Evaluate open-source multimodal models (Ultravox, Whisper, etc.).
    * Test baseline performance on medical terminology.
    * Setup local inference pipeline on consumer hardware (Mac MPS / NVIDIA CUDA).
* **Phase 2:**
    * Select the best performing 1-2 models.
    * **Fine-tune** models using a specialized dataset of medicine names, dosages, and medical instructions.
* **Phase 3:**
    * Develop a user-friendly interface for clinical settings.
    * Integrate structured output (JSON/EHR compatible formats).

## 🧪 Current Experiment: Ultravox v0.7

This specific script runs a local instance of `fixie-ai/ultravox-v0_7-glm-4_6`. It is designed to test how well a generic multimodal model handles immediate "listen-and-transcribe" tasks without intermediate ASR steps.

### Technical Highlights
* **Type:** End-to-End Speech-to-Text (Multimodal LLM).
* **Infrastructure:** Runs locally to ensure data privacy (critical for medical data).
* **Patch:** Includes a custom monkey patch to fix compatibility issues with `transformers >= 4.48`.

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
git clone [https://github.com/YOUR_USERNAME/doctor-prescription-poc.git](https://github.com/YOUR_USERNAME/doctor-prescription-poc.git)
cd doctor-prescription-poc

# Create virtual environment
python3 -m venv venv

# Activate, install prerequisits (Mac/Linux)
source venv/bin/activate
pip install -r requirements.txt
# Activate, install prerequisits (Windows)
venv\Scripts\activate
pip install -r requirements.txt
# Run the model
python ultravox.py

Open the local Gradio URL