# 🩺 AI Doctor Prescription Assistant (POC Phase 1)

**Current Status:** *Model Evaluation & Feasibility Study with Interactive Prototype*

This project aims to build an AI-powered assistant to transcribe doctor prescriptions accurately. It currently features a **Live Audio Interface** for testing audio transcription models locally.

## 📂 Project Structure

- **`experiments/`**: Contains experimental scripts and the initial prototype using `Qwen2-Audio` and `Ultravox`.  
  👉 **[See Experiments Documentation](experiments/README.md)** for details on running the local Gradio interface and model tests.

- **`backend/`**: (Coming soon) Server-side logic for the full application.
- **`frontend/`**: (Coming soon) Client-side application for the full application.

## 🚀 Key Features of the Prototype

* **Interactive Web UI:** Simple "Record" and "Stop" interface (powered by Gradio).
* **Local Inference:** Runs entirely on your machine (Mac MPS / CUDA), ensuring **patient data privacy**.
* **Auto-Logging:** Saves transcriptions to `transcripts.txt`.
* **Model:** Uses `Qwen/Qwen2-Audio-7B-Instruct`.

## 🎯 Project Goals

1.  **Phase 1 (Current):** Evaluate open-source multimodal models and establish a baseline for medical transcription.
2.  **Phase 2:** Fine-tune selected models on medical datasets.
3.  **Phase 3:** Develop a full-stack clinical application.

## 🛠️ Setup & Installation

### 1. Prerequisites
* **Python 3.10+**
* **FFmpeg**: `brew install ffmpeg` (Mac) or download from [ffmpeg.org](https://ffmpeg.org/) (Windows).
* **Hardware:** Mac M1/M2/M3 (16GB+ RAM) or NVIDIA GPU (16GB+ VRAM).

### 2. Quick Start

```bash
# Clone the repository
git clone https://github.com/jay-4502/doctor-prescription-poc.git
cd doctor-prescription-poc

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r frontend/requirements.txt -r backend/requirements.txt

# Set path from project's root directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the server
python -m backend.src.main

# Run Frontend
steamlit run frontend/app.py
```

For instructions on running the experimental models, please refer to the **[Experiments README](experiments/README.md)**.