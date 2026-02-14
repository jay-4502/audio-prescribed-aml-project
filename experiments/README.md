# Experimental Audio Models

This directory contains experimental scripts for evaluating and interacting with various audio models, specifically focusing on the `Qwen2-Audio-7B-Instruct` model and `Ultravox`. These scripts were used during the initial feasibility study and model selection phase.

## Scripts Overview

### `qwen_audio.py`
A script to run inference on audio files using the `Qwen2-Audio-7B-Instruct` model. This is useful for batch processing or testing the model on pre-recorded audio samples.

### `qwen_audio_live.py`
This script launches a local Gradio web interface that allows you to:
- Record audio directly from your microphone.
- Transcribe the audio using the `Qwen2-Audio-7B-Instruct` model.
- Automatically save transcriptions to `transcripts.txt`.

### `ultavox.py`
An experimental script for testing the **Ultravox** model, another multimodal model evaluated during the project's initial phase.

## Usage

Ensure you have the project dependencies installed (see root [README.md](../README.md)).

Navigate to the project root directory and run the scripts as modules or directly:

```bash
# Activate your virtual environment first
source venv/bin/activate  # Mac/Linux
# .\venv\Scripts\activate  # Windows

# Run the live Gradio interface (Recommended for testing)
python experiments/qwen_audio_live.py

# Run batch inference on files
python experiments/qwen_audio.py

# Run Ultravox experiment
python experiments/ultavox.py
```

## Requirements
These scripts rely on the project-wide `requirements.txt`. Ensure you have satisfied the hardware requirements (GPU/MPS) detailed in the main documentation.
