import torch
import librosa
import json
import re
import logging
from typing import Dict, Any, Optional
from backend.src.model_loader import get_model_components

# --- Configuration ---
logger = logging.getLogger(__name__)

# Moving prompt to a constant makes it easier to A/B test later
SYSTEM_PROMPT = (
    "You are a helpful medical scribe. Your task is to extract medication details from the doctor's audio. "
    "Output STRICTLY valid JSON. Do not add any conversational text before or after the JSON. "
    "The JSON format must be exactly: "
    "{\"medicines\": [{\"name\": \"...\", \"dosage\": \"...\", \"timing\": \"...\", \"frequency\": \"...\", \"days\": \"...\"}]}"
    "Rules:\n"
    "1. 'timing' must be one of: 'After Food', 'Before Food', 'With Food'.\n"
    "2. If a field is missing, use an empty string \"\".\n"
    "3. Do not invent information."
)

def clean_and_parse_json(raw_text: str) -> Dict[str, Any]:
    """
    Robustly extracts JSON from LLM output, handling Markdown code blocks 
    and common formatting errors.
    """
    data = {"medicines": [], "raw_text": raw_text}
    
    try:
        # 1. Remove Markdown code blocks (common LLM artifact)
        clean_text = re.sub(r"```json\s*", "", raw_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"```", "", clean_text)
        
        # 2. Find the first '{' and the last '}'
        start = clean_text.find('{')
        end = clean_text.rfind('}') + 1
        
        if start != -1 and end != 0:
            json_str = clean_text[start:end]
            parsed_data = json.loads(json_str)
            
            # Merge parsed data into our default structure
            if isinstance(parsed_data, dict):
                data.update(parsed_data)
            else:
                logger.warning("LLM returned JSON, but it wasn't a dictionary.")
        else:
            logger.warning("No JSON braces found in LLM response.")
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON Parsing failed: {e}. Raw text snippet: {raw_text[:50]}...")
    except Exception as e:
        logger.error(f"Unexpected parsing error: {e}")
        
    return data

def transcribe_audio(audio_file_path: str) -> str:
    """
    Pass 1: Loads audio, runs inference via an Audio Model (e.g., Qwen-Audio or Whisper), 
    and returns ONLY a raw, word-for-word text transcript.
    """
    # Assuming you have a getter for your audio model
    processor, audio_model = get_audio_model_components()
    logger.info(f"Pass 1 - Transcribing audio file: {audio_file_path}")

    try:
        audio, sr = librosa.load(audio_file_path, sr=16000)

        # Notice the prompt change: We only want the literal transcription here.
        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": "Transcribe the audio exactly as spoken."} 
            ]}
        ]

        text_input = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        inputs = processor(
            text=text_input, audio=audio, return_tensors="pt", sampling_rate=16000
        ).to(audio_model.device) 

        logger.info("Running transcription generation...")
        with torch.no_grad():
            generate_ids = audio_model.generate(
                **inputs, 
                max_new_tokens=512,  
                temperature=0.1,  # Keep temp very low for accurate transcription
                do_sample=True
            )

        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        transcript_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
        
        logger.debug(f"Transcript generated: {transcript_text}")
        return transcript_text

    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise RuntimeError(f"Audio transcription failed: {e}") from e


def structure_from_text(raw_text: str) -> Dict[str, Any]:
    """
    Pass 2: Takes a raw text transcript, passes it to a Text-Only LLM, 
    and returns a structured JSON dictionary.
    """
    # Assuming you have a getter for your text model (e.g., Llama 3, Qwen Text, GPT-4 API)
    tokenizer, text_model = get_text_model_components()
    logger.info("Pass 2 - Extracting structured data from transcript...")

    try:
        # Here is where your SYSTEM_PROMPT goes now.
        prompt = f"{SYSTEM_PROMPT}\n\nTranscript:\n{transcript_text}"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(text_model.device)

        logger.info("Running structuring generation...")
        with torch.no_grad():
            generate_ids = text_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2, # Low temp for strict JSON adherence
                do_sample=True
            )
            
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        structured_llm_response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        
        logger.debug(f"Raw structured LLM response: {structured_llm_response}")
        
        # Finally, parse the LLM's text output into an actual Python dictionary
        return clean_and_parse_json(structured_llm_response)
        
    except Exception as e:
        logger.error(f"Text structuring or JSON parsing failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "medicines": [],
            "raw_text": transcript_text if transcript_text else "Error during text structuring."
        }


def process_audio_pipeline(audio_file_path: str) -> Dict[str, Any]:
    """
    Orchestrates the two-pass pipeline, ensuring safe fallback for the frontend.
    """
    try:
        # Step 1
        transcript = transcribe_audio(audio_file_path)
        # Step 2
        structured_data = structure_transcript_to_json(transcript)
        
        # Optionally, inject the raw transcript into the final payload for frontend reference
        structured_data["raw_text"] = transcript
        return structured_data

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "medicines": [],
            "raw_text": "Fatal error during processing pipeline."
        }
