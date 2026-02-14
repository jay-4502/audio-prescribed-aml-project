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

def transcribe_and_structure(audio_file_path: str) -> Dict[str, Any]:
    """
    Loads audio, runs inference via Qwen-Audio, and returns structured data.
    """
    # 1. Get Model & Processor (Singleton ensures fast access)
    processor, model = get_model_components()
    
    logger.info(f"Processing audio file: {audio_file_path}")

    try:
        # 2. Load Audio (Librosa is robust but can be slow, 16k is standard for Qwen)
        audio, sr = librosa.load(audio_file_path, sr=16000)

        # 3. Prepare Inputs
        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": SYSTEM_PROMPT}
            ]}
        ]

        text_input = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        # safely move inputs to the same device as the model
        inputs = processor(
            text=text_input,
            audio=audio,
            return_tensors="pt",
            sampling_rate=16000
        ).to(model.device) 

        # 4. Run Inference
        logger.info("Running generation...")
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs, 
                max_new_tokens=512,  # Increased slightly to prevent JSON cutoff
                temperature=0.2,     # Lower temp = more deterministic/structured output
                do_sample=True
            )

        # 5. Decode Response
        # Slice off the input tokens so we only get the new generated text
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        response_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
        
        logger.debug(f"Raw LLM Response: {response_text}")

        # 6. Parse and Return
        return clean_and_parse_json(response_text)

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        # Return a safe fallback so the frontend doesn't crash
        return {
            "error": str(e),
            "medicines": [],
            "raw_text": "Error during processing."
        }