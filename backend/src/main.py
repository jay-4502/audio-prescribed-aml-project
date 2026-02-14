import logging
import shutil
import uuid
import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from backend.src.processing import transcribe_and_structure

# --- Configuration ---
# Define paths relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Points to 'project root' folder
TEMP_DIR = BASE_DIR / "data" / "temp"                     # Points to data/temp

# Ensure temp directory exists
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ScribeAPI")

# --- App Definition ---
app = FastAPI(
    title="AML Scribe API",
    description="Medical transcription service using Qwen-Audio",
    version="1.0.0"
)

# --- Middleware (Security) ---
# Allows your Streamlit app (or other frontends) to communicate safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with specific domain like "http://localhost:8501"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routes ---

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """
    K8s/Docker health probe endpoint.
    """
    return {"status": "healthy", "service": "scribe-api"}

@app.post("/scribe/process", status_code=status.HTTP_200_OK)
def process_audio(file: UploadFile = File(...)):
    """
    Receives an audio file, saves it securely, processes it via ML pipeline,
    and returns structured prescription data.
    """
    # 1. Generate a secure, unique filename
    file_extension = Path(file.filename).suffix or ".wav"
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    temp_file_path = TEMP_DIR / unique_filename
    
    logger.info(f"Received upload: {file.filename} -> saving as {unique_filename}")

    try:
        # 2. Save the file (Stream to disk to save RAM)
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 3. Run ML Pipeline
        # Note: We use a synchronous 'def' (not async) so FastAPI runs this 
        # in a threadpool, preventing the ML model from blocking the server.
        logger.info(f"Starting inference for {unique_filename}")
        result_json = transcribe_and_structure(str(temp_file_path))
        
        logger.info(f"Inference successful for {unique_filename}")
        return {"status": "success", "data": result_json}

    except ValueError as ve:
        # Catch specific errors (e.g., File format issues)
        logger.warning(f"Validation error: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
        
    except Exception as e:
        # Catch generic server errors
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="An error occurred during prescription generation."
        )
        
    finally:
        # 4. Cleanup (Guaranteed execution)
        if temp_file_path.exists():
            try:
                os.remove(temp_file_path)
                logger.debug(f"Cleaned up {unique_filename}")
            except OSError as e:
                logger.error(f"Error deleting temp file {unique_filename}: {e}")

if __name__ == "__main__":
    import uvicorn
    # Use standard host/port configuration
    uvicorn.run(app, host="0.0.0.0", port=8000)
