"""
Configuration Management
==========================
Centralized configuration for the AI Doctor Prescription Assistant project.
Manages model paths, API endpoints, and environment settings across all modules.

Usage:
    from config import Config
    config = Config()
    model_path = config.get_model_path('whisper')
"""

import os
from pathlib import Path
from enum import Enum

# Project root (one level up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent


class ModelType(Enum):
    """Supported model types"""
    WHISPER_FINETUNED = "whisper-large-v3-turbo-finetuned"
    QWEN_AUDIO = "qwen2-audio"
    ULTRAVOX = "ultravox"


class Config:
    """
    Configuration singleton for the project.
    Handles all paths, model references, and environment settings.
    """

    # =====================================================================
    # DIRECTORIES
    # =====================================================================
    BASE_DIR = PROJECT_ROOT
    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data"
    BACKEND_DIR = BASE_DIR / "backend"
    FRONTEND_DIR = BASE_DIR / "frontend"
    EXPERIMENTS_DIR = BASE_DIR / "experiments"
    
    # Subdirectories
    DATA_TEMP_DIR = DATA_DIR / "temp"
    DATA_CACHE_DIR = DATA_DIR / "cache"
    LOGS_DIR = BASE_DIR / "logs"

    # =====================================================================
    # MODEL PATHS
    # =====================================================================
    # Local fine-tuned models (in models/ directory)
    MODEL_PATHS = {
        ModelType.WHISPER_FINETUNED: MODELS_DIR / "whisper-large-v3-turbo-finetuned",
    }

    # HuggingFace model IDs (for downloading from hub)
    HF_MODEL_IDS = {
        ModelType.QWEN_AUDIO: "Qwen/Qwen2-Audio-7B-Instruct",
        ModelType.ULTRAVOX: "fixie-ai/ultravox-v0_7-glm-4_6",
    }

    # =====================================================================
    # INFERENCE SETTINGS
    # =====================================================================
    # Audio processing
    SAMPLE_RATE = 16000  # Hz
    AUDIO_CHUNK_SIZE = 1.0  # seconds
    
    # Model inference
    MAX_NEW_TOKENS = 128
    TEMPERATURE = 0.6
    USE_FLOAT16 = True  # For memory efficiency on Mac/GPU

    # =====================================================================
    # API & SERVER SETTINGS
    # =====================================================================
    # Backend API
    API_HOST = os.getenv("API_HOST", "localhost")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_URL = f"http://{API_HOST}:{API_PORT}"
    
    # Frontend
    FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", 8501))

    # =====================================================================
    # LOGGING & DEBUG
    # =====================================================================
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = LOGS_DIR / "app.log"
    DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true"

    # =====================================================================
    # CLASS METHODS
    # =====================================================================

    @classmethod
    def get_model_path(cls, model_type: ModelType) -> Path:
        """
        Get the path to a model.
        
        Args:
            model_type: ModelType enum value
            
        Returns:
            Path object pointing to the model directory
            
        Raises:
            ValueError: If model type is not found or path doesn't exist
        """
        if model_type in cls.MODEL_PATHS:
            path = cls.MODEL_PATHS[model_type]
            if not path.exists():
                raise ValueError(
                    f"Model path does not exist: {path}\n"
                    f"Please ensure the model is downloaded to {path}"
                )
            return path
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @classmethod
    def get_hf_model_id(cls, model_type: ModelType) -> str:
        """
        Get the HuggingFace model ID for download.
        
        Args:
            model_type: ModelType enum value
            
        Returns:
            HuggingFace model ID string
            
        Raises:
            ValueError: If model type is not found
        """
        if model_type in cls.HF_MODEL_IDS:
            return cls.HF_MODEL_IDS[model_type]
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [cls.MODELS_DIR, cls.DATA_DIR, cls.DATA_TEMP_DIR, 
                         cls.DATA_CACHE_DIR, cls.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def print_config(cls) -> None:
        """Print current configuration (useful for debugging)."""
        print("\n" + "=" * 60)
        print("PROJECT CONFIGURATION")
        print("=" * 60)
        print(f"Base Directory:     {cls.BASE_DIR}")
        print(f"Models Directory:   {cls.MODELS_DIR}")
        print(f"Data Directory:     {cls.DATA_DIR}")
        print(f"API URL:            {cls.API_URL}")
        print(f"Sample Rate:        {cls.SAMPLE_RATE} Hz")
        print(f"Debug Mode:         {cls.DEBUG_MODE}")
        print("=" * 60 + "\n")


# Initialize directories on module load
Config.ensure_directories()

# For convenience: default config instance
config = Config()


if __name__ == "__main__":
    # Test configuration
    config.print_config()
    print("\nAvailable Models:")
    for model_type in ModelType:
        try:
            path = Config.get_model_path(model_type)
            print(f"  ✓ {model_type.value}: {path}")
        except ValueError as e:
            print(f"  ✗ {model_type.value}: {e}")
