import os
import logging
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from typing import Tuple, Any

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Allow overriding via environment variables (e.g., in Docker or .env)
MODEL_ID = os.getenv("SCRIBE_MODEL_ID", "Qwen/Qwen2-Audio-7B-Instruct")
# CACHE_DIR = os.getenv("HF_HOME", "./models/cache") 
CACHE_DIR = os.getenv("HF_HOME")  # If HF_HOME is not set, it will default to the transformers library's default cache directory

class AudioModelLoader:
    """
    Singleton class to manage the lifecycle of the heavy ML model.
    Prevents multiple loads and manages memory efficient loading.
    """
    _instance = None
    _model = None
    _processor = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AudioModelLoader, cls).__new__(cls)
        return cls._instance

    def load_model(self) -> Tuple[Any, Any]:
        """
        Loads the model and processor if not already loaded.
        Returns: (processor, model)
        """
        if self._model is not None and self._processor is not None:
            return self._processor, self._model

        logger.info(f"Initializing Model Loader for: {MODEL_ID}")
        
        # Determine Device
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Target Device: {device.upper()}")

        try:
            # 1. Load Processor
            logger.info("Loading Processor...")
            self._processor = AutoProcessor.from_pretrained(
                MODEL_ID, 
                trust_remote_code=True,
                cache_dir=CACHE_DIR
            )

            # 2. Load Model
            # Note: low_cpu_mem_usage=True is critical for loading 7B models on 16GB RAM machines
            logger.info("Loading Model Weights (this may take time)...")
            self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                cache_dir=CACHE_DIR
            )
            
            logger.info("Model Loaded Successfully!")
            
        except OSError as e:
            logger.critical(f"Failed to download/load model. Check internet or disk space. Error: {e}")
            raise RuntimeError("Model loading failed.") from e
        except torch.cuda.OutOfMemoryError:
            logger.critical("GPU Out of Memory! Try a smaller model or use CPU offloading.")
            raise
        except Exception as e:
            logger.critical(f"Unexpected error loading model: {e}")
            raise

        return self._processor, self._model

# --- Global Accessor ---
# This is the only function external modules should use.
def get_model_components() -> Tuple[Any, Any]:
    loader = AudioModelLoader()
    return loader.load_model()
