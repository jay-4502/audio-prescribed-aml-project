# Project Structure Recommendations

## ✅ Changes Made

### 1. **Model Organization**
```
OLD:  whisper-turbo-finetuned-final/  (at root level)
NEW:  models/whisper-large-v3-turbo-finetuned/  (standardized naming)
```

**Benefits:**
- All models in a single `models/` directory
- Standardized naming convention: `{model_type}-{variant}-{status}`
- Easier to add multiple models (Qwen, Ultravox, etc.) in future
- Updated `.gitignore` to exclude all models from git

### 2. **Configuration Management**
Created `config.py` - Centralized configuration system:
- **Model paths**: Unified access to all model locations
- **Directory structure**: Single source of truth for all paths
- **API/Server settings**: Easily configurable via environment variables
- **Audio settings**: Centralized audio processing parameters

**Usage Example:**
```python
from config import Config, ModelType
model_path = Config.get_model_path(ModelType.WHISPER_FINETUNED)
```

### 3. **Code Updates**
- Updated `experiments/whisper_audio.py` to use `config.py` for model paths
- Made paths relative to project root (not hardcoded relative paths)

---

## 📁 Recommended Directory Structure

```
audio_prescribed-aml-project/
├── config.py                      # ← NEW: Centralized configuration
├── requirements.txt               # Root requirements
├── README.md                      
├── .gitignore                     # ← UPDATED
│
├── models/                        # ← NEW: Standardized models directory
│   ├── whisper-large-v3-turbo-finetuned/   # ← MOVED & RENAMED
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── model.safetensors
│   │   └── preprocessor_config.json
│   └── (future models here)
│
├── logs/                          # ← RECOMMENDED: Add for logging
│   ├── app.log
│   └── training.log
│
├── data/
│   ├── temp/                      # Temporary audio files
│   ├── cache/                     # ← NEW: Model cache, embeddings
│   └── (sample datasets)
│
├── backend/
│   ├── requirements.txt
│   └── src/
│       ├── __init__.py           # ← NEW: Package initialization
│       ├── main.py               # FastAPI app
│       ├── model_loader.py       # ← UPDATE: Use config.py
│       ├── processing.py
│       └── utils/                # ← NEW: Utility modules
│           ├── __init__.py
│           ├── audio_utils.py
│           └── logger.py
│
├── frontend/
│   ├── requirements.txt
│   └── app.py
│
├── experiments/
│   ├── README.md
│   ├── requirements.txt           # ← RECOMMENDED: Separate requirements
│   ├── config/                    # ← NEW: Experiment configs
│   │   └── model_configs.yaml
│   └── *.py (experiment scripts)
│
└── references/
    └── (archived notebooks, papers)
```

---

## 🔧 Specific Improvements Recommended

### 1. **Add `__init__.py` to Backend Package**
```python
# backend/src/__init__.py
"""
Backend package for AI Doctor Prescription Assistant.
"""

__version__ = "1.0.0"
```

**Why:** Makes backend a proper Python package; enables cleaner imports.

---

### 2. **Create Backend Utils Module**
```
backend/src/utils/
├── __init__.py
├── logger.py          # Centralized logging
├── audio_utils.py     # Audio processing helpers
└── validators.py      # Input validation
```

**Why:** Separates concerns; makes code more modular and testable.

---

### 3. **Add Logging System**
Create `backend/src/utils/logger.py`:
```python
import logging
from config import Config

def setup_logger(name):
    """Setup logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, Config.LOG_LEVEL))
    
    # File handler
    fh = logging.FileHandler(Config.LOG_FILE)
    # Console handler
    ch = logging.StreamHandler()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
```

**Why:** Centralized logging; better for debugging and monitoring.

---

### 4. **Environment Variables & .env File**
Create `.env.example`:
```bash
# API Configuration
API_HOST=localhost
API_PORT=8000
FRONTEND_PORT=8501

# Model Settings
MODEL_CACHE_DIR=./data/cache
HF_HOME=./data/cache/huggingface

# Debug & Logging
DEBUG=False
LOG_LEVEL=INFO

# Device Settings
DEVICE=mps  # or cuda, cpu
USE_FLOAT16=True
```

Add to `.gitignore`:
```
.env
.env.local
```

**Why:** Easy configuration without changing code; security (no secrets in code).

---

### 5. **Create Separate Experiment Requirements**
```
experiments/requirements.txt
```
Include only experiment-specific packages to avoid bloating backend/frontend.

**Current structure mixes all requirements at root. Better approach:**
- `backend/requirements.txt` - API only
- `frontend/requirements.txt` - UI only  
- `experiments/requirements.txt` - Experiments + dev tools
- `requirements.txt` - Meta-file listing the above

---

### 6. **Add Data Directory Structure**
```
data/
├── temp/              # Temporary audio files (cleared regularly)
├── cache/             # Model cache, embeddings
├── samples/           # Sample audio files for testing
└── outputs/           # Transcription results
```

Update `.gitignore`:
```
data/temp/
data/*.wav
data/*.mp3
data/cache/huggingface/
```

**Why:** Organized data flow; easier to manage and clean up.

---

### 7. **Add Version/Metadata Tracking**
Create `backend/src/version.py`:
```python
__version__ = "1.0.0-alpha"
MODEL_VERSION = {
    "whisper": "large-v3-turbo",
    "finetuned_on": "medicines dataset",
}
```

**Why:** Track model and app versions; important for reproducibility.

---

### 8. **Create Docker Support** (Optional)
Add `Dockerfile` and `docker-compose.yml`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV PYTHONPATH=/app
CMD ["uvicorn", "backend.src.main:app", "--host", "0.0.0.0"]
```

**Why:** Ensures consistent environment across machines.

---

## 📋 Implementation Priority

**High Priority (Do Now):**
1. ✅ Model directory organization → `models/`
2. ✅ Config.py for centralized settings
3. Add `__init__.py` to backend/src
4. Update model_loader.py to use config.py
5. Add .env.example

**Medium Priority (Do Soon):**
6. Create logging utility
7. Reorganize experiment files
8. Add data directory structure
9. Add version tracking

**Low Priority (Do Later):**
10. Docker setup
11. More sophisticated utils modules
12. CI/CD pipeline

---

## ✨ Code Updates Needed

### 1. Update `backend/src/model_loader.py`
```python
from config import Config, ModelType

class AudioModelLoader:
    def load_model(self):
        model_path = Config.get_model_path(ModelType.WHISPER_FINETUNED)
        # Use model_path instead of hardcoded path
```

### 2. Create `backend/src/__init__.py`
```python
"""Backend package for AI Doctor Prescription Assistant."""
__version__ = "1.0.0"
```

### 3. Update `backend/src/main.py`
```python
from backend.src.utils.logger import setup_logger
from config import Config

logger = setup_logger(__name__)
```

---

## 🎯 Summary

Your current structure is good! The recommendations above will make it:
- **Scalable**: Easy to add new models, experiments, features
- **Maintainable**: Clear separation of concerns
- **Reproducible**: Configuration management + version tracking
- **Professional**: Standard Python project structure
- **Production-ready**: Proper logging, error handling, environment config

**Start with the high-priority items**, then iterate. The `config.py` file I created is already in place—use it as your centralized configuration hub!
