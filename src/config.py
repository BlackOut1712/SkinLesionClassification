from pathlib import Path

#This file contains constants and paths used by other modules

#PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "chosen.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

#MODEL CONFIGURATION
IMAGE_SIZE = (224, 224)
NUM_CHANNELS = 3
NUM_CLASSES = 7

#INPUT VALIDATION
ALLOWED_IMAGE_FORMATS = ("JPEG", "PNG")
MAX_FILE_SIZE_MB = 15
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB*1024*1024
MAX_IMAGE_PIXELS = 8_294_400 