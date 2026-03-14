"""
OpenEdge - Configuration constants

All magic numbers and configuration values in one place.
Change these to customize the pipeline behavior.
"""

from pathlib import Path

# === Model Configuration ===
IMG_SIZE = 640  # Input image size for YOLO models (width, height)
SUPPORTED_IMAGE_EXTS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
]  # Valid calibration image formats
MAX_CALIBRATION_IMAGES = 20  # Maximum calibration images to use for INT8 quantization
MIN_CALIBRATION_IMAGES = 1  # Minimum required calibration images

# === Hardware Configuration ===
DEFAULT_BAUD_RATE = 115200  # Serial monitor speed for embedded devices
DEFAULT_TARGET = "esp32"  # Default deployment target

# === TFLite Micro Configuration ===
TENSOR_ARENA_MULTIPLIER = 2.0  # Model size multiplier for tensor arena
TENSOR_ARENA_SAFETY_MARGIN = 1.15  # Additional 15% safety margin
MAX_OPS_RESOLVER = 10  # Maximum ops supported by MicroMutableOpResolver

# === Output Configuration ===
DEFAULT_OUTPUT_DIR = Path("output")  # Default output directory
CONTEXT_FILE = "context.json"  # Pipeline state file name

# === Supported Targets ===
TARGETS = {
    "esp32": "esp32-s3-devkitc-1",
    "stm32": "stm32f407vg",
    "arduino": "arduino nano 33",
}

# === Model Formats ===
SUPPORTED_PT_EXTS = [".pt", ".pth"]  # PyTorch model extensions
SUPPORTED_TFLITE_EXTS = [".tflite"]  # TFLite model extensions
SUPPORTED_MODEL_EXTS = (
    SUPPORTED_PT_EXTS + SUPPORTED_TFLITE_EXTS
)  # All supported model formats

# === C Code Generation ===
MODEL_DATA_VAR = "model_data"  # C variable name for model data
MODEL_SIZE_VAR = "MODEL_SIZE"  # C macro for model size
TENSOR_ARENA_VAR = "TENSOR_ARENA_SIZE"  # C macro for tensor arena size
