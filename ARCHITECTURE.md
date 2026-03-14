# OpenEdge Architecture

## Overview

OpenEdge is a modular pipeline for converting and deploying YOLO models to embedded devices. It transforms PyTorch models into optimized TFLite models and generates C code for embedded deployment.

## Pipeline Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Convert   │───▶│   Quantize  │───▶│   Optimize  │───▶│   Generate  │───▶│    Build    │
│             │    │             │    │             │    │             │    │             │
│  .pt ──▶    │    │   FP32 ──▶  │    │  Estimate   │    │  TFLite ──▶ │    │  C code ──▶ │
│  TFLite     │    │   INT8      │    │  memory     │    │  C arrays   │    │  firmware   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Step 1: Convert
**Module:** `openedge/convert.py`

Converts PyTorch YOLO models to TensorFlow Lite format.

**Input:**
- YOLO model file (.pt or .pth)

**Output:**
- TFLite model file (model.tflite) ~12-13MB

**Key Function:**
```python
def convert_model(ctx: Context) -> Context:
    # Uses ultralytics YOLO.export(format="tflite")
```

### Step 2: Quantize
**Module:** `openedge/quantize.py`

Quantizes TFLite model to INT8 format for smaller size and faster inference on embedded devices.

**Input:**
- TFLite model file
- Calibration images directory (for determining quantization ranges)

**Output:**
- INT8 quantized TFLite model (model_int8.tflite) ~3-4MB

**Key Function:**
```python
def quantize_model(ctx: Context, calibration_dir: Path) -> Context:
    # Re-exports with int8=True using ultralytics
```

### Step 3: Optimize
**Module:** `openedge/optimize.py`

Prepares model for TFLite Micro by calculating required tensor arena size.

**Input:**
- Quantized or regular TFLite model

**Output:**
- Optimized model (model_optimized.tflite)
- Tensor arena size calculation

**Memory Calculation:**
```python
tensor_arena = model_size * 2 * 1.15  # 2x size + 15% safety margin
```

### Step 4: Generate
**Module:** `openedge/generate.py`

Converts TFLite binary to C byte arrays for embedding in firmware.

**Input:**
- Optimized TFLite model

**Output:**
- `model_data.cc` - C array containing model bytes
- `model_data.h` - Header with MODEL_SIZE and TENSOR_ARENA_SIZE constants

**Format:**
```c
const unsigned char model_data[] = {
    0x1c, 0x00, 0x00, 0x00, ...
};
```

### Step 5: Build
**Module:** `openedge/build.py`

Generates firmware structure for target platform.

**Input:**
- C array files (model_data.cc/.h)
- Target platform (esp32, stm32, arduino)

**Output:**
- `firmware_{target}/main.cc` - Arduino sketch with TFLite Micro boilerplate
- `platformio.ini` - PlatformIO configuration

## Module Structure

```
openedge/
├── __init__.py          # Package version and exports
├── app.py               # Backward compatibility layer
├── cli.py               # CLI commands (deploy, convert, optimize, etc.)
├── constants.py         # All configuration constants
├── utils.py             # Context class and helper functions
├── convert.py           # YOLO → TFLite conversion
├── quantize.py          # TFLite → INT8 quantization
├── optimize.py          # TFLite Micro preparation
├── generate.py          # TFLite → C arrays
├── build.py             # Firmware generation
└── validate.py          # Model testing
```

## Constants

All configurable values are in `openedge/constants.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| IMG_SIZE | 640 | Model input resolution |
| MAX_CALIBRATION_IMAGES | 20 | Max images for quantization |
| MIN_CALIBRATION_IMAGES | 1 | Min images required |
| TENSOR_ARENA_MULTIPLIER | 2.0 | Arena size multiplier |
| TENSOR_ARENA_SAFETY_MARGIN | 1.15 | Additional 15% margin |
| DEFAULT_BAUD_RATE | 115200 | Serial monitor speed |

## Context Object

The `Context` dataclass (`openedge/utils.py`) maintains state between pipeline steps:

```python
@dataclass
class Context:
    model_path: Optional[Path]      # Original .pt file
    target: str                      # Target platform
    output_dir: Path                # Output directory
    tflite_path: Optional[Path]     # After convert
    quantized_path: Optional[Path]  # After quantize
    optimized_path: Optional[Path]  # After optimize
    tensor_arena: Optional[int]     # Memory requirement
```

Context is automatically saved to `context.json` after each step for resumable pipelines.

## Supported Targets

| Target | Board ID | Platform |
|--------|----------|----------|
| esp32 | esp32-s3-devkitc-1 | ESP32-S3 |
| stm32 | stm32f407vg | STM32F4 |
| arduino | arduino nano 33 | ESP32-S3 based |

## Dependencies

**Required:**
- typer >= 0.12.0
- numpy >= 1.24.0
- pillow >= 10.0.0

**Optional:**
- ultralytics >= 8.0.0 (for YOLO conversion)
- tensorflow >= 2.15.0 (for validation)

Install with:
```bash
pip install -e ".[all]"  # Everything
pip install -e ".[yolo]" # Just ultralytics
pip install -e ".[tf]"   # Just tensorflow
```

## Adding New Targets

To add support for a new platform:

1. Add target to `TARGETS` dict in `constants.py`:
```python
TARGETS = {
    "esp32": "esp32-s3-devkitc-1",
    "stm32": "stm32f407vg",
    "arduino": "arduino nano 33",
    "new_target": "board-id",  # Add here
}
```

2. Update `FIRMWARE_TEMPLATE` in `build.py` if platform needs special handling
3. Update `PLATFORMIO_TEMPLATE` if platform needs custom config
4. Add tests in `tests/test_openedge.py`

## Performance Characteristics

**YOLOv8n (INT8) on ESP32-S3:**
- Model size: 3.4MB
- Flash requirement: 4MB minimum
- RAM requirement: ~8MB for tensor arena
- Inference time: ~30ms per frame
- Input resolution: 640x640

**Pipeline Timing:**
- Convert: ~15-30 seconds
- Quantize: ~15-30 seconds (includes re-export)
- Optimize/Generate/Build: < 1 second each