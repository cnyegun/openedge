# OpenEdge

Open-source embedded ML deployment pipeline. Convert and deploy YOLO models to embedded devices without proprietary tools.

## Install

```bash
# Clone and install from GitHub
git clone https://github.com/cnyegun/openedge
cd openedge
pip install -e .

# For YOLO conversion (required for .pt models)
pip install -e ".[yolo]"

# For validation/testing (optional)
pip install -e ".[tf]"

# Install everything (recommended)
pip install -e ".[all]"
```

## Quick Start

```bash
# Full pipeline: convert → quantize → optimize → generate → build
openedge deploy model.pt --target esp32 --output firmware/ --calibration calibration_images/

# Or run individual steps
openedge convert model.pt --output output/
openedge optimize output/model.tflite
openedge generate output/model_optimized.tflite
openedge build output/model_data.cc --target esp32

# Validate on test images (requires tensorflow)
openedge validate output/model_optimized.tflite test_images/
```

## Commands

| Command | Description |
|---------|-------------|
| `deploy` | Full pipeline (convert → quantize → optimize → generate → build) |
| `convert` | Convert YOLO .pt/.pth to TFLite |
| `optimize` | Optimize TFLite model for embedded devices |
| `generate` | Generate C code from TFLite model |
| `build` | Build firmware for target platform |
| `validate` | Test model on image dataset |
| `version` | Show version |

## Supported Targets

- **ESP32** - esp32-s3-devkitc-1
- **STM32** - stm32f407vg
- **Arduino** - Arduino Nano 33 BLE (nRF52840 based)

## Example Output

```bash
$ openedge deploy yolov8n.pt --target esp32 --output firmware/ --calibration ./images
Deploying yolov8n.pt to esp32...
  Converted: output/model.tflite (12.9MB)
  Quantized: output/model_int8.tflite (3.4MB)
  Optimized: output/model_optimized.tflite
  Tensor arena: 8047019 bytes
  Generated: output/model_data.cc
  Built: output/firmware_esp32

Done! Output in firmware/
```

## Supported Models

Currently supports YOLOv8 models in PyTorch format:
- `.pt` files (standard PyTorch)
- `.pth` files (alternate extension)

## Hardware Requirements

**For 3.4MB INT8 YOLO model:**
- Flash: 4MB minimum (8MB recommended)
- RAM: 8MB for tensor arena during inference
- Inference time: ~30ms on ESP32-S3

## Known Limitations (v0.1.0)

1. **Validation command** uses hardcoded tensor indices optimized for YOLOv8n. May not work with other model architectures.

2. **Firmware template** includes basic TFLite ops (Conv2D, MaxPool, etc.). Complex models may require adding additional ops to `main.cc`.

3. **INT8 quantization** uses ultralytics' internal calibration. For custom calibration, export manually with `YOLO.export(data='custom.yaml')`.

## Performance Metrics

Tested on 100 COCO validation images:
- **Compression:** 3.8x (13MB → 3.4MB)
- **Speedup:** 2.9x faster (80ms → 28ms)
- **Accuracy:** 99.04% match with FP32 (0.96% relative error)
- **Flash:** Fits on ESP32 (3.4MB < 4MB limit)

## License

Apache 2.0

## Contributing

Contributions welcome! Please open an issue or PR on GitHub.