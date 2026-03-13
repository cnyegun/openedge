# OpenEdge

Open-source embedded ML deployment pipeline. Convert and deploy YOLO models to embedded devices without proprietary tools.

## Install

```bash
pip install openedge

# For YOLO conversion (optional)
pip install ultralytics

# For quantization/validation (optional)
pip install tensorflow
```

## Quick Start

```bash
# Full pipeline: convert -> quantize -> optimize -> generate -> build
openedge deploy model.pt --target esp32 --output firmware/

# Or run individual steps
openedge convert model.pt --output output/
openedge quantize output/model.tflite --calibration images/
openedge optimize output/model_int8.tflite
openedge generate output/model_optimized.tflite
openedge build output/model_data.cc --target esp32

# Validate accuracy on test images
openedge validate output/model_optimized.tflite --dataset test_images/
```

## Commands

| Command | Description |
|---------|-------------|
| `deploy` | Full pipeline (convert → quantize → optimize → generate → build) |
| `convert` | Convert .pt/.onnx/.h5 to TFLite |
| `quantize` | Quantize to INT8 for smaller size |
| `optimize` | Optimize for TFLite Micro |
| `generate` | Generate C code from TFLite |
| `build` | Build firmware for target |
| `validate` | Test accuracy on dataset |
| `version` | Show version |

## Supported Targets

- ESP32
- STM32
- Arduino (ESP32-S3)

## Example Output

```
$ openedge deploy yolov8n.pt --target esp32 --output firmware/
Deploying yolov8n.pt to esp32...
  Converted: firmware/model_pytorch.tflite
  Optimized: firmware/model_optimized.tflite
  Tensor arena: 8047019 bytes
  Generated: firmware/model_data.cc

Done! Output in firmware/
```

## License

Apache 2.0