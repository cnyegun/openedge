# OpenEdge

Open-source embedded ML deployment pipeline. Convert, quantize, optimize, and deploy YOLO models to embedded devices.

## Install

```bash
pip install -e .
```

## Quick Start

```bash
# Deploy model to ESP32
openedge deploy --model model.pt --target esp32 --output firmware/

# Or run individual steps
openedge convert run --model model.pt
openedge quantize run --model output/model.tflite --calibration data/
openedge optimize run --model output/model_int8.tflite
openedge generate run --model output/model_optimized.tflite
openedge build run --model output/ --target esp32
```

## Features

- **Convert**: PyTorch → ONNX → TFLite
- **Quantize**: INT8 with accuracy validation
- **Optimize**: TFLite Micro transforms + memory calculation
- **Generate**: C arrays for embedded compilation
- **Build**: Firmware for ESP32 (more targets coming)
- **Validate**: Test accuracy before hardware deployment

## Supported Targets

- ESP32
- STM32
- Arduino (ESP32-S3 compatible)

## License

Apache 2.0