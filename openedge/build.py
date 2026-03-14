"""Build firmware for target embedded platforms."""

import shutil
from pathlib import Path

import typer

from openedge.constants import (
    TARGETS,
    DEFAULT_BAUD_RATE,
    MAX_OPS_RESOLVER,
    TENSOR_ARENA_VAR,
)
from openedge.generate import generate_c_arrays
from openedge.utils import check_file

# Firmware template for Arduino/ESP32
FIRMWARE_TEMPLATE = f"""#include <Arduino.h>
#include "model_data.h"

extern "C" {{
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
}}

static tflite::MicroMutableOpResolver<{MAX_OPS_RESOLVER}> resolver;
static uint8_t tensor_arena[{TENSOR_ARENA_VAR}];

void setup() {{
    Serial.begin({DEFAULT_BAUD_RATE});
    const tflite::Model* model = tflite::GetModel(model_data);
    resolver.AddAdd();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMaxPool2D();
    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, {TENSOR_ARENA_VAR});
    interpreter.AllocateTensors();
    Serial.println("Model loaded. Ready for inference.");
}}

void loop() {{
    delay(1000);
    Serial.println("Waiting for inference...");
}}
"""

# PlatformIO configuration template
PLATFORMIO_TEMPLATE = """[env:{target}]
platform = espressif32
board = {board}
framework = arduino
monitor_speed = {baud}
build_flags = -DTF_LITE_MICRO
lib_deps = tensorflow/TensorFlowLite_ESP32@^1.0.0
"""


def build_firmware(model_path: Path, target: str, output_dir: Path) -> str:
    """Build firmware for target platform.

    Args:
        model_path: Path to TFLite model or C array file
        target: Target platform (esp32, stm32, arduino)
        output_dir: Directory for output files

    Returns:
        Path to firmware directory as string
    """
    if target not in TARGETS:
        raise ValueError(f"Unsupported target: {target}. Use: {list(TARGETS.keys())}")

    check_file(model_path, "Model file")
    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"  Building firmware for {target}...")

    # Generate C arrays if input is TFLite
    if model_path.suffix == ".tflite":
        generate_c_arrays(model_path, output_dir)
    else:
        # Copy existing C files
        dest = output_dir / f"{model_path.stem}.cc"
        if model_path.resolve() != dest.resolve():
            shutil.copy(model_path, dest)

    # Create firmware directory
    firmware_dir = output_dir / f"firmware_{target}"
    firmware_dir.mkdir(exist_ok=True)

    # Write firmware files
    (firmware_dir / "main.cc").write_text(FIRMWARE_TEMPLATE)

    board = TARGETS[target]
    platformio_content = PLATFORMIO_TEMPLATE.format(
        target=target, board=board, baud=DEFAULT_BAUD_RATE
    )
    (output_dir / "platformio.ini").write_text(platformio_content)

    typer.echo(f"  Built: {firmware_dir}")
    typer.echo(f"  Config: {output_dir}/platformio.ini")

    return str(firmware_dir)
