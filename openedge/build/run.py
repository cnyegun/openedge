import shutil
from pathlib import Path

from openedge.utils import generate_c_arrays, validate_input_file


TARGETS = {
    "esp32": {"board": "esp32-s3-devkitc-1"},
    "stm32": {"board": "stm32f407vg"},
    "arduino": {"board": "esp32-s3"},
}

SKETCH_TEMPLATE = """#include <Arduino.h>
#include "model_data.h"

extern "C" {{
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
}}

static tflite::MicroMutableOpResolver<10> resolver;
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

void setup() {{
    Serial.begin(115200);
    const tflite::Model* model = tflite::GetModel(model_data);
    resolver.AddAdd();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddSoftmax();
    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter.AllocateTensors();
    Serial.println("Model loaded. Ready for inference.");
}}

void loop() {{
    delay(1000);
    Serial.println("Waiting for inference...");
}}
"""

PLATFORMIO_TEMPLATE = """[env:{board}]
platform = espressif32
board = {board}
framework = arduino
monitor_speed = 115200
build_flags = -DTF_LITE_MICRO
"""


def run(model_path: Path, target: str, output_dir: Path, verbose: bool = False):
    if target not in TARGETS:
        raise ValueError(
            f"Unsupported target: {target}. Supported: {list(TARGETS.keys())}"
        )

    model_path = Path(model_path)
    validate_input_file(model_path, "Model file")

    if model_path.suffix == ".tflite":
        generate_c_arrays(model_path, output_dir)
    elif model_path.suffix in (".cc", ".c"):
        _copy_c_array(model_path, output_dir)
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")

    return _create_firmware(target, output_dir)


def _copy_c_array(model_path: Path, output_dir: Path):
    dest_cc = output_dir / f"{model_path.stem}.cc"
    if model_path.resolve() != dest_cc.resolve():
        shutil.copy(model_path, dest_cc)

    h_path = model_path.with_suffix(".h")
    if h_path.exists():
        dest_h = output_dir / h_path.name
        if h_path.resolve() != dest_h.resolve():
            shutil.copy(h_path, dest_h)


def _create_firmware(target: str, output_dir: Path):
    sketch_dir = output_dir / f"firmware_{target}"
    sketch_dir.mkdir(exist_ok=True)

    (sketch_dir / "main.cc").write_text(SKETCH_TEMPLATE)
    (output_dir / "platformio.ini").write_text(
        PLATFORMIO_TEMPLATE.format(board=TARGETS[target]["board"])
    )

    return str(sketch_dir)
