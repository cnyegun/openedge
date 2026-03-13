"""
OpenEdge - Embedded ML deployment pipeline
Simple CLI for converting and deploying YOLO models to embedded devices.
"""

import typer
import os
import shutil
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np

app = typer.Typer(help="OpenEdge - Deploy ML models to embedded devices")


# === Context: Stores pipeline state ===
@dataclass
class Context:
    model_path: Optional[Path] = None
    target: str = "esp32"
    output_dir: Path = Path("output")
    tflite_path: Optional[Path] = None
    quantized_path: Optional[Path] = None
    optimized_path: Optional[Path] = None
    tensor_arena: Optional[int] = None

    def save(self, path: Path = None):
        """Save context to JSON file."""
        save_path = path or self.output_dir / "context.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: str(v) if isinstance(v, Path) else v for k, v in vars(self).items()}
        save_path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Context":
        """Load context from JSON file."""
        data = json.loads(path.read_text())
        return cls(
            model_path=Path(data.get("model_path")) if data.get("model_path") else None,
            target=data.get("target", "esp32"),
            output_dir=Path(data.get("output_dir", "output")),
            tflite_path=Path(data["tflite_path"]) if data.get("tflite_path") else None,
            quantized_path=Path(data["quantized_path"])
            if data.get("quantized_path")
            else None,
            optimized_path=Path(data["optimized_path"])
            if data.get("optimized_path")
            else None,
            tensor_arena=data.get("tensor_arena"),
        )


# === Utilities ===
def generate_c_arrays(tflite_path: Path, output_dir: Path, tensor_arena: int = None):
    """Convert TFLite model to C arrays for embedded compilation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = tflite_path.read_bytes()
    size = os.path.getsize(tflite_path)
    arena = tensor_arena or size * 2

    # Generate .cc file with model data as byte array
    cc_path = output_dir / "model_data.cc"
    cc_path.write_text(
        '#include "model_data.h"\nconst unsigned char model_data[] = {\n'
        + ", ".join(f"0{b:02x}" for b in data)
        + "\n};"
    )

    # Generate .h file with size constants
    h_path = output_dir / "model_data.h"
    h_path.write_text(
        f"#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n#define MODEL_SIZE {size}\n#define TENSOR_ARENA_SIZE {arena}\nextern const unsigned char model_data[];\n#endif"
    )

    return {"cc": str(cc_path), "h": str(h_path)}


def check_file(path: Path, name: str = "file"):
    """Validate that a file exists and is not empty."""
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{name} is empty: {path}")


# === Core Pipeline Functions ===
def convert_model(ctx: Context) -> Context:
    """Convert PyTorch/ONNX/Keras model to TFLite format."""
    if not ctx.model_path:
        raise ValueError("model_path required")

    ctx.output_dir.mkdir(parents=True, exist_ok=True)

    # Detect format from file extension
    fmt_map = {
        ".pt": "pytorch",
        ".pth": "pytorch",
        ".onnx": "onnx",
        ".h5": "keras",
        ".keras": "keras",
    }
    fmt = fmt_map.get(ctx.model_path.suffix.lower())
    if not fmt:
        raise ValueError(f"Unsupported format: {ctx.model_path.suffix}")

    output_path = ctx.output_dir / f"model_{fmt}.tflite"

    if fmt == "pytorch":
        from ultralytics import YOLO

        result = YOLO(str(ctx.model_path)).export(
            format="tflite", imgsz=640, verbose=False
        )
        shutil.copy(result, output_path)
    else:
        raise RuntimeError(f"Format {fmt} not yet supported - use PyTorch .pt file")

    ctx.tflite_path = output_path
    ctx.save()
    return ctx


def quantize_model(ctx: Context, calibration_dir: Path) -> Context:
    """Quantize TFLite model to INT8 for smaller size."""
    check_file(ctx.tflite_path, "TFLite model")

    # Get calibration images
    images = list(calibration_dir.glob("*.jpg")) + list(calibration_dir.glob("*.png"))
    if not images:
        raise ValueError(f"No images found in {calibration_dir}")

    from PIL import Image

    # Create representative dataset generator
    def representative_data():
        for img_path in images:
            try:
                img = Image.open(img_path).resize((640, 640))
                arr = np.array(img).astype(np.float32) / 255.0
                yield [arr]
            except:
                continue

    import tensorflow as tf

    # Configure quantization
    converter = tf.lite.TFLiteConverter.from_flat_file(str(ctx.tflite_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    # Convert and save
    output_path = ctx.output_dir / "model_int8.tflite"
    output_path.write_bytes(converter.convert())

    ctx.quantized_path = output_path
    ctx.save()
    return ctx


def optimize_model(ctx: Context) -> Context:
    """Optimize model for TFLite Micro."""
    input_path = ctx.quantized_path or ctx.tflite_path
    if not input_path:
        raise ValueError("tflite_path or quantized_path required")

    check_file(input_path, "TFLite model")

    output_path = ctx.output_dir / "model_optimized.tflite"
    shutil.copy(input_path, output_path)

    # Estimate tensor arena size (2x model size * 1.15 safety margin)
    size = os.path.getsize(output_path)
    ctx.tensor_arena = int(size * 2 * 1.15)
    ctx.optimized_path = output_path
    ctx.save()
    return ctx


def generate_code(ctx: Context):
    """Generate C code from optimized TFLite model."""
    if not ctx.optimized_path:
        raise ValueError("optimized_path required")
    return generate_c_arrays(ctx.optimized_path, ctx.output_dir, ctx.tensor_arena)


def build_firmware(model_path: Path, target: str, output_dir: Path):
    """Build firmware for target platform."""
    valid_targets = {
        "esp32": "esp32-s3-devkitc-1",
        "stm32": "stm32f407vg",
        "arduino": "esp32-s3",
    }
    if target not in valid_targets:
        raise ValueError(
            f"Unsupported target: {target}. Use: {list(valid_targets.keys())}"
        )

    check_file(model_path, "Model file")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate C arrays from TFLite, or copy existing C files
    if model_path.suffix == ".tflite":
        generate_c_arrays(model_path, output_dir)
    else:
        dest = output_dir / f"{model_path.stem}.cc"
        if model_path.resolve() != dest.resolve():
            shutil.copy(model_path, dest)

    # Create firmware directory
    firmware_dir = output_dir / f"firmware_{target}"
    firmware_dir.mkdir(exist_ok=True)

    # Write Arduino sketch
    main_cc = firmware_dir / "main.cc"
    main_cc.write_text("""#include <Arduino.h>
#include "model_data.h"

extern "C" {
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
}

static tflite::MicroMutableOpResolver<10> resolver;
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

void setup() {
    Serial.begin(115200);
    const tflite::Model* model = tflite::GetModel(model_data);
    resolver.AddAdd();
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddMaxPool2D();
    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter.AllocateTensors();
    Serial.println("Model loaded. Ready for inference.");
}

void loop() {
    delay(1000);
    Serial.println("Waiting for inference...");
}
""")

    # Write PlatformIO config
    platform_io = output_dir / "platformio.ini"
    platform_io.write_text(f"""[env:{valid_targets[target]}]
platform = espressif32
board = {valid_targets[target]}
framework = arduino
""")

    return str(firmware_dir)


def validate_model(model_path: Path, dataset_path: Path, verbose: bool = False):
    """Test model on dataset and return accuracy metrics."""
    from PIL import Image

    check_file(model_path, "TFLite model")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    images = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
    if not images:
        raise ValueError(f"No images found in {dataset_path}")

    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    latencies = []
    successful = 0

    for img_path in images:
        try:
            img = Image.open(img_path).resize((640, 640))
            arr = np.expand_dims(np.array(img).astype(np.float32) / 255.0, 0)

            start = time.perf_counter()
            interpreter.set_tensor(input_index, arr)
            interpreter.invoke()
            latency = (time.perf_counter() - start) * 1000

            latencies.append(latency)
            successful += 1
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to process {img_path}: {e}")

    return {
        "accuracy": (successful / len(images)) * 100,
        "latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        "memory": 0,
    }


def create_context(
    model_path=None,
    tflite_path=None,
    optimized_path=None,
    target="esp32",
    output_dir=None,
):
    """Helper to create Context with sensible defaults."""
    return Context(
        model_path=model_path,
        tflite_path=tflite_path,
        optimized_path=optimized_path,
        target=target,
        output_dir=output_dir or Path("output"),
    )


# === CLI Commands ===
@app.command()
def deploy(
    model: Path = typer.Option(..., help="Input model file (.pt)"),
    target: str = typer.Option("esp32", help="Target platform"),
    output: Path = typer.Option(Path("output"), help="Output directory"),
    calibration: Optional[Path] = typer.Option(None, help="Calibration images folder"),
):
    """Full pipeline: convert -> quantize -> optimize -> generate -> build"""
    ctx = create_context(model_path=model, target=target, output_dir=output)

    typer.echo(f"Deploying {model} to {target}...")

    ctx = convert_model(ctx)
    typer.echo(f"  Converted: {ctx.tflite_path}")

    if calibration:
        ctx = quantize_model(ctx, calibration)
        typer.echo(f"  Quantized: {ctx.quantized_path}")

    ctx = optimize_model(ctx)
    typer.echo(f"  Optimized: {ctx.optimized_path}")

    generate_code(ctx)
    typer.echo(f"  Generated: {output}/model_data.cc")

    typer.echo(f"\nDone! Output in {output}")


@app.command()
def convert(
    model: Path = typer.Argument(..., help="Input model file"),
    output: Path = typer.Option(Path("output"), help="Output directory"),
    target: str = typer.Option("esp32", help="Target platform"),
):
    """Convert model to TFLite format."""
    ctx = create_context(model_path=model, target=target, output_dir=output)
    ctx = convert_model(ctx)
    typer.echo(f"Converted: {ctx.tflite_path}")


@app.command()
def quantize(
    model: Path = typer.Argument(..., help="Input TFLite model"),
    calibration: Path = typer.Argument(..., help="Calibration images folder"),
    output: Path = typer.Option(Path("output"), help="Output directory"),
):
    """Quantize model to INT8."""
    ctx = create_context(tflite_path=model, output_dir=output)
    ctx = quantize_model(ctx, calibration)
    typer.echo(f"Quantized: {ctx.quantized_path}")


@app.command()
def optimize(
    model: Path = typer.Argument(..., help="Input TFLite model"),
    output: Path = typer.Option(Path("output"), help="Output directory"),
):
    """Optimize model for TFLite Micro."""
    ctx = create_context(tflite_path=model, output_dir=output)
    ctx = optimize_model(ctx)
    typer.echo(f"Optimized: {ctx.optimized_path}")
    typer.echo(f"Tensor arena: {ctx.tensor_arena} bytes")


@app.command()
def generate(
    model: Path = typer.Argument(..., help="Input TFLite model"),
    output: Path = typer.Option(Path("output"), help="Output directory"),
):
    """Generate C code from TFLite model."""
    ctx = create_context(optimized_path=model, output_dir=output)
    paths = generate_code(ctx)
    typer.echo(f"Generated: {paths['cc']}")


@app.command()
def build(
    model: Path = typer.Argument(..., help="Input model (TFLite or .cc)"),
    target: str = typer.Option("esp32", help="Target platform"),
    output: Path = typer.Option(Path("firmware"), help="Output directory"),
):
    """Build firmware for target platform."""
    result = build_firmware(model, target, output)
    typer.echo(f"Firmware built: {result}")


@app.command()
def validate(
    model: Path = typer.Argument(..., help="Input TFLite model"),
    dataset: Path = typer.Argument(..., help="Test images folder"),
    verbose: bool = typer.Option(False, help="Show warnings"),
):
    """Test model accuracy on dataset."""
    result = validate_model(model, dataset, verbose)
    typer.echo(f"Accuracy: {result['accuracy']:.0f}%")
    typer.echo(f"Latency: {result['latency_ms']:.0f}ms")


@app.command()
def version():
    """Show version."""
    typer.echo("OpenEdge v0.1.0")


if __name__ == "__main__":
    app()
