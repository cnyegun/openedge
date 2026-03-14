"""Quantize TFLite models to INT8 for smaller size."""

import glob
import shutil
from pathlib import Path

import typer

from openedge.constants import (
    IMG_SIZE,
    SUPPORTED_IMAGE_EXTS,
    MIN_CALIBRATION_IMAGES,
    MAX_CALIBRATION_IMAGES,
)
from openedge.utils import check_file, Context

# Optional dependency
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def quantize_model(ctx: Context, calibration_dir: Path) -> Context:
    """Quantize TFLite model to INT8 using ultralytics internal calibration.

    Args:
        ctx: Pipeline context with tflite_path and model_path set
        calibration_dir: Directory containing calibration images

    Returns:
        Updated context with quantized_path set
    """
    check_file(ctx.tflite_path, "TFLite model")

    # Validate calibration images exist
    images = []
    for ext in SUPPORTED_IMAGE_EXTS:
        images.extend(calibration_dir.glob(f"*{ext}"))

    if not images:
        raise ValueError(f"No calibration images found in {calibration_dir}")

    if len(images) < MIN_CALIBRATION_IMAGES:
        raise ValueError(
            f"Need at least {MIN_CALIBRATION_IMAGES} calibration images, "
            f"found {len(images)}"
        )

    if len(images) > MAX_CALIBRATION_IMAGES:
        typer.echo(
            f"  Using first {MAX_CALIBRATION_IMAGES} of {len(images)} "
            f"calibration images"
        )

    typer.echo(f"  Found {len(images)} calibration images")

    if YOLO is None:
        raise RuntimeError("Install ultralytics: pip install ultralytics")

    if not ctx.model_path or not ctx.model_path.exists():
        raise RuntimeError("Original model path required for INT8 quantization")

    typer.echo(f"  Re-exporting {ctx.model_path.name} with INT8 quantization...")

    model = YOLO(str(ctx.model_path))

    # Export with INT8 (ultralytics handles calibration internally)
    result = model.export(format="tflite", imgsz=IMG_SIZE, int8=True, verbose=False)
    result_path = Path(result)

    # Find the INT8 model in the saved_model directory
    saved_model_dir = result_path.parent
    int8_files = glob.glob(str(saved_model_dir / "*int8*.tflite"))
    output_path = ctx.output_dir / "model_int8.tflite"

    if int8_files:
        shutil.copy(int8_files[0], output_path)
        size_mb = output_path.stat().st_size / 1024 / 1024
        typer.echo(f"  Quantized: {output_path} ({size_mb:.1f}MB)")
    else:
        shutil.copy(result_path, output_path)
        typer.echo(f"  Quantized: {output_path}")

    ctx.quantized_path = output_path
    ctx.save()
    return ctx
