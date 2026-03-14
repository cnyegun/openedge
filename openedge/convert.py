"""Convert PyTorch YOLO models to TFLite format."""

import shutil
from pathlib import Path

import typer

from openedge.constants import SUPPORTED_PT_EXTS, IMG_SIZE
from openedge.utils import check_file, Context

# Optional dependency
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def convert_model(ctx: Context) -> Context:
    """Convert PyTorch YOLO model to TFLite format.

    Args:
        ctx: Pipeline context with model_path set

    Returns:
        Updated context with tflite_path set
    """
    if not ctx.model_path:
        raise ValueError("model_path required in context")

    ctx.output_dir.mkdir(parents=True, exist_ok=True)

    if ctx.model_path.suffix not in SUPPORTED_PT_EXTS:
        raise ValueError(
            f"Unsupported format: {ctx.model_path.suffix}. "
            f"Use {', '.join(SUPPORTED_PT_EXTS)}"
        )

    if YOLO is None:
        raise RuntimeError("Install ultralytics: pip install ultralytics")

    typer.echo(f"  Converting {ctx.model_path.name} to TFLite...")

    output_path = ctx.output_dir / "model.tflite"
    result_path = YOLO(str(ctx.model_path)).export(
        format="tflite", imgsz=IMG_SIZE, verbose=False
    )
    # result_path is str, convert to Path for clarity
    shutil.copy(Path(result_path), output_path)

    size_mb = output_path.stat().st_size / 1024 / 1024
    typer.echo(f"  Converted: {output_path} ({size_mb:.1f}MB)")

    ctx.tflite_path = output_path
    ctx.save()
    return ctx
