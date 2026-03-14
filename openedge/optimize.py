"""Optimize TFLite models for embedded deployment."""

import shutil
from pathlib import Path

import typer

from openedge.constants import (
    TENSOR_ARENA_MULTIPLIER,
    TENSOR_ARENA_SAFETY_MARGIN,
)
from openedge.utils import check_file, Context


def optimize_model(ctx: Context) -> Context:
    """Optimize model for TFLite Micro.

    Calculates tensor arena size and copies model to optimized location.

    Args:
        ctx: Pipeline context with quantized_path or tflite_path set

    Returns:
        Updated context with optimized_path and tensor_arena set
    """
    input_path = ctx.quantized_path or ctx.tflite_path
    if not input_path:
        raise ValueError("tflite_path or quantized_path required in context")

    check_file(input_path, "TFLite model")
    ctx.output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"  Optimizing {input_path.name} for TFLite Micro...")

    output_path = ctx.output_dir / "model_optimized.tflite"
    shutil.copy(input_path, output_path)

    # Calculate tensor arena size
    size = output_path.stat().st_size
    tensor_arena = int(size * TENSOR_ARENA_MULTIPLIER * TENSOR_ARENA_SAFETY_MARGIN)

    ctx.tensor_arena = tensor_arena
    ctx.optimized_path = output_path
    ctx.save()

    typer.echo(f"  Optimized: {output_path}")
    typer.echo(f"  Tensor arena: {tensor_arena:,} bytes")

    return ctx
