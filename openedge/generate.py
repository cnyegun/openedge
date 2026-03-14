"""C code generation from TFLite models."""

import os
import shutil
from pathlib import Path

import typer

from openedge.constants import (
    TENSOR_ARENA_MULTIPLIER,
    TENSOR_ARENA_SAFETY_MARGIN,
    MODEL_DATA_VAR,
    MODEL_SIZE_VAR,
    TENSOR_ARENA_VAR,
)
from openedge.utils import check_file, Context


def generate_c_arrays(tflite_path: Path, output_dir: Path, tensor_arena: int = None):
    """Convert TFLite model to C arrays for embedded compilation.

    Args:
        tflite_path: Path to TFLite model file
        output_dir: Directory to write C files
        tensor_arena: Optional custom tensor arena size (calculated if None)

    Returns:
        dict with paths to generated .cc and .h files
    """
    check_file(tflite_path, "TFLite model")
    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"  Generating C arrays from {tflite_path.name}...")

    # Read model data
    data = tflite_path.read_bytes()
    size = len(data)

    # Calculate tensor arena size if not provided
    if tensor_arena is None:
        tensor_arena = int(size * TENSOR_ARENA_MULTIPLIER * TENSOR_ARENA_SAFETY_MARGIN)

    # Generate .cc file with model data as byte array
    cc_path = output_dir / "model_data.cc"
    with cc_path.open("w", encoding="utf-8") as f:
        f.write(f'#include "model_data.h"\n')
        f.write(f"const unsigned char {MODEL_DATA_VAR}[] = {{\n")
        # Write in chunks of 16 bytes for readability
        for i in range(0, len(data), 16):
            chunk = data[i : i + 16]
            hex_values = ", ".join(f"0x{b:02x}" for b in chunk)
            f.write(f"    {hex_values},\n")
        f.write("};\n")

    # Generate .h file with size constants
    h_path = output_dir / "model_data.h"
    h_content = f"""#ifndef MODEL_DATA_H
#define MODEL_DATA_H
#define {MODEL_SIZE_VAR} {size}
#define {TENSOR_ARENA_VAR} {tensor_arena}
extern const unsigned char {MODEL_DATA_VAR}[];
#endif
"""
    h_path.write_text(h_content, encoding="utf-8")

    typer.echo(f"  Generated: {cc_path}")
    typer.echo(f"  Generated: {h_path}")

    return {"cc": str(cc_path), "h": str(h_path)}


def generate_code(ctx: Context):
    """Generate C code from optimized TFLite model.

    Args:
        ctx: Pipeline context with optimized_path set

    Returns:
        dict with paths to generated files
    """
    if not ctx.optimized_path:
        raise ValueError("optimized_path required in context")

    return generate_c_arrays(ctx.optimized_path, ctx.output_dir, ctx.tensor_arena)
