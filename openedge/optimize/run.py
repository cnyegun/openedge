import shutil
import os

from openedge.constants import (
    DEFAULT_TENSOR_ARENA_MULTIPLIER,
    OPTIMIZE_TENSOR_ARENA_MULTIPLIER,
)
from openedge.utils import validate_input_file


def run(ctx):
    if not ctx.tflite_path and not ctx.quantized_path:
        raise ValueError("tflite_path or quantized_path required")

    input_path = ctx.quantized_path or ctx.tflite_path
    validate_input_file(input_path, "TFLite model")

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = ctx.output_dir / "model_optimized.tflite"

    shutil.copy(input_path, output_path)

    file_size = os.path.getsize(output_path)
    ctx.memory_required = int(file_size * DEFAULT_TENSOR_ARENA_MULTIPLIER)
    ctx.tensor_arena = int(ctx.memory_required * OPTIMIZE_TENSOR_ARENA_MULTIPLIER)

    ctx.optimized_path = output_path
    ctx.save()
    return ctx
