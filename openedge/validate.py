"""Validate TFLite models on image datasets."""

import time
from pathlib import Path

import numpy as np
import typer
from PIL import Image

from openedge.constants import IMG_SIZE, SUPPORTED_IMAGE_EXTS
from openedge.utils import check_file

# Optional dependency
try:
    import tensorflow as tf
except ImportError:
    tf = None


def validate_model(model_path: Path, dataset_path: Path, verbose: bool = False) -> dict:
    """Test model on dataset and return performance metrics.

    Args:
        model_path: Path to TFLite model
        dataset_path: Directory containing test images
        verbose: Print warnings for failed images

    Returns:
        dict with inference_success_rate, latency_ms, memory,
        successful_inferences, total_images
    """
    check_file(model_path, "TFLite model")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Collect test images
    images = []
    for ext in SUPPORTED_IMAGE_EXTS:
        images.extend(dataset_path.glob(f"*{ext}"))

    if not images:
        raise ValueError(f"No images found in {dataset_path}")

    if tf is None:
        raise RuntimeError(
            "TensorFlow required for validation. Install with: pip install tensorflow"
        )

    typer.echo(f"  Validating on {len(images)} images...")

    # Load model
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    latencies = []
    successful = 0

    for img_path in images:
        try:
            # Load and preprocess image
            img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            arr = np.expand_dims(np.array(img).astype(np.float32) / 255.0, 0)

            # Run inference
            start = time.perf_counter()
            interpreter.set_tensor(input_index, arr)
            interpreter.invoke()
            latency = (time.perf_counter() - start) * 1000

            latencies.append(latency)
            successful += 1

        except Exception as e:
            if verbose:
                typer.echo(f"  Warning: Failed to process {img_path}: {e}")

    success_rate = (successful / len(images)) * 100
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    typer.echo(f"  Success rate: {success_rate:.1f}%")
    typer.echo(f"  Avg latency: {avg_latency:.1f}ms")

    return {
        "inference_success_rate": success_rate,
        "latency_ms": avg_latency,
        "memory": 0,  # TODO: Implement memory tracking
        "successful_inferences": successful,
        "total_images": len(images),
    }
