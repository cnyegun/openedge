import time
import numpy as np


def run(model_path, dataset_path, verbose=False):
    from pathlib import Path
    from openedge.constants import DEFAULT_IMAGE_SIZE
    from openedge.utils import validate_input_file, validate_directory

    validate_input_file(model_path, "TFLite model")
    validate_directory(dataset_path, "Dataset directory")

    images = list(Path(dataset_path).glob("*.jpg")) + list(
        Path(dataset_path).glob("*.png")
    )
    if not images:
        raise ValueError(f"No images found in {dataset_path}")

    try:
        import tensorflow as tf
    except ImportError:
        raise RuntimeError("tensorflow required: pip install tensorflow")

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    latencies = []
    successful_inferences = 0

    for img_path in images:
        try:
            import PIL.Image as PILImage

            img = PILImage.open(img_path).resize(
                (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
            )
            arr = np.array(img).astype(np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)

            start = time.perf_counter()
            interpreter.set_tensor(input_details[0]["index"], arr)
            interpreter.invoke()
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
            successful_inferences += 1
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to process {img_path}: {e}")
            continue

    accuracy = (successful_inferences / len(images)) * 100 if images else 0
    avg_latency = np.mean(latencies) if latencies else 0

    input_size = int(np.prod(input_details[0]["shape"]))
    output_size = int(np.prod(output_details[0]["shape"]))
    memory = (input_size + output_size) * 4

    return {
        "accuracy": accuracy,
        "latency_ms": avg_latency,
        "memory": memory,
        "successful_inferences": successful_inferences,
        "total_images": len(images),
    }
