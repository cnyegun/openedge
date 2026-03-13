from pathlib import Path
import numpy as np


def run(ctx, calibration_dir: Path, threshold: float = 2.0):
    from openedge.utils import validate_input_file, validate_directory

    if not ctx.tflite_path:
        raise ValueError("tflite_path required (run convert first)")

    validate_input_file(ctx.tflite_path, "TFLite model")
    validate_directory(calibration_dir, "Calibration directory")

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = ctx.output_dir / "model_int8.tflite"

    images = list(calibration_dir.glob("*.jpg")) + list(calibration_dir.glob("*.png"))
    if not images:
        raise ValueError(f"No images found in {calibration_dir}")

    representative_data = _create_representative_dataset(images)

    try:
        import tensorflow as tf
    except ImportError:
        raise RuntimeError("tensorflow required for quantization")

    converter = tf.lite.TFLiteConverter.from_flat_file(str(ctx.tflite_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    ctx.quantized_path = output_path
    ctx.save()

    return ctx


def _create_representative_dataset(images):
    def gen():
        for img_path in images:
            try:
                import PIL.Image as PILImage

                img = PILImage.open(img_path).resize((640, 640))
            except Exception:
                continue
            arr = np.array(img).astype(np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)
            yield [arr]

    return gen
