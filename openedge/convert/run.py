from pathlib import Path

from openedge.constants import DEFAULT_IMAGE_SIZE

FORMAT_MAP = {
    ".pt": "pytorch",
    ".pth": "pytorch",
    ".onnx": "onnx",
    ".h5": "keras",
    ".keras": "keras",
}


def detect_format(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    fmt = FORMAT_MAP.get(path.suffix.lower())
    if not fmt:
        raise ValueError(f"Unsupported format: {path.suffix}")
    return fmt


def run(ctx):
    if not ctx.model_path:
        raise ValueError("model_path required")

    fmt = detect_format(ctx.model_path)
    ctx.model_format = fmt

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = ctx.output_dir / f"model_{fmt}.tflite"

    if fmt == "pytorch":
        _convert_pytorch(ctx.model_path, output_path)
    elif fmt == "onnx":
        _convert_onnx(ctx.model_path, output_path)
    elif fmt == "keras":
        _convert_keras(ctx.model_path, output_path)

    ctx.tflite_path = output_path
    ctx.save()
    return ctx


def _convert_pytorch(input_path: Path, output_path: Path):
    try:
        from ultralytics import YOLO

        model = YOLO(str(input_path))
        result = model.export(format="tflite", imgsz=DEFAULT_IMAGE_SIZE, verbose=False)
        import shutil

        shutil.copy(result, output_path)
    except Exception as e:
        if "ultralytics" in str(e).lower() or "tensorflow" in str(e).lower():
            raise RuntimeError(f"Missing dependency: {e}")
        raise


def _convert_onnx(input_path: Path, output_path: Path):
    raise RuntimeError("ONNX conversion not yet implemented - use PyTorch model")


def _convert_keras(input_path: Path, output_path: Path):
    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(str(input_path))
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(output_path, "wb") as f:
            f.write(tflite_model)
    except ImportError:
        raise RuntimeError("tensorflow required: pip install tensorflow")
