import os
from pathlib import Path

from openedge.constants import DEFAULT_TENSOR_ARENA_MULTIPLIER


def generate_c_arrays(
    tflite_path: Path, output_dir: Path, tensor_arena: int = None
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cc_path = output_dir / "model_data.cc"
    h_path = output_dir / "model_data.h"

    with open(tflite_path, "rb") as f:
        data = f.read()

    size = os.path.getsize(tflite_path)
    if tensor_arena is None:
        tensor_arena = int(size * DEFAULT_TENSOR_ARENA_MULTIPLIER)

    with open(cc_path, "w") as f:
        f.write('#include "model_data.h"\n')
        f.write("const unsigned char model_data[] = {\n")
        for i, b in enumerate(data):
            if i % 16 == 0:
                f.write("\n")
            f.write(f"0x{b:02x}, ")
        f.write("\n};\n")

    with open(h_path, "w") as f:
        f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n")
        f.write(f"#define MODEL_SIZE {size}\n")
        f.write(f"#define TENSOR_ARENA_SIZE {tensor_arena}\n")
        f.write("extern const unsigned char model_data[];\n")
        f.write("#endif\n")

    return {"cc": str(cc_path), "h": str(h_path)}


def validate_input_file(path: Path, name: str = "file"):
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{name} is not a file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{name} is empty: {path}")


def validate_directory(path: Path, name: str = "directory", create: bool = False):
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    if not path.is_dir():
        raise ValueError(f"{name} is not a directory: {path}")


def estimate_tensor_arena(
    model_size: int, multiplier: float = DEFAULT_TENSOR_ARENA_MULTIPLIER
) -> int:
    return int(model_size * multiplier)
