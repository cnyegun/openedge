"""Utility functions and shared classes for OpenEdge."""

import os
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from openedge.constants import DEFAULT_OUTPUT_DIR, CONTEXT_FILE


@dataclass
class Context:
    """Pipeline context for tracking state between steps."""

    model_path: Optional[Path] = None
    target: str = "esp32"
    output_dir: Path = DEFAULT_OUTPUT_DIR
    tflite_path: Optional[Path] = None
    quantized_path: Optional[Path] = None
    optimized_path: Optional[Path] = None
    tensor_arena: Optional[int] = None

    def save(self, path: Path = None):
        """Save context to JSON file for resuming pipeline."""
        path = path or self.output_dir / CONTEXT_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Context":
        """Load context from JSON file."""
        data = json.loads(path.read_text())
        # Convert path fields back to Path objects
        path_fields = {
            "model_path",
            "tflite_path",
            "quantized_path",
            "optimized_path",
            "output_dir",
        }
        return cls(
            **{k: Path(v) if v and k in path_fields else v for k, v in data.items()}
        )


def check_file(path: Path, name: str = "file"):
    """Validate that a file exists and is not empty."""
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{name} is empty: {path}")


def create_context(
    model_path: Path = None,
    tflite_path: Path = None,
    optimized_path: Path = None,
    target: str = "esp32",
    output_dir: Path = None,
) -> Context:
    """Helper to create Context with sensible defaults."""
    return Context(
        model_path=model_path,
        tflite_path=tflite_path,
        optimized_path=optimized_path,
        target=target,
        output_dir=output_dir or DEFAULT_OUTPUT_DIR,
    )
