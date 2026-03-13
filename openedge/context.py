from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json


@dataclass
class Context:
    model_path: Optional[Path] = None
    target: str = "esp32"
    output_dir: Path = field(default_factory=lambda: Path("output"))
    config: dict = field(default_factory=dict)

    model_format: Optional[str] = None
    tflite_path: Optional[Path] = None
    quantized_path: Optional[Path] = None
    optimized_path: Optional[Path] = None

    accuracy_before: Optional[float] = None
    accuracy_after: Optional[float] = None

    memory_required: Optional[int] = None
    tensor_arena: Optional[int] = None

    def save(self, path: Path = None):
        path = path or self.output_dir / "context.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_path": str(self.model_path) if self.model_path else None,
            "target": self.target,
            "output_dir": str(self.output_dir),
            "model_format": self.model_format,
            "tflite_path": str(self.tflite_path) if self.tflite_path else None,
            "quantized_path": str(self.quantized_path) if self.quantized_path else None,
            "optimized_path": str(self.optimized_path) if self.optimized_path else None,
            "accuracy_before": self.accuracy_before,
            "accuracy_after": self.accuracy_after,
            "memory_required": self.memory_required,
            "tensor_arena": self.tensor_arena,
            "config": self.config,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Context":
        data = json.loads(path.read_text())
        return cls(
            model_path=Path(data["model_path"]) if data.get("model_path") else None,
            target=data.get("target", "esp32"),
            output_dir=Path(data.get("output_dir", "output")),
            model_format=data.get("model_format"),
            config=data.get("config", {}),
            memory_required=data.get("memory_required"),
            accuracy_before=data.get("accuracy_before"),
            accuracy_after=data.get("accuracy_after"),
            tensor_arena=data.get("tensor_arena"),
            tflite_path=Path(data["tflite_path"]) if data.get("tflite_path") else None,
            quantized_path=Path(data["quantized_path"])
            if data.get("quantized_path")
            else None,
            optimized_path=Path(data["optimized_path"])
            if data.get("optimized_path")
            else None,
        )
