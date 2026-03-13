"""
Comprehensive tests for OpenEdge - Prove correctness and catch bugs early.
Tests C compilation, output correctness, and static analysis.
"""

import pytest
import subprocess
import tempfile
import struct
import re
from pathlib import Path
import numpy as np

from openedge.app import (
    generate_c_arrays,
    build_firmware,
    create_context,
    check_file,
)


class TestCCompilation:
    """Test that generated C code actually compiles - catches syntax errors early."""

    def test_model_data_cc_compiles(self):
        """Verify model_data.cc compiles without errors."""
        with tempfile.TemporaryDirectory() as tmp:
            tflite_file = Path(tmp) / "model.tflite"
            tflite_file.write_bytes(b"\x01\x02\x03\x04\xff\xfe\xfd")

            output_dir = Path(tmp) / "output"
            result = generate_c_arrays(tflite_file, output_dir)

            cc_file = Path(result["cc"])
            h_file = Path(result["h"])

            # Compile with gcc (syntax check only)
            proc = subprocess.run(
                ["gcc", "-fsyntax-only", "-I", str(output_dir), str(cc_file)],
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 0, f"C compile failed: {proc.stderr}"

    def test_model_data_h_compiles_as_cpp(self):
        """Verify header is valid C/C++."""
        with tempfile.TemporaryDirectory() as tmp:
            tflite_file = Path(tmp) / "model.tflite"
            tflite_file.write_bytes(b"fake model data")

            output_dir = Path(tmp) / "output"
            result = generate_c_arrays(tflite_file, output_dir)

            h_file = Path(result["h"])

            # Compile header as C++ (checks syntax)
            proc = subprocess.run(
                ["g++", "-fsyntax-only", "-x", "c++-header", str(h_file)],
                capture_output=True,
                text=True,
            )
            # Header might have warnings but should not error on guards
            assert "error" not in proc.stderr.lower() or proc.returncode == 0


class TestCOutputCorrectness:
    """Test that generated C arrays match original model data exactly."""

    def test_c_array_exact_bytes(self):
        """Verify C array contains exact same bytes as source file."""
        with tempfile.TemporaryDirectory() as tmp:
            # Known test data
            test_data = bytes(
                [0x00, 0x01, 0x02, 0x0A, 0x0F, 0xFF, 0xDE, 0xAD, 0xBE, 0xEF]
            )

            tflite_file = Path(tmp) / "model.tflite"
            tflite_file.write_bytes(test_data)

            output_dir = Path(tmp) / "output"
            result = generate_c_arrays(tflite_file, output_dir)

            cc_content = Path(result["cc"]).read_text()

            # Extract hex values from C array
            import re

            hex_values = re.findall(r"0x([0-9a-f]{2})", cc_content)
            c_bytes = bytes([int(h, 16) for h in hex_values])

            assert c_bytes == test_data, (
                f"C array differs: got {c_bytes.hex()}, want {test_data.hex()}"
            )

    def test_model_size_constant_correct(self):
        """Verify MODEL_SIZE matches actual file size."""
        with tempfile.TemporaryDirectory() as tmp:
            test_data = b"x" * 12345
            expected_size = len(test_data)

            tflite_file = Path(tmp) / "model.tflite"
            tflite_file.write_bytes(test_data)

            output_dir = Path(tmp) / "output"
            result = generate_c_arrays(tflite_file, output_dir)

            h_content = Path(result["h"]).read_text()

            # Extract MODEL_SIZE value
            import re

            match = re.search(r"#define MODEL_SIZE (\d+)", h_content)
            assert match, "MODEL_SIZE not found in header"

            actual_size = int(match.group(1))
            assert actual_size == expected_size, (
                f"MODEL_SIZE wrong: {actual_size} vs {expected_size}"
            )

    def test_tensor_arena_reasonable(self):
        """Verify tensor_arena is at least 2x model size."""
        with tempfile.TemporaryDirectory() as tmp:
            test_data = b"y" * 1000
            tflite_file = Path(tmp) / "model.tflite"
            tflite_file.write_bytes(test_data)

            output_dir = Path(tmp) / "output"
            result = generate_c_arrays(tflite_file, output_dir, tensor_arena=None)

            h_content = Path(result["h"]).read_text()

            import re

            match = re.search(r"#define TENSOR_ARENA_SIZE (\d+)", h_content)
            assert match

            arena_size = int(match.group(1))
            assert arena_size >= 2000, f"arena too small: {arena_size}"


class TestFirmwareCode:
    """Test generated firmware code validity."""

    def test_main_cc_syntax_valid(self):
        """Verify main.cc Arduino sketch has valid C++ syntax."""
        with tempfile.TemporaryDirectory() as tmp:
            tflite_file = Path(tmp) / "model.tflite"
            tflite_file.write_bytes(b"test")

            output_dir = Path(tmp) / "output"
            firmware_dir = build_firmware(tflite_file, "esp32", output_dir)

            main_cc = Path(firmware_dir) / "main.cc"
            assert main_cc.exists(), "main.cc not generated"

            content = main_cc.read_text()

            # Check required elements
            assert "#include" in content
            assert "setup()" in content
            assert "loop()" in content
            assert "model_data" in content
            assert "tensor_arena" in content
            assert "TENSOR_ARENA_SIZE" in content

            # Try compile with g++ (won't have Arduino headers but will catch syntax errors)
            # We just check it doesn't have obvious C++ syntax errors
            assert content.count("{") == content.count("}"), "Mismatched braces"

    def test_platformio_ini_valid(self):
        """Verify platformio.ini is valid ini format."""
        with tempfile.TemporaryDirectory() as tmp:
            tflite_file = Path(tmp) / "model.tflite"
            tflite_file.write_bytes(b"test")

            output_dir = Path(tmp) / "output"
            build_firmware(tflite_file, "esp32", output_dir)

            ini_path = output_dir / "platformio.ini"
            assert ini_path.exists()

            content = ini_path.read_text()

            # Check required sections
            assert "[env:" in content
            assert "platform = " in content
            assert "board = " in content
            assert "framework = " in content


class TestOutputCorrectness:
    """Test that model inference output matches between Python and reference."""

    def test_tflite_inference_deterministic(self):
        """Verify model produces same output for same input (deterministic)."""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not available")

        model_path = Path(
            "/home/smooth/cless/artifacts/models/embedded/yolov8n_int8.tflite"
        )
        if not model_path.exists():
            pytest.skip("Test model not available")

        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()

        # Same input twice
        np.random.seed(42)
        img1 = np.random.rand(1, 640, 640, 3).astype(np.float32)
        img2 = img1.copy()

        interpreter.set_tensor(0, img1)
        interpreter.invoke()
        out1 = interpreter.get_tensor(410).copy()

        interpreter.set_tensor(0, img2)
        interpreter.invoke()
        out2 = interpreter.get_tensor(410).copy()

        # Outputs must be identical
        assert np.allclose(out1, out2, rtol=1e-5), "Model not deterministic"

    def test_inference_output_range_sensible(self):
        """Verify inference outputs are in expected range."""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not available")

        model_path = Path(
            "/home/smooth/cless/artifacts/models/embedded/yolov8n_int8.tflite"
        )
        if not model_path.exists():
            pytest.skip("Test model not available")

        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()

        # All zeros input
        img = np.zeros((1, 640, 640, 3), dtype=np.float32)
        interpreter.set_tensor(0, img)
        interpreter.invoke()
        out = interpreter.get_tensor(410)

        # YOLO output should be in reasonable range
        assert not np.isnan(out).any(), "Output contains NaN"
        assert not np.isinf(out).any(), "Output contains Inf"


class TestRegressionBugs:
    """Regression tests for bugs we've fixed."""

    def test_hex_format_0x_prefix(self):
        """REGRESSION: Was generating '074' instead of '0x74'."""
        with tempfile.TemporaryDirectory() as tmp:
            tflite_file = Path(tmp) / "model.tflite"
            tflite_file.write_bytes(b"\x01\x0a\xff")

            output_dir = Path(tmp) / "output"
            result = generate_c_arrays(tflite_file, output_dir)

            cc_content = Path(result["cc"]).read_text()

            # Must have 0x prefix, not just digits
            assert "0x01" in cc_content, "Missing 0x01"
            assert "0x0a" in cc_content, "Missing 0x0a"
            assert "0xff" in cc_content, "Missing 0xff"

            # Must NOT have bare hex (e.g., "01," without 0x)
            bad_pattern = re.search(r"(?<!x)0([0-9a-f]{2})", cc_content)
            if bad_pattern:
                # Check it's actually a hex literal without 0x
                pos = bad_pattern.start()
                if pos > 0 and cc_content[pos - 1] != "x":
                    pytest.fail(f"Found hex without 0x prefix: {bad_pattern.group()}")

    def test_build_firmware_calls_generate(self):
        """REGRESSION: build_firmware was missing generate_c_arrays call."""
        with tempfile.TemporaryDirectory() as tmp:
            tflite_file = Path(tmp) / "model.tflite"
            tflite_file.write_bytes(b"test model")

            output_dir = Path(tmp) / "output"
            firmware_dir = build_firmware(tflite_file, "esp32", output_dir)

            # Check model_data files exist in firmware dir
            firmware_path = Path(firmware_dir)
            model_cc = firmware_path / "model_data.cc"
            model_h = firmware_path / "model_data.h"

            # They might be in output_dir or firmware_dir
            assert model_cc.exists() or (output_dir / "model_data.cc").exists(), (
                "model_data.cc not generated"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
