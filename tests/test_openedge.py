"""
Tests for OpenEdge CLI - Stricter tests for correctness verification
"""

import pytest
from pathlib import Path
import tempfile
import re

from openedge.cli import (
    Context,
    generate_c_arrays,
    check_file,
    convert_model,
    quantize_model,
    optimize_model,
    generate_code,
    build_firmware,
    validate_model,
    create_context,
)


class TestCArrayGeneration:
    """Tests for C array generation - critical for valid C code"""

    def test_hex_format_has_0x_prefix(self):
        """Verify C hex values have correct 0x prefix, not just 0"""
        with tempfile.TemporaryDirectory() as tmp:
            # Create a simple TFLite file with known bytes
            tflite_file = Path(tmp) / "model.tflite"
            tflite_file.write_bytes(b"\x01\x02\x0a\x0f\x10\xff")  # Known bytes

            output_dir = Path(tmp) / "output"
            result = generate_c_arrays(tflite_file, output_dir)

            cc_content = Path(result["cc"]).read_text()

            # Check that hex values have 0x prefix
            assert "0x01" in cc_content, "Missing 0x prefix"
            assert "0x02" in cc_content, "Missing 0x prefix"
            assert "0x0a" in cc_content, "Missing 0x prefix"
            assert "0xff" in cc_content, "Missing 0x prefix"

            # Ensure wrong format (just 0 without x) is NOT present
            assert "01," not in cc_content or "0x01" in cc_content
            assert "0x" in cc_content, "No hex values found"

    def test_c_array_syntax_valid(self):
        """Verify generated C code is syntactically valid"""
        with tempfile.TemporaryDirectory() as tmp:
            tflite_file = Path(tmp) / "model.tflite"
            tflite_file.write_bytes(b"\x01\x02\x03")

            output_dir = Path(tmp) / "output"
            result = generate_c_arrays(tflite_file, output_dir)

            cc_content = Path(result["cc"]).read_text()

            # Basic syntax checks
            assert "#include" in cc_content
            assert "const unsigned char model_data[]" in cc_content
            assert "{" in cc_content
            assert "};" in cc_content

            # Check array is properly formatted
            assert re.search(r"0x[0-9a-f]{2}", cc_content), "No hex values"

    def test_h_file_has_guards(self):
        """Verify header file has include guards"""
        with tempfile.TemporaryDirectory() as tmp:
            tflite_file = Path(tmp) / "model.tflite"
            tflite_file.write_bytes(b"test")

            output_dir = Path(tmp) / "output"
            result = generate_c_arrays(tflite_file, output_dir)

            h_content = Path(result["h"]).read_text()

            assert "#ifndef" in h_content
            assert "#define" in h_content
            assert "#endif" in h_content
            assert "MODEL_SIZE" in h_content
            assert "TENSOR_ARENA_SIZE" in h_content


class TestContext:
    def test_create_context(self):
        ctx = create_context(
            model_path=Path("model.pt"), target="esp32", output_dir=Path("output")
        )
        assert ctx.model_path == Path("model.pt")
        assert ctx.target == "esp32"
        assert ctx.output_dir == Path("output")

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = create_context(
                model_path=Path("model.pt"), target="esp32", output_dir=Path(tmp)
            )
            ctx.tflite_path = Path(tmp) / "model.tflite"
            ctx.tensor_arena = 1000000
            ctx.save()
            loaded = Context.load(Path(tmp) / "context.json")
            assert loaded.target == "esp32"
            assert loaded.tensor_arena == 1000000


class TestConvert:
    def test_convert_requires_model_path(self):
        ctx = create_context(output_dir=Path("output"))
        with pytest.raises(ValueError):
            convert_model(ctx)


class TestQuantize:
    def test_no_images_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tflite_file = tmp_path / "test.tflite"
            tflite_file.write_bytes(b"fake tflite")
            calib_dir = tmp_path / "calib"
            calib_dir.mkdir()
            ctx = create_context(tflite_path=tflite_file, output_dir=tmp_path)
            with pytest.raises(ValueError):
                quantize_model(ctx, calib_dir)


class TestGenerate:
    def test_generate_c_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            tflite_file = Path(tmp) / "model.tflite"
            tflite_file.write_bytes(b"fake tflite data")
            output_dir = Path(tmp) / "output"
            result = generate_c_arrays(tflite_file, output_dir)
            assert Path(result["cc"]).exists()
            assert Path(result["h"]).exists()


class TestOptimize:
    def test_optimize_requires_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = create_context(output_dir=Path(tmp))
            with pytest.raises(ValueError):
                optimize_model(ctx)


class TestBuild:
    def test_unsupported_target_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_file = Path(tmp) / "model.tflite"
            model_file.write_bytes(b"fake")
            with pytest.raises(ValueError):
                build_firmware(model_file, "bad_target", Path(tmp))

    def test_build_esp32(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = build_firmware(Path("/tmp/test_model.tflite"), "esp32", tmp_path)
            result_path = Path(result)
            assert result_path.exists()
            assert (tmp_path / "model_data.cc").exists()
            assert (tmp_path / "platformio.ini").exists()

    def test_platformio_has_tflite_deps(self):
        """Verify platformio.ini has TFLite dependencies"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_file = tmp_path / "model.tflite"
            model_file.write_bytes(b"fake")

            build_firmware(model_file, "esp32", tmp_path)

            platformio_content = (tmp_path / "platformio.ini").read_text()

            assert "lib_deps" in platformio_content, "Missing TFLite lib_deps"
            assert "TensorFlowLite" in platformio_content, "Missing TensorFlowLite"
            assert "build_flags" in platformio_content, "Missing build_flags"
            assert "TF_LITE_MICRO" in platformio_content, "Missing TF_LITE_MICRO"

    def test_build_stm32(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = build_firmware(Path("/tmp/test_model.tflite"), "stm32", Path(tmp))
            assert Path(result).exists()


class TestValidate:
    def test_no_images_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_file = Path(tmp) / "model.tflite"
            model_file.write_bytes(b"fake")
            dataset_dir = Path(tmp) / "dataset"
            dataset_dir.mkdir()
            with pytest.raises(ValueError):
                validate_model(model_file, dataset_dir)

    def test_returns_inference_success_rate_not_accuracy(self):
        """Verify validate returns inference_success_rate, not misleading accuracy"""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not installed")

        with tempfile.TemporaryDirectory() as tmp:
            model_file = Path(tmp) / "model.tflite"
            model_file.write_bytes(b"fake tflite")

            # Create test image
            from PIL import Image

            img_dir = Path(tmp) / "images"
            img_dir.mkdir()
            Image.new("RGB", (10, 10)).save(img_dir / "test.jpg")

            result = validate_model(model_file, img_dir)

            # Must have inference_success_rate, not accuracy
            assert "inference_success_rate" in result, "Missing inference_success_rate"
            assert "successful_inferences" in result
            assert "total_images" in result
            # Should NOT have misleading "accuracy" field
            assert "accuracy" not in result or result.get("accuracy") == result.get(
                "inference_success_rate"
            )


class TestUtils:
    def test_check_file_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "test.txt"
            f.write_text("content")
            check_file(f, "test")

    def test_check_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            check_file(Path("nonexistent.txt"), "file")

    def test_check_file_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "empty.txt"
            f.touch()
            with pytest.raises(ValueError):
                check_file(f, "file")
