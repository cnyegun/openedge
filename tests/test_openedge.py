"""
Tests for OpenEdge CLI
"""

import pytest
from pathlib import Path
import tempfile

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
