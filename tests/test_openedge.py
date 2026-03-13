import pytest
from pathlib import Path
import tempfile

from openedge.context import Context


class TestContext:
    def test_context_creation(self):
        ctx = Context(
            model_path=Path("model.pt"), target="esp32", output_dir=Path("output")
        )
        assert ctx.model_path == Path("model.pt")
        assert ctx.target == "esp32"
        assert ctx.output_dir == Path("output")

    def test_context_save_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = Context(
                model_path=Path("model.pt"), target="esp32", output_dir=Path(tmp)
            )
            ctx.tflite_path = Path(tmp) / "model.tflite"
            ctx.tensor_arena = 1000000
            ctx.save()

            loaded = Context.load(Path(tmp) / "context.json")
            assert loaded.target == "esp32"
            assert loaded.tensor_arena == 1000000


class TestConvert:
    def test_detect_format_pt(self):
        from openedge.convert.run import detect_format

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "model.pt"
            p.touch()
            assert detect_format(p) == "pytorch"

    def test_detect_format_onnx(self):
        from openedge.convert.run import detect_format

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "model.onnx"
            p.touch()
            assert detect_format(p) == "onnx"

    def test_detect_format_keras(self):
        from openedge.convert.run import detect_format

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "model.h5"
            p.touch()
            assert detect_format(p) == "keras"

    def test_detect_format_unsupported(self):
        from openedge.convert.run import detect_format

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "model.unknown"
            p.touch()
            with pytest.raises(ValueError):
                detect_format(p)

    def test_convert_run_requires_model_path(self):
        from openedge.convert.run import run

        ctx = Context(output_dir=Path("output"))
        with pytest.raises(ValueError):
            run(ctx)

    def test_convert_onnx_requires_tensorflow(self):
        from openedge.convert.run import _convert_onnx

        with tempfile.TemporaryDirectory() as tmp:
            onnx_file = Path(tmp) / "model.onnx"
            onnx_file.write_bytes(b"fake onnx")
            tflite_file = Path(tmp) / "model.tflite"
            with pytest.raises(RuntimeError):
                _convert_onnx(onnx_file, tflite_file)


class TestQuantize:
    def test_no_images_raises(self):
        from openedge.quantize.run import run

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tflite_file = tmp_path / "test.tflite"
            tflite_file.write_bytes(b"fake tflite")
            calib_dir = tmp_path / "calib"
            calib_dir.mkdir()
            ctx = Context(tflite_path=tflite_file, output_dir=tmp_path)
            with pytest.raises(ValueError):
                run(ctx, calib_dir)


class TestGenerate:
    def test_generate_cc_arrays(self):
        from openedge.generate.run import run

        with tempfile.TemporaryDirectory() as tmp:
            ctx = Context(
                optimized_path=Path(
                    "/home/smooth/cless/artifacts/models/embedded/yolov8n_int8.tflite"
                ),
                output_dir=Path(tmp),
                tensor_arena=500000,
            )
            paths = run(ctx)
            assert Path(paths["cc"]).exists()
            assert Path(paths["h"]).exists()


class TestOptimize:
    def test_optimize_uses_quantized_or_tflite(self):
        from openedge.optimize.run import run

        with tempfile.TemporaryDirectory() as tmp:
            ctx = Context(output_dir=Path(tmp))
            with pytest.raises(ValueError):
                run(ctx)


class TestBuild:
    def test_unsupported_target_raises(self):
        from openedge.build.run import run

        with tempfile.TemporaryDirectory() as tmp:
            model_file = Path(tmp) / "model.tflite"
            model_file.write_bytes(b"fake")
            with pytest.raises(ValueError):
                run(model_file, "unsupported_target", Path(tmp))

    def test_build_esp32(self):
        from openedge.build.run import run

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = run(Path("/tmp/test_model.tflite"), "esp32", tmp_path)
            result_path = Path(result)
            assert result_path.exists()
            assert (tmp_path / "model_data.cc").exists()
            assert (tmp_path / "model_data.h").exists()
            assert (tmp_path / "platformio.ini").exists()

    def test_build_stm32(self):
        from openedge.build.run import run

        with tempfile.TemporaryDirectory() as tmp:
            result = run(Path("/tmp/test_model.tflite"), "stm32", Path(tmp))
            assert Path(result).exists()


class TestValidate:
    def test_no_images_raises(self):
        from openedge.validate.run import run

        with tempfile.TemporaryDirectory() as tmp:
            model_file = Path(tmp) / "model.tflite"
            model_file.write_bytes(b"fake")
            dataset_dir = Path(tmp) / "dataset"
            dataset_dir.mkdir()
            with pytest.raises(ValueError):
                run(model_file, dataset_dir)


class TestUtils:
    def test_generate_c_arrays(self):
        from openedge.utils import generate_c_arrays

        with tempfile.TemporaryDirectory() as tmp:
            tflite_file = Path(tmp) / "model.tflite"
            tflite_file.write_bytes(b"fake tflite data")
            output_dir = Path(tmp) / "output"
            result = generate_c_arrays(tflite_file, output_dir)
            assert Path(result["cc"]).exists()
            assert Path(result["h"]).exists()

    def test_validate_input_file_exists(self):
        from openedge.utils import validate_input_file

        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "test.txt"
            f.write_text("content")
            validate_input_file(f, "test")
            assert True

    def test_validate_input_file_not_found(self):
        from openedge.utils import validate_input_file

        with pytest.raises(FileNotFoundError):
            validate_input_file(Path("nonexistent.txt"), "file")

    def test_validate_input_file_empty(self):
        from openedge.utils import validate_input_file

        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "empty.txt"
            f.touch()
            with pytest.raises(ValueError):
                validate_input_file(f, "file")

    def test_validate_directory_exists(self):
        from openedge.utils import validate_directory

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "dir"
            d.mkdir()
            validate_directory(d, "dir")
            assert True

    def test_validate_directory_not_found(self):
        from openedge.utils import validate_directory

        with pytest.raises(FileNotFoundError):
            validate_directory(Path("nonexistent"), "dir")

    def test_validate_directory_create(self):
        from openedge.utils import validate_directory

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "newdir"
            validate_directory(d, "dir", create=True)
            assert d.exists()

    def test_estimate_tensor_arena(self):
        from openedge.utils import estimate_tensor_arena

        result = estimate_tensor_arena(1000)
        assert result == 2000
        result_custom = estimate_tensor_arena(1000, 1.5)
        assert result_custom == 1500
