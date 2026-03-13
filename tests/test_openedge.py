import pytest
from pathlib import Path
import tempfile

from openedge.cli import C, g, chk, cnv, qn, opt, gen, bld, val, _c


class TestContext:
    def test_context_creation(self):
        ctx = C(m=Path("model.pt"), t="esp32", o=Path("output"))
        assert ctx.m == Path("model.pt")
        assert ctx.t == "esp32"
        assert ctx.o == Path("output")

    def test_context_save_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = C(m=Path("model.pt"), t="esp32", o=Path(tmp))
            ctx.tp = Path(tmp) / "model.tflite"
            ctx.ta = 1000000
            ctx.s()
            loaded = C.l(Path(tmp) / "c.json")
            assert loaded.t == "esp32"
            assert loaded.ta == 1000000


class TestConvert:
    def test_convert_run_requires_model_path(self):
        ctx = C(o=Path("output"))
        with pytest.raises(ValueError):
            cnv(ctx)


class TestQuantize:
    def test_no_images_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tflite_file = tmp_path / "test.tflite"
            tflite_file.write_bytes(b"fake tflite")
            calib_dir = tmp_path / "calib"
            calib_dir.mkdir()
            ctx = C(tp=tflite_file, o=tmp_path)
            with pytest.raises(ValueError):
                qn(ctx, calib_dir)


class TestGenerate:
    def test_gen_c(self):
        with tempfile.TemporaryDirectory() as tmp:
            tflite_file = Path(tmp) / "model.tflite"
            tflite_file.write_bytes(b"fake tflite data")
            output_dir = Path(tmp) / "output"
            result = g(tflite_file, output_dir)
            assert Path(result["cc"]).exists()
            assert Path(result["h"]).exists()


class TestOptimize:
    def test_optimize_requires_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = C(o=Path(tmp))
            with pytest.raises(ValueError):
                opt(ctx)


class TestBuild:
    def test_unsupported_target_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_file = Path(tmp) / "model.tflite"
            model_file.write_bytes(b"fake")
            with pytest.raises(ValueError):
                bld(model_file, "bad_target", Path(tmp))

    def test_build_esp32(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            result = bld(Path("/tmp/test_model.tflite"), "esp32", tmp_path)
            result_path = Path(result)
            assert result_path.exists()
            assert (tmp_path / "md.cc").exists()
            assert (tmp_path / "platformio.ini").exists()

    def test_build_stm32(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = bld(Path("/tmp/test_model.tflite"), "stm32", Path(tmp))
            assert Path(result).exists()


class TestValidate:
    def test_no_images_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_file = Path(tmp) / "model.tflite"
            model_file.write_bytes(b"fake")
            dataset_dir = Path(tmp) / "dataset"
            dataset_dir.mkdir()
            with pytest.raises(ValueError):
                val(model_file, dataset_dir)


class TestUtils:
    def test_chk_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "test.txt"
            f.write_text("content")
            chk(f, "test")

    def test_chk_not_found(self):
        with pytest.raises(FileNotFoundError):
            chk(Path("nonexistent.txt"), "file")

    def test_chk_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "empty.txt"
            f.touch()
            with pytest.raises(ValueError):
                chk(f, "file")
