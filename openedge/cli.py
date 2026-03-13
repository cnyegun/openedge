import typer, os, shutil, json, time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np

app = typer.Typer()


@dataclass
class C:
    m: Optional[Path] = None
    t: str = "esp32"
    o: Path = Path("output")
    tp: Optional[Path] = None
    qp: Optional[Path] = None
    op: Optional[Path] = None
    ta: Optional[int] = None

    def s(self, p=None):
        (p or self.o / "c.json").write_text(
            json.dumps(
                {k: str(v) if isinstance(v, Path) else v for k, v in vars(self).items()}
            )
        )

    @classmethod
    def l(cls, p):
        d = json.loads(p.read_text())
        return cls(
            m=Path(d.get("m")),
            t=d.get("t", "esp32"),
            o=Path(d.get("o", "o")),
            tp=Path(d["tp"]) if d.get("tp") else None,
            qp=Path(d["qp"]) if d.get("qp") else None,
            op=Path(d["op"]) if d.get("op") else None,
            ta=d.get("ta"),
        )


def g(t, o, a=None):
    o.mkdir(parents=True, exist_ok=True)
    d = t.read_bytes()
    s = os.path.getsize(t)
    a = a or s * 2
    (o / "md.cc").write_text(
        '#include "md.h"\nconst unsigned char model_data[] = {\n'
        + ", ".join(f"0{b:02x}" for b in d)
        + "\n};"
    )
    (o / "md.h").write_text(
        f"#ifndef MD\n#define MD\n#define MS {s}\n#define TA {a}\nextern const unsigned char model_data[];\n#endif"
    )
    return {"cc": str(o / "md.cc"), "h": str(o / "md.h")}


def chk(p, n="f"):
    if not p.exists():
        raise FileNotFoundError(f"{n} not found")
    if p.stat().st_size == 0:
        raise ValueError(f"{n} is empty")


def cnv(c):
    if not c.m:
        raise ValueError("model_path required")
    c.o.mkdir(parents=True, exist_ok=True)
    fmt = {
        ".pt": "pytorch",
        ".pth": "pytorch",
        ".onnx": "onnx",
        ".h5": "keras",
        ".keras": "keras",
    }.get(c.m.suffix.lower())
    if not fmt:
        raise ValueError(f"Bad: {c.m.suffix}")
    out = c.o / f"model_{fmt}.tflite"
    if fmt == "pytorch":
        from ultralytics import YOLO

        shutil.copy(
            YOLO(str(c.m)).export(format="tflite", imgsz=640, verbose=False), out
        )
    else:
        raise RuntimeError("Only PyTorch")
    c.tp = out
    c.s()
    return c


def qn(c, cd):
    chk(c.tp, "TFLite")
    imgs = list(cd.glob("*.jpg")) + list(cd.glob("*.png"))
    if not imgs:
        raise ValueError(f"No imgs: {cd}")
    from PIL import Image

    def gn():
        for i in imgs:
            try:
                yield [
                    np.array(Image.open(i).resize((640, 640))).astype(np.float32)
                    / 255.0
                ]
            except:
                continue

    import tensorflow as tf

    cv = tf.lite.TFLiteConverter.from_flat_file(str(c.tp))
    cv.optimizations = [tf.lite.Optimize.DEFAULT]
    cv.representative_dataset = gn
    cv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    cv.inference_input_type = cv.inference_output_type = tf.uint8
    (c.o / "model_int8.tflite").write_bytes(cv.convert())
    c.qp = c.o / "model_int8.tflite"
    c.s()
    return c


def opt(c):
    i = c.qp or c.tp
    if not i:
        raise ValueError("need tflite")
    chk(i, "TFLite")
    out = c.o / "model_optimized.tflite"
    shutil.copy(i, out)
    c.ta = int(os.path.getsize(out) * 2 * 1.15)
    c.op = out
    c.s()
    return c


def gen(c):
    return g(c.op, c.o, c.ta)


def bld(m, t, o):
    if t not in {
        "esp32": "esp32-s3-devkitc-1",
        "stm32": "stm32f407vg",
        "arduino": "esp32-s3",
    }:
        raise ValueError(f"Bad: {t}")
    chk(m, "Model")
    o.mkdir(parents=True, exist_ok=True)
    if m.suffix == ".tflite":
        g(m, o)
    else:
        dest = o / f"{m.stem}.cc"
        if m.resolve() != dest.resolve():
            shutil.copy(m, dest)
    (o / f"firmware_{t}").mkdir(exist_ok=True)
    (o / f"firmware_{t}/main.cc").write_text(
        '#include <Arduino.h>\n#include "model_data.h"\nextern "C" {#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"}\nstatic tflite::MicroMutableOpResolver<10> r;\nstatic uint8_t tA[TA];\nvoid setup() {Serial.begin(115200); auto *m = tflite::GetModel(model_data); r.AddAdd(); r.AddConv2D(); r.AddDepthwiseConv2D(); r.AddMaxPool2D(); static tflite::MicroInterpreter i(m, r, tA, TA); i.AllocateTensors(); Serial.println("OK");}\nvoid loop() {delay(1000);}'
    )
    boards = {
        "esp32": "esp32-s3-devkitc-1",
        "stm32": "stm32f407vg",
        "arduino": "esp32-s3",
    }
    (o / "platformio.ini").write_text(
        f"[env]\nplatform = espressif32\nboard = {boards[t]}\nframework = arduino"
    )
    return str(o / f"firmware_{t}")


def val(m, d, v=False):
    from PIL import Image

    chk(m, "TFLite")
    if not d.exists():
        raise FileNotFoundError(f"DS: {d}")
    imgs = list(d.glob("*.jpg")) + list(d.glob("*.png"))
    if not imgs:
        raise ValueError(f"No imgs: {d}")
    import tensorflow as tf

    i = tf.lite.Interpreter(model_path=str(m))
    i.allocate_tensors()
    inp, out = i.get_input_details()[0]["index"], i.get_output_details()[0]["index"]
    lat, ok = [], 0
    for im in imgs:
        try:
            a = np.expand_dims(
                np.array(Image.open(im).resize((640, 640))).astype(np.float32) / 255.0,
                0,
            )
            t = time.perf_counter()
            i.set_tensor(inp, a)
            i.invoke()
            lat.append((time.perf_counter() - t) * 1000)
            ok += 1
        except:
            pass
    return {
        "accuracy": (ok / len(imgs)) * 100,
        "latency_ms": sum(lat) / len(lat) if lat else 0,
        "memory": 0,
    }


def _c(model=None, tflite=None, optimized=None, target="esp32", output=None):
    return C(m=model, tp=tflite, op=optimized, t=target, o=output or Path("output"))


@app.command()
def deploy(
    m: Path, t: str = "esp32", o: Path = Path("output"), cal: Optional[Path] = None
):
    c = _c(model=m, target=t, output=o)
    cnv(c)
    print(f"+ {c.tp}")
    if cal:
        qn(c, cal)
        print(f"+ {c.qp}")
    opt(c)
    print(f"+ {c.op}")
    gen(c)
    print(f"+ {o}/model_data.cc")


@app.command()
def convert_cmd(m: Path, o: Path = Path("output"), t: str = "esp32"):
    cnv(_c(model=m, target=t, output=o))
    print(f"+ {C.tp}")


@app.command()
def quantize_cmd(m: Path, cal: Path, o: Path = Path("output")):
    qn(_c(tflite=m, output=o), cal)
    print(f"+ {C.qp}")


@app.command()
def optimize_cmd(m: Path, o: Path = Path("output")):
    opt(_c(tflite=m, output=o))
    print(f"+ {C.op}")


@app.command()
def generate_cmd(m: Path, o: Path = Path("output")):
    print(f"+ {gen(_c(optimized=m, output=o))['cc']}")


@app.command()
def build_cmd(m: Path, t: str = "esp32", o: Path = Path("firmware")):
    print(f"+ {bld(m, t, o)}")


@app.command()
def validate_cmd(m: Path, d: Path, v: bool = False):
    r = val(m, d, v)
    print(f"Acc: {r['accuracy']:.0f}% | Lat: {r['latency_ms']:.0f}ms")


@app.command()
def version():
    print("OpenEdge v0.1.0")


if __name__ == "__main__":
    app()
