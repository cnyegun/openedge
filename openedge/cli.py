import typer
from pathlib import Path
from typing import Optional

from openedge.convert.run import run as convert_run
from openedge.quantize.run import run as quantize_run
from openedge.optimize.run import run as optimize_run
from openedge.generate.run import run as generate_run
from openedge.build.run import run as build_run
from openedge.validate.run import run as validate_run
from openedge.context import Context

app = typer.Typer(help="OpenEdge - Embedded ML deployment")
apps = {
    "convert": typer.Typer(help="Convert model to TFLite"),
    "quantize": typer.Typer(help="Quantize to INT8"),
    "optimize": typer.Typer(help="Optimize for TFLite Micro"),
    "generate": typer.Typer(help="Generate C arrays"),
    "build": typer.Typer(help="Build firmware"),
    "validate": typer.Typer(help="Validate model"),
}

for name, sub in apps.items():
    app.add_typer(sub, name=name)


def _ctx(model=None, tflite=None, optimized=None, target="esp32", output=None):
    return Context(
        model_path=model,
        tflite_path=tflite,
        optimized_path=optimized,
        target=target,
        output_dir=output or Path("output"),
    )


@app.command()
def deploy(
    model: Path = typer.Option(..., help="Input model"),
    target: str = typer.Option("esp32"),
    output: Path = typer.Option(Path("output")),
    calibration: Optional[Path] = typer.Option(None),
):
    ctx = _ctx(model=model, target=target, output=output)
    typer.echo(f"Deploying {model} → {target}")

    ctx = convert_run(ctx)
    typer.echo(f"✓ {ctx.tflite_path}")

    if calibration:
        ctx = quantize_run(ctx, calibration)
        typer.echo(f"✓ {ctx.quantized_path}")

    ctx = optimize_run(ctx)
    typer.echo(f"✓ {ctx.optimized_path}")

    generate_run(ctx)
    typer.echo(f"✓ {output}/model_data.cc")


@apps["convert"].command("run")
def convert_cmd(model: Path, output: Path = Path("output"), target: str = "esp32"):
    ctx = _ctx(model=model, target=target, output=output)
    ctx = convert_run(ctx)
    typer.echo(f"✓ {ctx.tflite_path}")


@apps["quantize"].command("run")
def quantize_cmd(model: Path, calibration: Path, output: Path = Path("output")):
    ctx = _ctx(tflite=model, output=output)
    ctx = quantize_run(ctx, calibration)
    typer.echo(f"✓ {ctx.quantized_path}")


@apps["optimize"].command("run")
def optimize_cmd(model: Path, output: Path = Path("output")):
    ctx = _ctx(tflite=model, output=output)
    ctx = optimize_run(ctx)
    typer.echo(f"✓ {ctx.optimized_path} (arena: {ctx.tensor_arena})")


@apps["generate"].command("run")
def generate_cmd(model: Path, output: Path = Path("output")):
    ctx = _ctx(optimized=model, output=output)
    paths = generate_run(ctx)
    typer.echo(f"✓ {paths['cc']}")


@apps["build"].command("run")
def build_cmd(model: Path, target: str = "esp32", output: Path = Path("firmware")):
    result = build_run(model, target, output)
    typer.echo(f"✓ {result}")


@apps["validate"].command("run")
def validate_cmd(model: Path, dataset: Path, verbose: bool = False):
    result = validate_run(model, dataset, verbose=verbose)
    typer.echo(
        f"Accuracy: {result['accuracy']:.0f}% | Latency: {result['latency_ms']:.0f}ms"
    )


@app.command()
def version():
    typer.echo("OpenEdge v0.1.0")


def main():
    app()


if __name__ == "__main__":
    main()
