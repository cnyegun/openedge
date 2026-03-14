"""OpenEdge CLI - Command line interface for embedded ML deployment."""

from pathlib import Path

import typer

from openedge import __version__
from openedge.constants import DEFAULT_TARGET, DEFAULT_OUTPUT_DIR
from openedge.utils import create_context
from openedge.convert import convert_model
from openedge.quantize import quantize_model
from openedge.optimize import optimize_model
from openedge.generate import generate_code
from openedge.build import build_firmware
from openedge.validate import validate_model

app = typer.Typer(help="OpenEdge - Deploy ML models to embedded devices")


@app.command()
def deploy(
    model: Path = typer.Argument(..., help="YOLO model file (.pt/.pth)"),
    target: str = typer.Option(DEFAULT_TARGET, help="Target platform"),
    output: Path = typer.Option(DEFAULT_OUTPUT_DIR, help="Output directory"),
    calibration: Path = typer.Option(
        None, help="Calibration images folder (required for INT8 quantization)"
    ),
):
    """Full pipeline: convert → quantize → optimize → generate → build."""
    typer.echo(f"Deploying {model.name} to {target}...")

    ctx = create_context(model_path=model, target=target, output_dir=output)

    # Run pipeline
    ctx = convert_model(ctx)

    if calibration:
        ctx = quantize_model(ctx, calibration)

    ctx = optimize_model(ctx)
    generate_code(ctx)
    build_firmware(ctx.optimized_path, target, output)

    typer.echo(f"\nDone! Output in {output}/")


@app.command()
def convert(
    model: Path = typer.Argument(..., help="YOLO model file (.pt/.pth)"),
    output: Path = typer.Option(DEFAULT_OUTPUT_DIR, help="Output directory"),
):
    """Convert YOLO model to TFLite format."""
    ctx = create_context(model_path=model, output_dir=output)
    ctx = convert_model(ctx)
    typer.echo(f"Converted: {ctx.tflite_path}")


@app.command()
def optimize(
    model: Path = typer.Argument(..., help="Input TFLite model"),
    output: Path = typer.Option(DEFAULT_OUTPUT_DIR, help="Output directory"),
):
    """Optimize TFLite model for embedded devices."""
    ctx = create_context(tflite_path=model, output_dir=output)
    ctx = optimize_model(ctx)
    typer.echo(f"Optimized: {ctx.optimized_path}")


@app.command()
def generate(
    model: Path = typer.Argument(..., help="Input TFLite model"),
    output: Path = typer.Option(DEFAULT_OUTPUT_DIR, help="Output directory"),
):
    """Generate C code from TFLite model."""
    ctx = create_context(optimized_path=model, output_dir=output)
    result = generate_code(ctx)
    typer.echo(f"Generated: {result['cc']}")


@app.command()
def build(
    model: Path = typer.Argument(..., help="TFLite model or C array file"),
    target: str = typer.Option(DEFAULT_TARGET, help="Target platform"),
    output: Path = typer.Option(DEFAULT_OUTPUT_DIR, help="Output directory"),
):
    """Build firmware for target platform."""
    result = build_firmware(model, target, output)
    typer.echo(f"Built: {result}")


@app.command()
def validate(
    model: Path = typer.Argument(..., help="TFLite model to validate"),
    dataset: Path = typer.Argument(..., help="Test images directory"),
    verbose: bool = typer.Option(False, help="Show detailed output"),
):
    """Validate model accuracy on test dataset."""
    metrics = validate_model(model, dataset, verbose)
    typer.echo(f"Success rate: {metrics['inference_success_rate']:.1f}%")
    typer.echo(f"Avg latency: {metrics['latency_ms']:.1f}ms")


@app.command()
def version():
    """Show version information."""
    typer.echo(f"OpenEdge v{__version__}")
