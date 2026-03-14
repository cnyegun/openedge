"""
OpenEdge - Embedded ML deployment pipeline

Backward compatibility - imports from new modular structure.
"""

__version__ = "0.1.0"

# Re-export main CLI for backward compatibility
from openedge.cli import app

# Re-export key functions for programmatic use
from openedge.utils import Context, create_context, check_file
from openedge.convert import convert_model
from openedge.quantize import quantize_model
from openedge.optimize import optimize_model
from openedge.generate import generate_c_arrays, generate_code
from openedge.build import build_firmware
from openedge.validate import validate_model

__all__ = [
    "app",
    "Context",
    "create_context",
    "check_file",
    "convert_model",
    "quantize_model",
    "optimize_model",
    "generate_c_arrays",
    "generate_code",
    "build_firmware",
    "validate_model",
]
