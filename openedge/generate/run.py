from openedge.utils import generate_c_arrays


def run(ctx):
    if not ctx.optimized_path:
        raise ValueError("optimized_path required")

    return generate_c_arrays(ctx.optimized_path, ctx.output_dir, ctx.tensor_arena)
