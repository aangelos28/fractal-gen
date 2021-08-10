"""
Contains methods for the CLI
"""

from rich.console import Console
from rich.table import Table

console = Console()


def display_cli_args(fractal_type, args):
    """
    Display CLI arguments in the form of a table

    Args:
        args: CLI arguments from argparse
    """

    args_table = Table(title="Arguments")

    args_table.add_column("Argument", justify="left", no_wrap=True)
    args_table.add_column("Value", justify="right")

    args_table.add_row("Width", str(args.width))
    args_table.add_row("Height", str(args.height))
    args_table.add_row("Output Image", str(args.output_image_path))
    args_table.add_row("Color Hue", str(args.color_hue))
    args_table.add_row("Color Saturation", str(args.color_saturation))
    args_table.add_row("Color Intensity", str(args.color_intensity))

    if fractal_type in {"mandelbrot", "burning-ship"}:
        args_table.add_row("Iterations", str(args.max_iterations))
        args_table.add_row("Real Start", str(args.re_start))
        args_table.add_row("Real End", str(args.re_end))
        args_table.add_row("Imaginary Start", str(args.im_start))
        args_table.add_row("Imaginary End", str(args.im_end))
        args_table.add_row("Use GPU", str(args.use_gpu))
    elif fractal_type == "julia":
        args_table.add_row("Iterations", str(args.max_iterations))
        args_table.add_row("Real Start", str(args.re_start))
        args_table.add_row("Real End", str(args.re_end))
        args_table.add_row("Imaginary Start", str(args.im_start))
        args_table.add_row("Imaginary End", str(args.im_end))
        args_table.add_row("CX", str(args.cx))
        args_table.add_row("CY", str(args.cy))
        args_table.add_row("Use GPU", str(args.use_gpu))
    elif fractal_type == "buddhabrot":
        if args.use_gpu:
            args_table.add_row("Iterations", str(args.max_iterations))
            args_table.add_row("Real Start", str(args.re_start))
            args_table.add_row("Real End", str(args.re_end))
            args_table.add_row("Imaginary Start", str(args.im_start))
            args_table.add_row("Imaginary End", str(args.im_end))
            args_table.add_row("Samples per thread", str(args.samples_per_thread))
            args_table.add_row("Use GPU", "True")
        else:
            args_table.add_row("Iterations", str(args.max_iterations))
            args_table.add_row("Real Start", str(args.re_start))
            args_table.add_row("Real End", str(args.re_end))
            args_table.add_row("Imaginary Start", str(args.im_start))
            args_table.add_row("Imaginary End", str(args.im_end))
            args_table.add_row("Total samples", str(args.total_samples))
            args_table.add_row("Use GPU", "False")

    console.print(args_table)
    print()


def display_header():
    """
    Display program header (banner)
    """

    print("""______              _        _        _____            
|  ___|            | |      | |      |  __ \           
| |_ _ __ __ _  ___| |_ __ _| |______| |  \/ ___ _ __  
|  _| '__/ _` |/ __| __/ _` | |______| | __ / _ \ '_ \ 
| | | | | (_| | (__| || (_| | |      | |_\ \  __/ | | |
\_| |_|  \__,_|\___|\__\__,_|_|       \____/\___|_| |_|

High-performance Python fractal generator                                                                                                           
""")
