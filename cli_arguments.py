"""
Contains methods for the CLI
"""

import argparse
from rich.console import Console
from rich.table import Table

console = Console()


def parse_cli_args():
    """
    Parse CLI arguments and return an object containing the values.
    """

    # Create main program parser
    global_parser = argparse.ArgumentParser(description="High-performance Python fractal generator")
    subparsers = global_parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = "fractal"

    global_parser.add_argument("--width", required=False, type=int, default="1920",
                               help="Width of the output image in pixels", dest="width")

    global_parser.add_argument("--height", required=False, type=int, default="1080",
                               help="Height of the output image in pixels", dest="height")

    global_parser.add_argument("--color-hue", required=False, type=int, default="204",
                               help="Hue of the color used for the mandelbrot visualization", dest="color_hue")

    global_parser.add_argument("--color-saturation", required=False, type=float, default="0.64",
                               help="Saturation of the color used for the mandelbrot visualization",
                               dest="color_saturation")

    global_parser.add_argument("--color-intensity", required=False, type=float, default="2.0",
                               help="Intensity of the color used for the mandelbrot visualization",
                               dest="color_intensity")

    global_parser.add_argument("--output-image", required=False, type=str, default="fractal.png",
                               help="Path of the output image file", dest="output_image_path")

    # Create mandelbrot subprogram parser
    parser_mandelbrot = subparsers.add_parser("mandelbrot", help="Generate a mandelbrot fractal")

    parser_mandelbrot.add_argument("--iterations", required=False, type=int, default="80",
                                   help="Max iterations for orbital escape", dest="max_iterations")

    parser_mandelbrot.add_argument("--real-start", required=False, type=float, default="-2.0",
                                   help="Minimum value of the real complex plane", dest="re_start")

    parser_mandelbrot.add_argument("--real-end", required=False, type=float, default="1.0",
                                   help="Maximum value of the real complex plane", dest="re_end")

    parser_mandelbrot.add_argument("--imag-start", required=False, type=float, default="-1.0",
                                   help="Minimum value of the imaginary complex plane", dest="im_start")

    parser_mandelbrot.add_argument("--imag-end", required=False, type=float, default="1.0",
                                   help="Maximum value of the imaginary complex plane", dest="im_end")

    parser_mandelbrot.add_argument("--use-gpu", required=False, type=bool, default=True,
                                   help="Whether to use CUDA to compute the mandelbrot", dest="use_gpu")

    # Create burning ship subprogram parser
    parser_burning_ship = subparsers.add_parser("burning-ship", help="Generate a burning-ship fractal")
    parser_burning_ship.add_argument("--iterations", required=False, type=int, default="400",
                                     help="Max iterations for orbital escape", dest="max_iterations")

    parser_burning_ship.add_argument("--real-start", required=False, type=float, default="-2.0",
                                     help="Minimum value of the real complex plane", dest="re_start")

    parser_burning_ship.add_argument("--real-end", required=False, type=float, default="1.0",
                                     help="Maximum value of the real complex plane", dest="re_end")

    parser_burning_ship.add_argument("--imag-start", required=False, type=float, default="-2.0",
                                     help="Minimum value of the imaginary complex plane", dest="im_start")

    parser_burning_ship.add_argument("--imag-end", required=False, type=float, default="1.0",
                                     help="Maximum value of the imaginary complex plane", dest="im_end")

    parser_burning_ship.add_argument("--use-gpu", required=False, type=bool, default=True,
                                     help="Whether to use CUDA to compute the mandelbrot", dest="use_gpu")

    # Create julia subprogram parser
    parser_julia = subparsers.add_parser("julia", help="Generate a julia fractal")
    parser_julia.add_argument("--iterations", required=False, type=int, default="200",
                              help="Max iterations for orbital escape", dest="max_iterations")

    parser_julia.add_argument("--cx", required=False, type=float, default="-0.4",
                              help="CX value used for the iteration",
                              dest="cx")

    parser_julia.add_argument("--cy", required=False, type=float, default="0.6",
                              help="CY value used for the iteration",
                              dest="cy")

    parser_julia.add_argument("--use-gpu", required=False, type=bool, default=True,
                              help="Whether to use CUDA to compute the mandelbrot", dest="use_gpu")

    return global_parser.parse_args()


def display_cli_args(args):
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

    if args.fractal in {"mandelbrot", "burning-ship"}:
        args_table.add_row("Iterations", str(args.max_iterations))
        args_table.add_row("Real Start", str(args.re_start))
        args_table.add_row("Real End", str(args.re_end))
        args_table.add_row("Imaginary Start", str(args.im_start))
        args_table.add_row("Imaginary End", str(args.im_end))
        args_table.add_row("Use GPU", str(args.use_gpu))
    elif args.fractal == "julia":
        args_table.add_row("Iterations", str(args.max_iterations))
        args_table.add_row("CX", str(args.cx))
        args_table.add_row("CY", str(args.cy))
        args_table.add_row("Use GPU", str(args.use_gpu))

    console.print(args_table)


def display_header():
    """
    Display program header (banner)
    """

    print("""
______              _        _        _____            
|  ___|            | |      | |      |  __ \           
| |_ _ __ __ _  ___| |_ __ _| |______| |  \/ ___ _ __  
|  _| '__/ _` |/ __| __/ _` | |______| | __ / _ \ '_ \ 
| | | | | (_| | (__| || (_| | |      | |_\ \  __/ | | |
\_| |_|  \__,_|\___|\__\__,_|_|       \____/\___|_| |_|

High-performance Python fractal generator                                                                                                           
""")
