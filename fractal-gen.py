#!/usr/bin/env python3

import argparse
from mandelbrot import compute_mandelbrot
from julia import compute_julia


def parse_cli_ags():
    """
    Parse CLI arguments and return an object containing the values.
    """

    # Create main program parser
    parser = argparse.ArgumentParser(description="High-performance fractal generator")
    subparsers = parser.add_subparsers()
    subparsers.required = True
    subparsers.dest = "fractal"

    # Create mandelbrot subprogram parser
    parser_mandelbrot = subparsers.add_parser("mandelbrot", help="Generate a mandelbrot fractal")
    parser_mandelbrot.add_argument("--width", required=False, type=int, default="1920",
                                   help="Width of the output image in pixels", dest="width")

    parser_mandelbrot.add_argument("--height", required=False, type=int, default="1080",
                                   help="Height of the output image in pixels", dest="height")

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

    parser_mandelbrot.add_argument("--color-hue", required=False, type=int, default="204",
                                   help="Hue of the color used for the mandelbrot visualization", dest="color_hue")

    parser_mandelbrot.add_argument("--color-saturation", required=False, type=float, default="0.64",
                                   help="Saturation of the color used for the mandelbrot visualization",
                                   dest="color_saturation")

    parser_mandelbrot.add_argument("--color-intensity", required=False, type=float, default="2.0",
                                   help="Intensity of the color used for the mandelbrot visualization",
                                   dest="color_intensity")

    parser_mandelbrot.add_argument("--output-image", required=False, type=str, default="mandelbrot.png",
                                   help="Path of the output image file", dest="output_image_path")

    parser_mandelbrot.add_argument("--use-gpu", required=False, type=bool, default=True,
                                   help="Whether to use CUDA to compute the mandelbrot", dest="use_gpu")

    # Create julia subprogram parser
    parser_julia = subparsers.add_parser("julia", help="Generate a julia fractal")
    parser_julia.add_argument("--width", required=False, type=int, default="1920",
                              help="Width of the output image in pixels", dest="width")

    parser_julia.add_argument("--height", required=False, type=int, default="1080",
                              help="Height of the output image in pixels", dest="height")

    parser_julia.add_argument("--iterations", required=False, type=int, default="200",
                              help="Max iterations for orbital escape", dest="max_iterations")

    parser_julia.add_argument("--cx", required=False, type=float, default="-0.4",
                              help="CX value used for the iteration",
                              dest="cx")

    parser_julia.add_argument("--cy", required=False, type=float, default="0.6",
                              help="CY value used for the iteration",
                              dest="cy")

    parser_julia.add_argument("--color-hue", required=False, type=int, default="204",
                              help="Hue of the color used for the mandelbrot visualization", dest="color_hue")

    parser_julia.add_argument("--color-saturation", required=False, type=float, default="0.64",
                              help="Saturation of the color used for the mandelbrot visualization",
                              dest="color_saturation")

    parser_julia.add_argument("--color-intensity", required=False, type=float, default="6.0",
                              help="Intensity of the color used for the mandelbrot visualization",
                              dest="color_intensity")

    parser_julia.add_argument("--output-image", required=False, type=str, default="julia.png",
                              help="Path of the output image file", dest="output_image_path")

    parser_julia.add_argument("--use-gpu", required=False, type=bool, default=True,
                              help="Whether to use CUDA to compute the mandelbrot", dest="use_gpu")

    return parser.parse_args()


def main():
    args = parse_cli_ags()

    generate_fractal(args)


def generate_fractal(args):
    """
    Generate a fractal based on the arguments passed. The type of fractal is determined by the subprogram specified
    in the CLI.
    Args:
        args: CLI arguments
    """

    if args.fractal == "mandelbrot":
        # Generate mandelbrot fractal
        mandel_pixels, mandel_image = compute_mandelbrot(width=args.width, height=args.height,
                                                         max_iterations=args.max_iterations,
                                                         re_start=args.re_start, re_end=args.re_end,
                                                         im_start=args.im_start, im_end=args.im_end,
                                                         color_hue=args.color_hue,
                                                         color_saturation=args.color_saturation,
                                                         color_intensity=args.color_intensity, use_gpu=args.use_gpu)

        mandel_image.save(args.output_image_path)
    elif args.fractal == "julia":
        # Generate julia fractal
        julia_pixels, julia_image = compute_julia(width=args.width, height=args.height,
                                                  max_iterations=args.max_iterations,
                                                  cx=args.cx, cy=args.cy, color_hue=args.color_hue,
                                                  color_saturation=args.color_saturation,
                                                  color_intensity=args.color_intensity, use_gpu=args.use_gpu)

        julia_image.save(args.output_image_path)


if __name__ == '__main__':
    main()
