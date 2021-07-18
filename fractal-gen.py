#!/usr/bin/env python3

from cli_arguments import *
from mandelbrot import generate_mandelbrot
from julia import generate_julia
from burning_ship import generate_burning_ship

console = Console()


def generate_fractal(args):
    """
    Generate a fractal based on the arguments passed. The type of fractal is determined by the subprogram specified
    in the CLI.
    Args:
        args: CLI arguments
    """

    if args.fractal == "mandelbrot":
        # Generate mandelbrot fractal
        console.print("Generating Mandelbrot fractal...", style="yellow")
        return generate_mandelbrot(width=args.width, height=args.height,
                                   max_iterations=args.max_iterations,
                                   re_start=args.re_start, re_end=args.re_end,
                                   im_start=args.im_start, im_end=args.im_end,
                                   color_hue=args.color_hue,
                                   color_saturation=args.color_saturation,
                                   color_intensity=args.color_intensity, use_gpu=args.use_gpu)
    elif args.fractal == "burning-ship":
        console.print("Generating Burning Ship fractal...", style="yellow")
        return generate_burning_ship(width=args.width, height=args.height,
                                     max_iterations=args.max_iterations,
                                     re_start=args.re_start, re_end=args.re_end,
                                     im_start=args.im_start, im_end=args.im_end,
                                     color_hue=args.color_hue,
                                     color_saturation=args.color_saturation,
                                     color_intensity=args.color_intensity, use_gpu=args.use_gpu)
    elif args.fractal == "julia":
        # Generate julia fractal
        console.print("Generating Julia fractal...", style="yellow")
        return generate_julia(width=args.width, height=args.height,
                              max_iterations=args.max_iterations,
                              cx=args.cx, cy=args.cy, color_hue=args.color_hue,
                              color_saturation=args.color_saturation,
                              color_intensity=args.color_intensity, use_gpu=args.use_gpu)


def main():
    args = parse_cli_args()

    display_header()

    display_cli_args(args)

    _, fractal_image = generate_fractal(args)

    console.print("Saving output image...", style="yellow")
    fractal_image.save(args.output_image_path)


if __name__ == '__main__':
    main()
