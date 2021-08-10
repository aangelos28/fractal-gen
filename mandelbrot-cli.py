import argparse

from cli.common import display_header, display_cli_args, console
from fractals.Mandelbrot import Mandelbrot
from fractals.common import Plane2d, HsvColor, ComplexPlane


def parse_cli_args():
    # Create Mandelbrot program parser
    parser = argparse.ArgumentParser(description="FractalGen: Mandelbrot Generator")

    parser.add_argument("--width", required=False, type=int, default="1920",
                        help="Width of the output image in pixels", dest="width")

    parser.add_argument("--height", required=False, type=int, default="1080",
                        help="Height of the output image in pixels", dest="height")

    parser.add_argument("--real-start", required=False, type=float, default="-2.2",
                        help="Minimum value of the real complex plane", dest="re_start")

    parser.add_argument("--real-end", required=False, type=float, default="1.2",
                        help="Maximum value of the real complex plane", dest="re_end")

    parser.add_argument("--imag-start", required=False, type=float, default="-1.2",
                        help="Minimum value of the imaginary complex plane", dest="im_start")

    parser.add_argument("--imag-end", required=False, type=float, default="1.2",
                        help="Maximum value of the imaginary complex plane", dest="im_end")

    parser.add_argument("--iterations", required=False, type=int, default="200",
                        help="Max iterations for orbital escape", dest="max_iterations")

    parser.add_argument("--color-hue", required=False, type=int, default="204",
                        help="Hue of the color used for the mandelbrot visualization", dest="color_hue")

    parser.add_argument("--color-saturation", required=False, type=float, default="0.64",
                        help="Saturation of the color used for the mandelbrot visualization",
                        dest="color_saturation")

    parser.add_argument("--color-intensity", required=False, type=float, default="3.0",
                        help="Intensity of the color used for the mandelbrot visualization",
                        dest="color_intensity")

    parser.add_argument("--output-image", required=False, type=str, default="mandelbrot.png",
                        help="Path of the output image file", dest="output_image_path")

    parser.add_argument("--use-gpu", required=False, action="store_true",
                        help="Whether to use CUDA to compute the buddhabrot", dest="use_gpu")

    return parser.parse_args()


def main():
    args = parse_cli_args()

    display_header()
    display_cli_args("mandelbrot", args)

    plane = Plane2d(args.width, args.height)
    complex_plane = ComplexPlane(args.re_start, args.re_end, args.im_start, args.im_end)
    hsv_color = HsvColor(args.color_hue, args.color_saturation, args.color_intensity)

    mandelbrot = Mandelbrot(plane, complex_plane, args.max_iterations, hsv_color)

    console.print("Generating Mandelbrot fractal...", style="yellow")
    mandelbrot_image = mandelbrot.compute(use_gpu=args.use_gpu)

    console.print("Saving output image...", style="yellow")
    mandelbrot_image.save(args.output_image_path)

    console.print("Done.\n", style="green")


if __name__ == '__main__':
    main()
