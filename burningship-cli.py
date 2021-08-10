import argparse

from cli.common import display_header, display_cli_args, console
from fractals.BurningShip import BurningShip
from fractals.common import Plane2d, HsvColor, ComplexPlane, image_rgb_from_hsv


def parse_cli_args():
    # Create Burning Ship program parser
    parser = argparse.ArgumentParser(description="FractalGen: Burning Ship Generator")

    parser.add_argument("--width", required=False, type=int, default="1920",
                        help="Width of the output image in pixels", dest="width")

    parser.add_argument("--height", required=False, type=int, default="1080",
                        help="Height of the output image in pixels", dest="height")

    parser.add_argument("--real-start", required=False, type=float, default="-2.2",
                        help="Minimum value of the real complex plane", dest="re_start")

    parser.add_argument("--real-end", required=False, type=float, default="1.2",
                        help="Maximum value of the real complex plane", dest="re_end")

    parser.add_argument("--imag-start", required=False, type=float, default="-1.9",
                        help="Minimum value of the imaginary complex plane", dest="im_start")

    parser.add_argument("--imag-end", required=False, type=float, default="0.7",
                        help="Maximum value of the imaginary complex plane", dest="im_end")

    parser.add_argument("--iterations", required=False, type=int, default="100",
                        help="Max iterations for orbital escape", dest="max_iterations")

    parser.add_argument("--color-hue", required=False, type=int, default="204",
                        help="Hue of the color used for the burning ship visualization", dest="color_hue")

    parser.add_argument("--color-saturation", required=False, type=float, default="0.64",
                        help="Saturation of the color used for the burning ship visualization",
                        dest="color_saturation")

    parser.add_argument("--color-intensity", required=False, type=float, default="2.0",
                        help="Intensity of the color used for the burning ship visualization",
                        dest="color_intensity")

    parser.add_argument("--output-image", required=False, type=str, default="burningship.png",
                        help="Path of the output image file", dest="output_image_path")

    parser.add_argument("--use-gpu", required=False, action="store_true",
                        help="Whether to use CUDA to compute the buddhabrot", dest="use_gpu")

    return parser.parse_args()


def main():
    args = parse_cli_args()

    display_header()
    display_cli_args("burning-ship", args)

    plane = Plane2d(args.width, args.height)
    complex_plane = ComplexPlane(args.re_start, args.re_end, args.im_start, args.im_end)
    hsv_color = HsvColor(args.color_hue, args.color_saturation, args.color_intensity)

    burning_ship = BurningShip(plane, complex_plane, args.max_iterations, hsv_color)

    console.print("Generating Burning Ship fractal...", style="yellow")
    burning_ship_image = image_rgb_from_hsv(burning_ship.compute(use_gpu=True))

    console.print("Saving output image...", style="yellow")
    burning_ship_image.save(args.output_image_path)

    console.print("Done.\n", style="green")


if __name__ == '__main__':
    main()
