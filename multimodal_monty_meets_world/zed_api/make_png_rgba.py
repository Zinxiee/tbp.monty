from PIL import Image
import argparse
import os

''' Example usage:
For default output PNG path (rgb_x_rgba.png):
$ python jpg_to_reshaped_png.py path/to/zed_rgb.png

To specify output PNG path:
$ python jpg_to_reshaped_png.py path/to/zed_rgb.png --output_png_path path/to/output_rgba.png
'''


def convert_zed_png_to_rgba(input_png_path, output_png_path="rgb_x_rgba.png"):
    """
    Convert a ZED RGB PNG file to RGBA PNG.

    Args:
        input_png_path: Path to the input PNG file from ZED capture.
        output_png_path: Path to save the output PNG file
    """
    try:
        if not input_png_path.lower().endswith(".png"):
            raise ValueError(f"Input file must have .png extension, got: {input_png_path}")

        if not os.path.isfile(input_png_path):
            raise FileNotFoundError(f"Input PNG file not found: {input_png_path}")

        print(f"Loading PNG: {input_png_path}")
        input_image = Image.open(input_png_path)
        width, height = input_image.size
        print(f"  Input dimensions: {width}x{height}")
        print(f"  Input mode: {input_image.mode}")

        if input_image.mode != "RGBA":
            output_image = input_image.convert("RGBA")
            print("Converted image mode to RGBA")
        else:
            output_image = input_image.copy()
            print("Image already RGBA; preserving content")

        print(f"Saving RGBA PNG: {output_png_path}")
        output_image.save(output_png_path, format="PNG")

        print(
            f"\nSuccess! PNG saved with dimensions: {width}x{height} (RGBA)"
            f"\nat location: {os.path.abspath(output_png_path)}"
        )
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a ZED RGB PNG to RGBA PNG."
    )
    parser.add_argument("input_png_path", help="Path to the input PNG file")
    parser.add_argument(
        "--output_png_path",
        default="rgb_x_rgba.png",
        help="Path to save the output RGBA PNG file (default: rgb_x_rgba.png)",
    )

    args = parser.parse_args()

    convert_zed_png_to_rgba(args.input_png_path, args.output_png_path)