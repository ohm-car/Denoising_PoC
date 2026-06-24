import argparse
import os

import numpy as np
import torch
from PIL import Image


def process_image(image_path, intensity):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32)
        tensor = torch.from_numpy(arr)
        processed = (tensor + intensity).clamp(0, 255).byte()
        processed_arr = processed.numpy()

        output_path = f"{os.path.splitext(image_path)[0]}_processed.png"
        Image.fromarray(processed_arr).save(output_path)
        print(f"Processed image saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Process an image using an integer argument.")
    parser.add_argument("intensity", type=int, help="Integer used to adjust the image")
    parser.add_argument("image_path", nargs="?", default="input.png", help="Path to the input image")
    return parser.parse_args()


def main():
    args = parse_args()
    process_image(args.image_path, args.intensity)


if __name__ == "__main__":
    main()

