import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
torch.manual_seed(95)

transform_to_tensor = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True), 
    ])

transform_to_image = v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        # v2.Grayscale(num_output_channels=1),
        v2.ToPILImage(),
    ])

def process_image(image_path, intensity):
    img = Image.open(image_path)

    img = transform_to_tensor(img)

    img_r = img * intensity

    img_p = torch.poisson(img_r)
    img_p = img_p / intensity

    #Save img_p as a PNG file
    img_poisson = transform_to_image(img_p)
    input_path = Path(image_path)
    output_path = Path(f'{str(input_path.parent.parent)}/images_n_{intensity}') / input_path.name
    # output_path = output_path.parent / f"{output_path.name}_poisson_{intensity}.png"
    img_poisson.save(output_path)
    print(f"Processed image saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Process an image using an integer argument.")
    parser.add_argument("-n", "--n", dest="intensity", type=int, required=True,
                        help="N value: How much quantum mottle noise to add to the image (higher values = less noise)")
    parser.add_argument("-p", "--path", dest="path", required=True,
                        help="Path to the input image or directory")
    return parser.parse_args()


def main():
    args = parse_args()

    path = Path(args.path)
    i = 0
    if path.is_dir():
        for file_path in sorted(path.iterdir()):
            if file_path.is_file():
                process_image(file_path, args.intensity)
                i += 1
                if i > 100:
                    break
    else:
        process_image(path, args.intensity)


if __name__ == "__main__":
    main()

