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

def process_image(image_path, intensity, output_path=None, overwrite=False):
    img = Image.open(image_path)

    img = transform_to_tensor(img)

    img_r = img * intensity

    img_p = torch.poisson(img_r)
    img_p = img_p / intensity

    #Save img_p as a PNG file
    img_poisson = transform_to_image(img_p)

    input_path = Path(image_path)
    if output_path is None:
        output_path = Path(f'{str(input_path.parent.parent)}/images_n_{intensity}') / input_path.name
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        print(f"Skipping existing {output_path}")
        return

    img_poisson.save(output_path)
    print(f"Processed image saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Process an image using an integer argument.")
    parser.add_argument("-n", "--n", dest="intensity", type=int, required=True,
                        help="N value: How much quantum mottle noise to add to the image (higher values = less noise)")
    parser.add_argument("-p", "--path", dest="path", required=True,
                        help="Path to the input image or directory")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true",
                        help="Overwrite existing noisy images")
    return parser.parse_args()


def main():
    args = parse_args()

    path = Path(args.path)
    i = 0
    # If user points to the root of the covid19 dataset repository and it contains IM_N_* folders,
    # process every original image into each IM_N_*/<class> folder using the parsed intensity from the folder name.
    if path.is_dir():
        # Detect layout: expected original images under COVID-19_Radiography_Dataset/<class>/images
        original_root = path / 'COVID-19_Radiography_Dataset'
        im_dirs = sorted([p for p in path.iterdir() if p.is_dir() and p.name.startswith('IM_N_')])
        if original_root.exists() and im_dirs:
            # For each IM_N_* directory, parse intensity and process every class
            for im_dir in im_dirs:
                try:
                    intensity = int(im_dir.name.split('_')[-1])
                except Exception:
                    print(f"Cannot parse intensity from folder name {im_dir.name}, skipping")
                    continue

                for class_dir in sorted(im_dir.iterdir()):
                    if not class_dir.is_dir():
                        continue
                    class_name = class_dir.name
                    src_images_dir = original_root / class_name / 'images'
                    if not src_images_dir.exists():
                        print(f"Source images for class {class_name} not found at {src_images_dir}, skipping")
                        continue

                    for file_path in sorted(src_images_dir.iterdir()):
                        if file_path.is_file():
                            target_path = class_dir / file_path.name
                            process_image(file_path, intensity, output_path=target_path, overwrite=args.overwrite)

            return
        else:
            # Fallback: treat provided path as a simple directory of images
            for file_path in sorted(path.iterdir()):
                if file_path.is_file():
                    process_image(file_path, args.intensity, overwrite=args.overwrite)
                    i += 1
                    if i > 100:
                        break
    else:
        process_image(path, args.intensity, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

