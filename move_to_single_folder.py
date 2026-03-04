import os
import shutil
from tqdm import tqdm
from glob import glob

source_pattern = "./NIH_Chest_XRay/images_0*/images"
dest_folder = "./NIH_Chest_XRay/images"

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

# Get list of all subdirectories
subdirs = glob(source_pattern)

print(f"Found {len(subdirs)} subfolders. Starting move...")

for subdir in subdirs:
    files = os.listdir(subdir)
    print(f"Moving {len(files)} files from {subdir}...")
    
    for f in tqdm(files):
        if f.endswith('.png'):
            src_path = os.path.join(subdir, f)
            dest_path = os.path.join(dest_folder, f)
            
            # Using os.rename for speed (instant on same drive)
            os.rename(src_path, dest_path)

print(f"Consolidation complete. Total images in {dest_folder}: {len(os.listdir(dest_folder))}")