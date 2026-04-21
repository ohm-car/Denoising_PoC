import os
import sys
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import v2

class NIHDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.pathologies = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
            'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
            'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['Image Index']
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"CRITICAL ERROR: Image not found at {img_path}.")
            print("Please verify your IMG_DIR path and ensure the dataset is fully extracted.")
            sys.exit(1)
        
        if self.transform:
            image = self.transform(image)

        # # Min-Max Normalize to [0, 1]
        # i_min, i_max = image.min(), image.max()
        # if i_max > i_min:
        #     image = (image - i_min) / (i_max - i_min)

        # Clinical Scaling for XRV compatibility [-1024, 1024]
        image = (image * 2048.0) - 1024.0
            
        labels = self.df.iloc[idx][self.pathologies].values.astype('float32')
        return image, torch.tensor(labels)

def get_nih_loaders(csv_path, img_dir, batch_size=16, resize_to=1024, test_size=0.1, val_size=0.1):
    df = pd.read_csv(csv_path)
    pathologies = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
        'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]
    
    for path in pathologies:
        df[path] = df['Finding Labels'].map(lambda x: 1 if path in x else 0)

    # 1. First Split: Test vs (Train + Val)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df['Patient ID']))
    test_df = df.iloc[test_idx].reset_index(drop=True)
    temp_df = df.iloc[train_val_idx].reset_index(drop=True)

    # 2. Second Split: Train vs Val
    adjusted_val_size = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=42)
    train_idx, val_idx = next(gss_val.split(temp_df, groups=temp_df['Patient ID']))
    train_df = temp_df.iloc[train_idx].reset_index(drop=True)
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)

    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_to, resize_to), antialias=True),
        v2.ToDtype(torch.float32, scale=True), 
    ])

    # Calculate optimal num_workers safely for both DDP and Single-GPU
    total_cores = os.cpu_count() if os.cpu_count() is not None else 2
     
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0
         
    # Give each GPU half of its fair share of CPU cores (minimum 1)
    optimal_workers = max(1, (total_cores // world_size) // 2)
     
    if rank == 0:
        print(f"Dynamically set num_workers to: {optimal_workers} per GPU")

    train_loader = DataLoader(NIHDataset(train_df, img_dir, transform), batch_size=batch_size, shuffle=True, num_workers=optimal_workers, pin_memory=True)
    val_loader = DataLoader(NIHDataset(val_df, img_dir, transform), batch_size=batch_size, shuffle=False, num_workers=optimal_workers, pin_memory=True)
    test_loader = DataLoader(NIHDataset(test_df, img_dir, transform), batch_size=batch_size, shuffle=False, num_workers=optimal_workers, pin_memory=True)

    loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

    return loaders, pathologies