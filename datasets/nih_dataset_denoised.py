import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from PIL import Image
import torchvision.transforms as transforms

"""Dataset access file that generates the dataloaders, does pre-processing, and implements the getitem method.
This is for the denoised NIH Chest X-Ray 14 dataset, for the 14-class classification problem."""

class DenoisedNIHDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): Dataframe for the specific split.
            img_dir (str): Path to your NEW denoised images folder.
            transform (callable, optional): Standard PyTorch transforms.
        """
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
        
        # Load the denoised image. 
        # Note: If your DDPM outputted grayscale, .convert('RGB') keeps it 3-channel for the model.
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        labels = self.df.iloc[idx][self.pathologies].values.astype('float32')
        return image, torch.tensor(labels)

def get_denoised_loaders(csv_path, denoised_img_dir, batch_size=16, resize_to=512, sample_frac=1.0):
    """
    Creates a 3-way Patient-Agnostic Split (70% Train, 10% Val, 20% Test).
    """
    df = pd.read_csv(csv_path)
    
    # 1. Multi-hot encoding
    pathologies = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
        'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]
    for path in pathologies:
        df[path] = df['Finding Labels'].map(lambda x: 1 if path in x else 0)

    # 2. Triple Split (Train/Val/Test) by Patient ID
    # Split 1: Separate Test set (20%)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df['Patient ID']))
    
    df_train_val = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx]

    # Split 2: Separate Val set from the remaining 80% (approx 12.5% of 80% = 10% total)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
    train_idx, val_idx = next(gss_val.split(df_train_val, groups=df_train_val['Patient ID']))
    
    train_df = df_train_val.iloc[train_idx]
    val_df = df_train_val.iloc[val_idx]

    # 3. Optional Subsampling (Useful for quick POC iterations)
    if sample_frac < 1.0:
        train_df = train_df.sample(frac=sample_frac, random_state=42)

    # 4. Transforms
    # For training, we add augmentation to avoid overfitting.
    train_transform = transforms.Compose([
        transforms.Resize((resize_to, resize_to)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((resize_to, resize_to)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 5. Loaders
    train_ds = DenoisedNIHDataset(train_df, denoised_img_dir, transform=train_transform)
    val_ds   = DenoisedNIHDataset(val_df,   denoised_img_dir, transform=val_test_transform)
    test_ds  = DenoisedNIHDataset(test_df,  denoised_img_dir, transform=val_test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader