import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import v2

"""Dataset access file that generates the dataloaders, does pre-processing, and implements the getitem method.
This is for the original NIH Chest X-Ray 14 dataset, for the 14-class classification problem."""

class NIHDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): The slice of Data_Entry_2017.csv for this split.
            img_dir (str): Path to the folder containing all 112k images.
            transform (callable, optional): PyTorch transforms.
        """
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform
        
        # Standard NIH-14 Pathology Labels
        self.pathologies = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
            'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
            'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get filename from CSV
        img_name = self.df.iloc[idx]['Image Index']
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load as RGB to ensure compatibility with ImageNet-pretrained weights
        # even though X-rays are fundamentally grayscale.
        try:
            image = Image.open(img_path).convert('L')

            # image = Image.open(img_path).convert('RGB')
        except (IOError, OSError):
            # Fallback for corrupted images if any exist in your download
            image = Image.new('RGB', (1024, 1024), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)

        # 2. XRV CLINICAL SCALING FIX
        # If transform is ToDtype(scale=True) or ToTensor(), image is [0, 1].
        # We shift it to [-1024, 1024] here so the Loader outputs "Ready" tensors.
        if not isinstance(image, torch.Tensor):
            image = transforms.functional.to_tensor(image)

        image = image.to(torch.float32)

        # 2. Min-Max Normalize to [0, 1]
        # We do this to ensure we aren't starting with 0-255
        i_min, i_max = image.min(), image.max()
        if i_max > i_min:
            image = (image - i_min) / (i_max - i_min)

        # 3. Final Clinical Scale
        # Math: (0 * 2048) - 1024 = -1024 | (1 * 2048) - 1024 = 1024
        image = (image * 2048.0) - 1024.0
            
        # Get labels from the one-hot encoded columns we'll create in the helper
        labels = self.df.iloc[idx][self.pathologies].values.astype('float32')
        
        return image, torch.tensor(labels)

def get_nih_loaders(csv_path, img_dir, batch_size=16, resize_to=None, test_size=0.2):
    """
    Args:
        csv_path: Path to Data_Entry_2017.csv
        img_dir: Path to your consolidated 'all_images' folder
        batch_size: Set low (e.g., 4 or 8) if using 1024x1024 on a consumer GPU
        resize_to: Int (e.g., 512) if you want to downsample. Defaults to 1024.
    """
    df = pd.read_csv(csv_path)
    
    # 1. Expand 'Finding Labels' into individual one-hot columns
    pathologies = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
        'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]
    
    for path in pathologies:
        df[path] = df['Finding Labels'].map(lambda x: 1 if path in x else 0)

    # 2. Patient-Agnostic Split (Crucial for Medical Imaging)
    # This prevents the same patient from appearing in both train and test.
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['Patient ID']))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # 3. Define Transforms
    # Defaulting to 1024x1024 if no resize is requested
    target_size = resize_to if resize_to else 1024
    
    test_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_to, resize_to), antialias=True),
        v2.ToDtype(torch.float32, scale=True), 
        # Note: Scaling to [-1024, 1024] happens in __getitem__
    ])

    # 4. Initialize Dataset & Loader
    # For Phase 2 Baseline, we only care about the test_loader
    test_ds = NIHDataset(test_df, img_dir, transform=test_transform)
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    return test_loader, pathologies