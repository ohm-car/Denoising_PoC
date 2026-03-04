import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_path, image_dir, split_file=None, transform=None):
        """
        Args:
            csv_path: Path to Data_Entry_2017.csv
            image_dir: Path to the directory containing all images
            split_file: Path to train_val_list.txt or test_list.txt (official NIH splits)
            transform: PyTorch transforms
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # Load main metadata
        df = pd.read_csv(csv_path)
        
        # Filter by official split if provided (handles patient-level separation)
        if split_file:
            with open(split_file, 'r') as f:
                valid_images = f.read().splitlines()
            df = df[df['Image Index'].isin(valid_images)]

        # Define the 14 pathologies
        self.labels = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        
        # Multi-hot encode the labels
        for label in self.labels:
            df[label] = df['Finding Labels'].map(lambda x: 1 if label in x else 0)
            
        self.data = df.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['Image Index']
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image and convert to RGB (standard for torchvision models)
        image = Image.open(img_path).convert('RGB')
        
        # Extract labels as a float tensor
        label_values = self.data.iloc[idx][self.labels].values.astype('float32')
        label_tensor = torch.tensor(label_values)

        if self.transform:
            image = self.transform(image)

        return image, label_tensor