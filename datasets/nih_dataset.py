# import torch
# import pandas as pd
# import os
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import torchvision.transforms as T

# class NIHDataset(Dataset):
#     def __init__(self, split='train', resolution=1024, data_path="./data"):
#         self.res = resolution
#         self.pathologies = [
#             "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
#             "Mass", "Nodule", "Pneumonia", "Pneumothorax", 
#             "Consolidation", "Edema", "Emphysema", "Fibrosis", 
#             "Pleural_Thickening", "Hernia"
#         ]
        
#         df = pd.read_csv(os.path.join(data_path, "Data_Entry_2017.csv"))
        
#         # Filter based on official .txt files
#         if split in ['train', 'val']:
#             with open(os.path.join(data_path, "train_val_list.txt"), 'r') as f:
#                 split_list = [line.strip() for line in f.readlines()]
#             df = df[df['Image Index'].isin(split_list)]
            
#             # Internal 90/10 split for training vs validation
#             train_df = df.sample(frac=0.9, random_state=42)
#             self.df = train_df if split == 'train' else df.drop(train_df.index)
#         else:
#             with open(os.path.join(data_path, "test_list.txt"), 'r') as f:
#                 split_list = [line.strip() for line in f.readlines()]
#             self.df = df[df['Image Index'].isin(split_list)]

#         self.transform = T.Compose([
#             T.Resize((self.res, self.res)),
#             T.Grayscale(),
#             T.ToTensor(),
#             T.Normalize(mean=[0.5], std=[0.5]) 
#         ])

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         img_path = os.path.join("./images", row['Image Index'])
#         image = Image.open(img_path).convert("L")
#         return self.transform(image), torch.tensor(row[self.pathologies].values.astype(float))

# def get_nih_loaders(batch_size=4, res=1024, data_path="./data"):
#     train_loader = DataLoader(NIHDataset('train', res, data_path), batch_size=batch_size, shuffle=True, num_workers=8)
#     val_loader = DataLoader(NIHDataset('val', res, data_path), batch_size=batch_size, shuffle=False, num_workers=8)
#     test_loader = DataLoader(NIHDataset('test', res, data_path), batch_size=batch_size, shuffle=False, num_workers=8)
#     return train_loader, val_loader, test_loader


import os
import pandas as pd
import torch
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
            print(f"Error loading {img_path}: {e}")
            image = Image.new('L', (1024, 1024))
        
        if self.transform:
            image = self.transform(image)

        if not isinstance(image, torch.Tensor):
            image = transforms.functional.to_tensor(image)

        image = image.to(torch.float32)

        # Min-Max Normalize to [0, 1]
        i_min, i_max = image.min(), image.max()
        if i_max > i_min:
            image = (image - i_min) / (i_max - i_min)

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

    train_loader = DataLoader(NIHDataset(train_df, img_dir, transform), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(NIHDataset(val_df, img_dir, transform), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(NIHDataset(test_df, img_dir, transform), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, pathologies