import os
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2


class CovidRadiographyDataset(Dataset):
    """
    Dataset loader for the COVID-19 Radiography Database.
    Expects a root data directory with the standard class subfolders:
      COVID, Normal, Viral Pneumonia, Lung Opacity
    """

    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    FOLDER_LABELS = {
        'COVID': 'COVID-19',
        'Normal': 'Normal',
        'Viral_Pneumonia': 'Viral_Pneumonia',
        'Lung_Opacity': 'Lung_Opacity',
    }
    TARGET_NAMES = ('COVID-19', 'Normal', 'Viral_Pneumonia', 'Lung_Opacity')

    def __init__(self, root_dir=None, classes=None, transform=None, intensity=0):
        if root_dir is None:
            root_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'covid19-radiography-database', 'COVID-19_Radiography_Dataset'))
        self.root_dir = root_dir
        self.transform = transform
        self.intensity = intensity
        self.classes = list(classes) if classes is not None else list(self.TARGET_NAMES)
        self.class_to_idx = {class_name: self.TARGET_NAMES.index(class_name) for class_name in self.classes}
        self.samples = self._make_dataset()
        if len(self.samples) == 0:
            raise RuntimeError('Found 0 files in the specified root directory: {}'.format(self.root_dir))

    def _is_image_file(self, filename):
        filename_lower = filename.lower()
        return filename_lower.endswith(self.IMAGE_EXTENSIONS)

    def _make_dataset(self):
        samples = []
        if not os.path.isdir(self.root_dir):
            return samples

        for folder_name in sorted(os.listdir(self.root_dir)):
            folder_path = os.path.join(self.root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            normalized = folder_name.strip()
            if normalized not in self.FOLDER_LABELS:
                continue
            target_name = self.FOLDER_LABELS[normalized]
            if target_name not in self.class_to_idx:
                continue
            target = self.class_to_idx[target_name]

            for root, _, filenames in os.walk(folder_path):
                for filename in sorted(filenames):
                    if self._is_image_file(filename):
                        path = os.path.join(root, filename)
                        if 'masks' in path.split('/'):
                            continue
                        samples.append((path, target))
        print(f"Found {len(samples)} samples in {self.root_dir} across {len(self.classes)} classes.")
        print(f"Example samples: {samples[:5]}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = Image.open(path)
        if image.mode != 'L':
            image = image.convert('L')

        if self.transform is not None:
            image = self.transform(image)
        return image, target


def get_covid_loaders(root_dir=None, batch_size=16, resize_to=224, test_size=0.1, val_size=0.1, intensity=0):
    """
    Load COVID-19 Radiography dataset with train/val/test splits.
    
    Args:
        root_dir: Path to dataset root (defaults to ../Data)
        batch_size: Batch size for dataloaders
        resize_to: Image resize dimension
        test_size: Fraction of data for test set
        val_size: Fraction of data for validation set
        intensity: Photon intensity value (0, 12000, 1200, or 200)
    
    Returns:
        dict: Dictionary with 'train', 'val', 'test' DataLoaders
        tuple: Target class names
    """
    if root_dir is None:
        root_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'covid19-radiography-database', 'COVID-19_Radiography_Dataset'))
    
    target_names = CovidRadiographyDataset.TARGET_NAMES
    
    # Load all samples from the dataset
    full_dataset = CovidRadiographyDataset(root_dir=root_dir, intensity=intensity)
    all_samples = full_dataset.samples
    
    # First split: separate test set
    train_val_samples, test_samples = train_test_split(
        all_samples, test_size=test_size, random_state=42, 
        stratify=[s[1] for s in all_samples]
    )
    
    # Second split: separate validation from training
    adjusted_val_size = val_size / (1 - test_size)
    train_samples, val_samples = train_test_split(
        train_val_samples, test_size=adjusted_val_size, random_state=42,
        stratify=[s[1] for s in train_val_samples]
    )
    
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

    #Delete  later
    optimal_workers = 1
    
    if rank == 0:
        print(f"Dynamically set num_workers to: {optimal_workers} per GPU")
    
    # Create datasets by modifying samples on instances
    train_dataset = CovidRadiographyDataset(root_dir=root_dir, transform=transform, intensity=intensity)
    train_dataset.samples = train_samples
    
    val_dataset = CovidRadiographyDataset(root_dir=root_dir, transform=transform, intensity=intensity)
    val_dataset.samples = val_samples
    
    test_dataset = CovidRadiographyDataset(root_dir=root_dir, transform=transform, intensity=intensity)
    test_dataset.samples = test_samples
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=optimal_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=optimal_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=optimal_workers, pin_memory=True
    )
    
    loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }
    
    return loaders, target_names
