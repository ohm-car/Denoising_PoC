import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import torchxrayvision as xrv
import argparse
from tqdm import tqdm
from datasets.nih_dataset import get_nih_loaders
from datasets.covid_dataset import get_covid_loaders


def main():
    parser = argparse.ArgumentParser(description='Train DenseNet121 classifier on NIH or COVID dataset')
    parser.add_argument('--dataset', type=str, choices=['nih', 'covid'], default=None, required=True,
                        help='Dataset to train on: nih or covid')
    parser.add_argument('-b', '--batch_size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('-p', '--p_count', type=int, choices=[0, 12000, 1200, 200], default=0,
                        help='Photon Intensity Value')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Create dataloaders from dataset file
    if args.dataset == 'covid':
        loaders, _ = get_covid_loaders(batch_size=args.batch_size if args.batch_size is not None else 40, resize_to=224, intensity=args.p_count)
        train_loader, val_loader = loaders['train'], loaders['val']
    elif args.dataset == 'nih':
        loaders, _ = get_nih_loaders(csv_path = "./Data/NIH_Chest_XRay/Data_Entry_2017.csv", img_dir = "./Data/NIH_Chest_XRay/images", batch_size=args.batch_size if args.batch_size is not None else 12, resize_to=512, intensity=args.p_count)
        train_loader, val_loader = loaders['train'], loaders['val']
    else:
        raise ValueError("Invalid dataset choice. Choose either 'nih' or 'covid'.")

    # Initialize model with randomly initialized weights
    if args.dataset == 'covid':
        model = xrv.models.DenseNet(weights=None)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, len(train_loader.dataset.classes))
    elif args.dataset == 'nih':
        model = models.resnet50(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(train_loader.dataset.pathologies))
    else:
        raise ValueError("Invalid dataset choice. Choose either 'nih' or 'covid'.")

    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{args.epochs}', leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Val Epoch {epoch+1}/{args.epochs}', leave=False):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100 * correct / total

        print(f'Epoch [{epoch+1}/{args.epochs}] - Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    print('Training completed!')


if __name__ == '__main__':
    main()
