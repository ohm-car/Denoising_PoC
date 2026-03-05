import torch
import torchxrayvision as xrv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix, multilabel_confusion_matrix
import timm
from datasets.nih_dataset import get_nih_loaders
import argparse

# 1. Configuration
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
CSV_PATH = "./NIH_Chest_XRay/Data_Entry_2017.csv"
IMG_DIR = "./NIH_Chest_XRay/images"
BATCH_SIZE = 256  # Keep low for 1024x1024
MODEL_NAME = 'densenet121'
IMG_RES = 224

def plot_confusion_matrices(mcm, pathologies, output_path="confusion_matrices.png"):
    """Plots a grid of 2x2 confusion matrices for all 14 pathologies."""
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.ravel()
    
    for i, path in enumerate(pathologies):
        sns.heatmap(mcm[i], annot=True, fmt='d', ax=axes[i], cmap='Blues', cbar=False)
        axes[i].set_title(f"Pathology: {path}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
        
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Confusion matrices saved to {output_path}")

def main():
    # 2. Load Data (Phase 2 focuses on Test Set)
    print("Loading Test DataLoader...")
    test_loader, pathologies = get_nih_loaders(
        # CSV_PATH, IMG_DIR, batch_size=BATCH_SIZE, resize_to=1024
        CSV_PATH, IMG_DIR, batch_size=BATCH_SIZE, resize_to=IMG_RES
    )

    # # 3. Load Model
    # print(f"Initializing {MODEL_NAME}...")
    # model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=14)
    # model = model.to(DEVICE).eval()

    # 3. Load Model (Using TorchXRayVision NIH Weights)
    print("Initializing DenseNet-121 with XRV NIH Weights...")
    model = xrv.models.DenseNet(weights="densenet121-res224-nih")
    model = model.to(DEVICE).eval()

    # XRV outputs 18 classes; we need to map to the standard NIH-14 order

    xrv_pathologies = model.pathologies
    indices = [xrv_pathologies.index(p) for p in pathologies if p in xrv_pathologies]

    print(pathologies, [xrv_pathologies[i] for i in indices])

    all_preds = []
    all_labels = []

    # 4. Inference Loop

    full_loop = input('Run the full validation set? 1 for yes, 0 for no')
    print(full_loop)

    print("Starting Inference...")

    if int(full_loop) != 1:
        with torch.no_grad():
            for j, (images, labels) in enumerate(tqdm(test_loader)):

                if j > 3:
                    break

                images = images.to(DEVICE)

                # Images are already 1-channel from Dataset.
                # Scale [0, 1] to XRV's [-1024, 1024]
                # images = (images * 2048) - 1024

                # print("Value of j: ", j)
                # Use Sigmoid for multi-label probabilities

                print(f"Scaled Min: {images.min().item()}, Max: {images.max().item()}, Mean: {images.mean().item()}")

                logits = model(images)
                preds = torch.sigmoid(logits)

                # ### NIH/XRV SPECIFIC CHANGE: Slice 15-class output down to 14-class NIH order ###
                preds_filtered = preds[:, indices]
                
                # all_preds.append(preds.cpu().numpy())
                all_preds.append(preds_filtered.cpu().numpy())
                all_labels.append(labels.numpy())

    else:
        with torch.no_grad():
            for images, labels in tqdm(test_loader):

                images = images.to(DEVICE)

                # Images are already 1-channel from Dataset.
                # Scale [0, 1] to XRV's [-1024, 1024]
                # images = (images * 2048) - 1024

                # print("Value of j: ", j)
                # Use Sigmoid for multi-label probabilities

                print(f"Scaled Min: {images.min().item()}, Max: {images.max().item()}, Mean: {images.mean().item()}")

                logits = model(images)
                preds = torch.sigmoid(logits)

                # ### NIH/XRV SPECIFIC CHANGE: Slice 15-class output down to 14-class NIH order ###
                preds_filtered = preds[:, indices]
                
                # all_preds.append(preds.cpu().numpy())
                all_preds.append(preds_filtered.cpu().numpy())
                all_labels.append(labels.numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # 5. Calculate Metrics
    results = {}
    print(f"\n--- Baseline Results ({IMG_RES}x{IMG_RES}) ---")
    for i, path in enumerate(pathologies):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            results[path] = auc
            print(f"{path:20} AUC: {auc:.4f}")
        except ValueError:
            results[path] = np.nan

    # 6. Confusion Matrices
    # Threshold at 0.5 for binary classification per label
    binary_preds = (all_preds > 0.5).astype(int)
    # print(all_labels, len(all_labels[0]), "ALL_LABELS")
    # print(binary_preds, len(binary_preds[0]), "binary_preds")
    mcm = multilabel_confusion_matrix(all_labels, binary_preds)
    
    plot_confusion_matrices(mcm, pathologies)
    
    # Save AUCs to CSV for Phase 5 comparison
    pd.Series(results).to_csv("baseline_auc_results.csv")

if __name__ == "__main__":
    main()