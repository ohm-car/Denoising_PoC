import torch
import torchxrayvision as xrv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, multilabel_confusion_matrix
import torch.nn.functional as F
from monai.inferers import DiffusionInferer
import argparse

# Assuming these are in your local directory structure
from models.diffusion_denoiser import get_diffusion_stack
from datasets.nih_dataset import get_nih_loaders

# --- Global Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
major, minor = torch.cuda.get_device_capability()
# Only use BF16 on Ampere (8.0) or newer (Ada, Hopper, etc.
DTYPE = torch.bfloat16 if major >= 8 else torch.float16
# DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

CSV_PATH = "./NIH_Chest_XRay/Data_Entry_2017.csv"
IMG_DIR = "./NIH_Chest_XRay/images"
DIFFUSION_WEIGHTS = "weights/weights_164266/denoiser_res_512_epoch_25.pt"

# Resolution Steps
LOAD_RES = 1024
DENOISE_RES = 512
CLASSIFY_RES = 224

BATCH_SIZE = 2

# Purification Settings (t=200 is standard for mild denoising)
PURIFY_TIMESTEP = 200

def plot_confusion_matrices(mcm, pathologies, output_path="denoised_confusion_matrices.png"):
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.ravel()
    for i, path in enumerate(pathologies):
        sns.heatmap(mcm[i], annot=True, fmt='d', ax=axes[i], cmap='Blues', cbar=False)
        axes[i].set_title(f"Denoised: {path}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Confusion matrices saved to {output_path}")

def main():
    # 1. Load Models
    print(f"🚀 Initializing Models on {DEVICE}...")
    
    # Load 512px Denoiser (Architecture: F, F, F, T as established)
    denoiser, scheduler = get_diffusion_stack(res=DENOISE_RES)
    scheduler.clip_sample = False
    denoiser.load_state_dict(torch.load(DIFFUSION_WEIGHTS, map_location=DEVICE))
    denoiser.to(DEVICE).eval()
    inferer = DiffusionInferer(scheduler)

    # Load 224px Classifier
    classifier = xrv.models.DenseNet(weights="densenet121-res224-nih")
    classifier.to(DEVICE).eval()

    # 2. Data Loading (Load at 1024px)
    print(f"📦 Loading Data at {LOAD_RES}px...")
    loaders, pathologies = get_nih_loaders(
        CSV_PATH, IMG_DIR, batch_size=BATCH_SIZE, resize_to=LOAD_RES
    )
    test_loader = loaders['test']
    test_loader.num_workers = 1

    # Match XRV pathology indices
    xrv_pathologies = classifier.pathologies
    indices = [xrv_pathologies.index(p) for p in pathologies if p in xrv_pathologies]

    all_preds = []
    all_labels = []

    # 3. Processing Pipeline
    print("🎬 Starting Denoising + Inference Pipeline...")
    with torch.inference_mode():
        for images, labels in tqdm(test_loader):
            images = images.to(DEVICE)

            # --- STEP A: Resize 1024 -> 512 ---
            img_512 = F.interpolate(images, size=(DENOISE_RES, DENOISE_RES), mode='bilinear')
            img_512 = img_512 / 1024.0

            # --- STEP B: Denoising (Purification) ---
            with torch.amp.autocast(device_type='cuda', dtype=DTYPE):
                # Add noise to the image at t=PURIFY_TIMESTEP
                t_tensor = torch.full((img_512.shape[0],), PURIFY_TIMESTEP, device=DEVICE).long()
                noise = torch.randn_like(img_512)
                noisy_img = scheduler.add_noise(img_512, noise, t_tensor)

                # Run reverse diffusion from t=PURIFY_TIMESTEP to t=0
                scheduler.set_timesteps(num_inference_steps=250) # Faster inference
                # Filter timesteps to start from our specific noise level
                purify_steps = [t for t in scheduler.timesteps if t <= PURIFY_TIMESTEP]
                
                denoised_img = noisy_img
                for t in purify_steps:
                    t_batch = torch.full((img_512.shape[0],), t, device=DEVICE).long()
                    model_output = denoiser(denoised_img, t_batch)
                    denoised_img = scheduler.step(model_output, t, denoised_img)[0]

            # --- STEP C: Resize 512 -> 224 ---
            denoised_img = denoised_img * 1024.0
            img_224 = F.interpolate(denoised_img, size=(CLASSIFY_RES, CLASSIFY_RES), mode='bilinear')

            # --- STEP D: Inference ---
            # Classifier expects [-1024, 1024], which the pipeline maintains
            # logits = classifier(img_224)
            # preds = torch.sigmoid(logits)
            preds = classifier(img_224)
            
            all_preds.append(preds[:, indices].cpu().numpy())
            all_labels.append(labels.numpy())

    # 4. Final Metrics
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    results = {}
    print(f"\n--- Denoised Baseline Results ({CLASSIFY_RES}x{CLASSIFY_RES}) ---")
    for i, path in enumerate(pathologies):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            results[path] = auc
            print(f"{path:20} AUC: {auc:.4f}")
        except ValueError:
            results[path] = np.nan

    print(f"Mean AUC: {np.mean(list(results.values())):.4f}")

    # 5. Output
    mcm = multilabel_confusion_matrix(all_labels, (all_preds > 0.5).astype(int))
    plot_confusion_matrices(mcm, pathologies, output_path="denoised_metrics.png")
    pd.Series(results).to_csv("denoised_auc_results.csv")

if __name__ == "__main__":
    main()
