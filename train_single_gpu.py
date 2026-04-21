import torch
import os
import argparse
import gc
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint
from models.diffusion_denoiser import get_diffusion_stack
from datasets.nih_dataset import get_nih_loaders

# --- Global Configuration ---
DEVICE = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
CSV_PATH = "./NIH_Chest_XRay/Data_Entry_2017.csv"
IMG_DIR = "./NIH_Chest_XRay/images"
BATCH_SIZE = 1
IMG_RES = 512
LEARNING_RATE = 2e-5
EPOCHS = 50

def main():

    # These flags have no effect on the RTX 8000 but massive gains on L40S/5060
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- NEW: Parse the run number argument ---
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--job_id", type=str, required=True, help="Number for the weights directory")
    args = parser.parse_args()

    gc.collect()
    torch.cuda.empty_cache()
    # Turing optimization: TF32 is not available, but we can still use this for allocator efficiency
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # # Auto-detect precision: Turing (7.5) uses float16
    # dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Strict Hardware-Only Precision Check
    major, minor = torch.cuda.get_device_capability()
    # Only use BF16 on Ampere (8.0) or newer (Ada, Hopper, etc.)
    dtype = torch.bfloat16 if major >= 8 else torch.float16

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    print(f"Hardware: {device_name} | Using Precision: {dtype}")

    model, scheduler = get_diffusion_stack(res=IMG_RES)
    model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')
    
    loaders, _ = get_nih_loaders(CSV_PATH, IMG_DIR, batch_size=BATCH_SIZE, resize_to=IMG_RES)
    train_loader = loaders['train']

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        
        for images, _ in loop:
            images = images.to(DEVICE)

            # This ensures the standard Gaussian noise isn't "invisible" to the model.
            images_norm = (images + 1024.0) / 2048.0
            
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                noise = torch.randn_like(images_norm).to(DEVICE)
                t = torch.randint(0, scheduler.num_train_timesteps, (images_norm.shape[0],), device=DEVICE)
                noisy_images = scheduler.add_noise(original_samples=images_norm, noise=noise, timesteps=t)
                
                # Checkpointing for 512px VRAM safety
                noisy_images.requires_grad_(True) 
                noise_pred = checkpoint(model, noisy_images, t, use_reentrant=False)
                
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            # Mandatory for float16 on Turing to prevent NaNs
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loop.set_postfix(loss=f"{loss.item():.4f}")

        if ((epoch + 1) % 5 == 0) or (epoch+1 <= 10):
            save_dir = f"weights/weights_{args.job_id}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{save_dir}/denoiser_res_{IMG_RES}_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()
