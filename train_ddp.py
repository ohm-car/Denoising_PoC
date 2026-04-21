import os
import gc
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from models.diffusion_denoiser import get_diffusion_stack
from datasets.nih_dataset import get_nih_loaders

# --- Global Configuration ---
CSV_PATH = "./NIH_Chest_XRay/Data_Entry_2017.csv"
IMG_DIR = "./NIH_Chest_XRay/images"
BATCH_SIZE = 4 # Note: This will be the batch size PER GPU
IMG_RES = 512
LEARNING_RATE = 2e-5
EPOCHS = 50

def main():

    # --- NEW: Parse the run number argument ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, required=True, help="Number for the weights directory")
    args = parser.parse_args()


    # 1. Initialize DDP Process Group
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 2. Memory & Precision Optimizations
    gc.collect()
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Auto-detect precision
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    if local_rank == 0:
        device_name = torch.cuda.get_device_name(local_rank)
        print(f"Hardware: {device_name} | Using Precision: {dtype} | World Size: {dist.get_world_size()}")

    # 3. Model & Optimizer Setup
    model, scheduler = get_diffusion_stack(res=IMG_RES)
    model = model.to(device)
    # Wrap model in DDP
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # GradScaler is only needed if falling back to float16
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))

    # 4. Dataset & Distributed Sampler Setup
    # Extract the raw dataset from the loader to re-wrap it with a DistributedSampler
    temp_loaders, _ = get_nih_loaders(CSV_PATH, IMG_DIR, batch_size=BATCH_SIZE, resize_to=IMG_RES)
    train_dataset = temp_loaders['train'].dataset
    
    sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler, 
        num_workers=1, # Adjust based on your CPU cores
        pin_memory=True
    )

    # 5. Training Loop
    for epoch in range(EPOCHS):
        # Mandatory for DDP to shuffle data differently each epoch
        sampler.set_epoch(epoch)
        model.train()
        
        # Only show the progress bar on the master GPU (rank 0) to avoid terminal spam
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", disable=(local_rank != 0))
        
        for images, _ in loop:
            images = images.to(device)

            # --- NORMALIZATION FIX ---
            # Scale clinical [-1024, 1024] images to [0, 1] 
            images_norm = (images + 1024.0) / 2048.0
            
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                noise = torch.randn_like(images_norm).to(device)
                t = torch.randint(0, scheduler.num_train_timesteps, (images_norm.shape[0],), device=device)
                
                noisy_images = scheduler.add_noise(original_samples=images_norm, noise=noise, timesteps=t)
                
                # Checkpointing for 512px VRAM safety
                noisy_images.requires_grad_(True) 
                # use_reentrant=False is the recommended standard for DDP + Checkpointing
                noise_pred = checkpoint(model, noisy_images, t, use_reentrant=False)
                
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if local_rank == 0:
                loop.set_postfix(loss=f"{loss.item():.4f}")

        # 6. Save Checkpoint (Only on Master Node)
        if local_rank == 0 and (epoch + 1) % 5 == 0:
            save_dir = f"weights/weights_{args.run_num}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.module.state_dict(), f"{save_dir}/denoiser_res_{IMG_RES}_epoch_{epoch+1}.pt")
    # Clean up the process group at the end
    dist.destroy_process_group()

if __name__ == "__main__":
    main()