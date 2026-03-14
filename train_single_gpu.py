import torch
from tqdm import tqdm
from monai.inferers import DiffusionInferer
from models.diffusion_denoiser import get_diffusion_stack
from datasets.nih_dataset import get_nih_loaders

# --- Global Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = "./NIH_Chest_XRay/Data_Entry_2017.csv"
IMG_DIR = "./NIH_Chest_XRay/images"
BATCH_SIZE = 1
IMG_RES = 512
LEARNING_RATE = 2e-5
EPOCHS = 50

def main():
    torch.set_float32_matmul_precision('high')
    
    # 1. Load Model & Infrastructure
    model, scheduler = get_diffusion_stack(res=IMG_RES)
    model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')
    inferer = DiffusionInferer(scheduler)
    
    # 2. Get DataLoaders
    train_loader, val_loader, _, _ = get_nih_loaders(
        CSV_PATH, IMG_DIR, batch_size=BATCH_SIZE, resize_to=IMG_RES
    )

    # 3. Training Loop
    print(f"Starting Training at {IMG_RES}x{IMG_RES}...")
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        
        for images, _ in loop:
            images = images.to(DEVICE)
            
            # Use BFloat16 for L40S/RTX 6000 stability
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                noise = torch.randn_like(images)
                t = torch.randint(0, 1000, (images.shape[0],), device=DEVICE)
                noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=t)
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loop.set_postfix(loss=loss.item())

        # Checkpoint every epoch
        torch.save(model.state_dict(), f"denoiser_res{IMG_RES}_latest.pt")

if __name__ == "__main__":
    main()