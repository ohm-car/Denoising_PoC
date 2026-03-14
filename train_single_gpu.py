import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from monai.inferers import DiffusionInferer
from models.diffusion_denoiser import get_diffusion_stack
from datasets.nih_dataset import get_nih_loaders

def train_single(csv_path, img_dir, res=1024):
    DEVICE = torch.device("cuda")
    torch.set_float32_matmul_precision('high') 
    
    model, scheduler = get_diffusion_stack(res=res)
    model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler = torch.amp.GradScaler('cuda')
    inferer = DiffusionInferer(scheduler)
    
    # Updated to receive three loaders
    train_loader, val_loader, _, _ = get_nih_loaders(csv_path, img_dir, batch_size=4, resize_to=res)

    model.train()
    for epoch in range(50):
        loop = tqdm(train_loader, desc=f"Epoch {epoch}")
        for images, _ in loop:
            images = images.to(DEVICE)
            
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

        torch.save(model.state_dict(), f"denoiser_1024_epoch_{epoch}.pt")

if __name__ == "__main__":
    # Update these paths to your actual local paths
    train_single(csv_path="NIH_Chest_XRay/Data_Entry_2017.csv", img_dir="./NIH_Chest_XRay/images")