import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from monai.inferers import DiffusionInferer
from models.diffusion_denoiser import get_diffusion_stack
from nih_dataset import get_loaders

def train_single(res=1024):
    DEVICE = torch.device("cuda")
    torch.set_float32_matmul_precision('high') # L40S optimization
    
    model, scheduler = get_diffusion_stack(res=res)
    model.to(DEVICE)
    # model = torch.compile(model) # Optional: Adds 20% speedup on L40S
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler = torch.amp.GradScaler('cuda')
    inferer = DiffusionInferer(scheduler)
    
    train_loader, _ = get_loaders(batch_size=4, res=res)

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

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix(loss=loss.item())

        torch.save(model.state_dict(), f"denoiser_single_{res}.pt")

if __name__ == "__main__":
    train_single(res=1024)