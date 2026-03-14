import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from monai.inferers import DiffusionInferer
from models.diffusion_denoiser import get_diffusion_stack
from nih_dataset import NIHDataset
from tqdm import tqdm

def train_ddp():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    RES = 1024
    torch.set_float32_matmul_precision('high')
    
    # Auto-detect BFloat16 support
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    model, scheduler = get_diffusion_stack(res=RES)
    model = DDP(model.to(device), device_ids=[rank])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))
    
    dataset = NIHDataset(split='train', resolution=RES)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=8)

    for epoch in range(50):
        sampler.set_epoch(epoch)
        pbar = tqdm(loader, disable=(rank != 0))
        for images, _ in pbar:
            images = images.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                noise = torch.randn_like(images)
                t = torch.randint(0, 1000, (images.shape[0],), device=device)
                pred = model(images, t)
                loss = torch.nn.functional.mse_loss(pred, noise)
            
            optimizer.zero_grad()
            if dtype == torch.bfloat16:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

    dist.destroy_process_group()

if __name__ == "__main__":
    train_ddp()