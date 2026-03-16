import torch
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler

def get_diffusion_stack(res=1024):
    # 5 levels for 1024px to ensure the bottleneck handles the spatial complexity
    channels = (128, 256, 512, 512, 1024) if res >= 1024 else (128, 256, 512, 512)
    # channels = (128, 256, 512, 512, 1024) if res >= 1024 else (64, 128, 256, 512)
    att_levels = (False, False, False, True, True) if res >= 1024 else (False, False, True, True)

    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=channels,
        attention_levels=att_levels,
        num_res_blocks=2,
        num_head_channels=64,
        # use_checkpoint=True
    )
    
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        schedule="linear_beta"
    )
    return model, scheduler