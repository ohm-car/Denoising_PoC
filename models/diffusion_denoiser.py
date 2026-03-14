import torch
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler

def get_diffusion_stack(res=1024):
    # Scale channels and attention based on resolution
    if res >= 1024:
        channels = (128, 256, 512, 512, 1024)
        att_levels = (False, False, False, True, True)
    else:
        channels = (128, 256, 512, 512)
        att_levels = (False, False, True, True)

    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=channels,
        attention_levels=att_levels,
        num_res_blocks=2,
        num_head_channels=64,
    )
    
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        schedule="linear_beta"
    )
    return model, scheduler