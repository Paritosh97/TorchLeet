import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            # (Batch_Size, Channel=3, Height, Weidth) -> (Batch_Size, 128, Height, Weidth)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (Batch_Size, 128, Height, Weidth) -> (Batch_Size, 128, Height, Weidth)
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, Height, Weidth) -> (Batch_Size, 128, Height, Weidth)
            VAE_ResidualBlock(128, 128),

            # (Batch_Size, 128, Height, Weidth) -> (Batch_Size, 128, Height/2, Weidth/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (Batch_Size, 128, Height/2, Weidth/2) -> (Batch_Size, 256, Height/2, Weidth/2)
            VAE_ResidualBlock(128, 256),

            # (Batch_Size, 256, Height/2, Weidth/2) -> (Batch_Size, 128, Height/2, Weidth/2)
            VAE_ResidualBlock(256, 256),

            # (Batch_Size, 256, Height/2, Weidth/2) -> (Batch_Size, 256, Height/4, Weidth/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (Batch_Size, 256, Height/4, Weidth/4) -> (Batch_Size, 512, Height/4, Weidth/4)
            VAE_ResidualBlock(256, 512),

            # (Batch_Size, 512, Height/4, Weidth/4) -> (Batch_Size, 512, Height/4, Weidth/4)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height/4, Weidth/4) -> (Batch_Size, 512, Height/8, Weidth/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

             # (Batch_Size, 512, Height/8, Weidth/8) -> (Batch_Size, 512, Height/8, Weidth/8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height/8, Weidth/8) -> (Batch_Size, 512, Height/8, Weidth/8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height/8, Weidth/8) -> (Batch_Size, 512, Height/8, Weidth/8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height/8, Weidth/8) -> (Batch_Size, 512, Height/8, Weidth/8)
            VAE_AttentionBlock(512),

            # (Batch_Size, 512, Height/8, Weidth/8) -> (Batch_Size, 512, Height/8, Weidth/8)
            VAE_ResidualBlock(512, 512),

            # (Batch_Size, 512, Height/8, Weidth/8) -> (Batch_Size, 512, Height/8, Weidth/8)
            nn.GroupNorm(32, 512),

            # (Batch_Size, 512, Height/8, Weidth/8) -> (Batch_Size, 512, Height/8, Weidth/8)
            nn.SiLU(),

            # (Batch_Size, 512, Height/8, Weidth/8) -> (Batch_Size, 8, Height/8, Weidth/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (Batch_Size, 8, Height/8, Weidth/8) -> (Batch_Size, 8, Height/8, Weidth/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x : (Batch_Size, Channel, Height, Weidth)
        # noise : (Batch_Size, Out_Channels, Height/8, Weidth/8)

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # Padding_L, Padding_R, Padding_T, Padding_B
                x = F.pad(x, (0, 1, 0, 1))

            x = module(x)

        # (Batch_Size, 8, Height/8, Weidth/8) -> two tensors of (Batch_Size, 4, Height/8, Weidth/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # (Batch_Size, 8, Height/8, Weidth/8) -> two tensors of (Batch_Size, 4, Height/8, Weidth/8)
        log_variance = torch.clamp(log_variance, -30.0, 20.0)

        # (Batch_Size, 8, Height/8, Weidth/8) -> two tensors of (Batch_Size, 4, Height/8, Weidth/8)
        variance = log_variance.exp()

        # (Batch_Size, 8, Height/8, Weidth/8) -> two tensors of (Batch_Size, 4, Height/8, Weidth/8)
        stdev = variance.sqrt()

        # Z=N(0, 1) -> N(mean, variance)=X
        x = noise * stdev + mean

        # scale the output
        x = x * 0.18215

        return x