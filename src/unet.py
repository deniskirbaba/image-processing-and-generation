import torch
import torch.nn as nn
from torchvision.transforms.v2.functional import center_crop


class DoubleConv(nn.Module):
    """
    Block of 2 sequential conv layers.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, dtype=torch.bfloat16):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, dtype=dtype),
            nn.BatchNorm2d(out_ch, dtype=dtype),
            nn.Hardswish(),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=kernel_size // 2, dtype=dtype),
            nn.BatchNorm2d(out_ch, dtype=dtype),
            nn.Hardswish(),
        )

    def forward(self, x):
        return self.layer(x)


class UNet(nn.Module):
    """
    U-Net implementation.

    Parameters:
        - img_size: Tuple[int, int], input image dimensions (H, W).
        - in_channels: Number of input channels.
        - depth: Number of downsampling layers.
        - n_classes: Number of output classes.
        - start_channels: Number of base feature map channels.
        - dtype: Desired PyTorch datatype.
    """
    def __init__(self, img_size=(512, 1024), in_channels=3, depth=2, n_classes=19, start_channels=16, dtype=torch.bfloat16):
        super().__init__()

        self.img_size = img_size
        self.n_classes = n_classes
        self.depth = depth
        self.start_channels = start_channels  # Number of channels after the first Conv layer (doubled for next)
        self.dtype = dtype

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        cur_channels = start_channels
        self.down_blocks.append(DoubleConv(in_channels, cur_channels, dtype=dtype))
        for _ in range(depth):
            self.down_blocks.append(
                nn.Sequential(
                    nn.MaxPool2d(2),
                    DoubleConv(cur_channels, cur_channels * 2, dtype=dtype),
                )
            )
            cur_channels *= 2

        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(cur_channels, cur_channels * 2, dtype=dtype),
            nn.ConvTranspose2d(
                in_channels=cur_channels * 2,
                out_channels=cur_channels,
                kernel_size=2,
                stride=2,
                dtype=torch.bfloat16,
            ),
        )

        cur_channels *= 2  # Because of concat with tensor

        # Up part
        self.up_blocks = nn.ModuleList()
        for _ in range(depth):
            self.up_blocks.append(
                nn.Sequential(
                    DoubleConv(cur_channels, cur_channels // 2, dtype=dtype),
                    nn.ConvTranspose2d(
                        in_channels=cur_channels // 2,
                        out_channels=cur_channels // 4,
                        kernel_size=2,
                        stride=2,
                        dtype=torch.bfloat16,
                    ),
                )
            )
            cur_channels //= 2

        # Final layers
        self.final_conv = nn.Sequential(
            DoubleConv(cur_channels, cur_channels // 2, dtype=dtype),
            nn.Conv2d(cur_channels // 2, n_classes, kernel_size=1, dtype=dtype),
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass of UNet.

        Inputs:
            - x: torch.Tensor of shape [batch, channels, height, width]
        """
        mem = []

        # Downsampling part
        for block in self.down_blocks:
            x = block(x)
            mem.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Upsampling part
        for block in self.up_blocks:
            x = block(torch.concat((x, center_crop(mem.pop(), list(x.shape[2:]))), dim=1))

        # Final layers
        x = self.final_conv(torch.concat((x, center_crop(mem.pop(), list(x.shape[2:]))), dim=1))

        return x
