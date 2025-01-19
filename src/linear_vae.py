from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim) -> None:
        super().__init__()

        self.fc_block1 = nn.Sequential(
            nn.Linear(input_dim, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.Hardswish(),
        )

        self.fc_block2 = nn.Sequential(
            nn.Linear(4096, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.Hardswish(),
            nn.Dropout(0.25),
        )

        self.fc_block3 = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.Hardswish(),
        )

        self.fc_block4 = nn.Sequential(
            nn.Linear(512, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.Hardswish(),
        )

        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        h_ = self.fc_block4(self.fc_block3(self.fc_block2(self.fc_block1(x))))
        mean = self.fc_mean(h_)
        log_var = self.fc_log_var(h_)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim) -> None:
        super().__init__()

        self.fc_block1 = nn.Sequential(
            nn.Linear(latent_dim, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.Hardswish(),
        )

        self.fc_block2 = nn.Sequential(
            nn.Linear(128, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.Hardswish(),
        )

        self.fc_block3 = nn.Sequential(
            nn.Linear(512, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.Hardswish(),
        )

        self.fc_block4 = nn.Sequential(
            nn.Linear(2048, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.Hardswish(),
            nn.Dropout(0.25),
        )

        self.final_block = nn.Sequential(nn.Linear(4096, output_dim), nn.Sigmoid())

    def forward(self, x) -> torch.Tensor:
        return self.final_block(self.fc_block4(self.fc_block3(self.fc_block2(self.fc_block1(x)))))


class LinearVAE(nn.Module):
    def __init__(self, input_shape: tuple, latent_dim: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.enc = Encoder(3 * input_shape[0] * input_shape[1], latent_dim)
        self.dec = Decoder(latent_dim, 3 * input_shape[0] * input_shape[1])

    def reparameterize(self, mean, log_var) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mean + eps * std

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.enc(self.flatten(x))
        z = self.reparameterize(mu, log_var)

        return torch.reshape(self.dec(z), (x.shape)), mu, log_var
