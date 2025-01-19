import torch
from torch import nn


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.Hardswish(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Hardswish(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Hardswish(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.Hardswish(),
            nn.Flatten(),
        )

        self.fc_mean = nn.Linear(512 * 5 * 9, latent_dim)
        self.fc_log_var = nn.Linear(512 * 5 * 9, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        h = self.encoder(x)
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        return mean, log_var


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_shape: tuple) -> None:
        super().__init__()
        self.latent_to_feature = nn.Linear(latent_dim, 512 * 5 * 9)

        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Hardswish(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.Hardswish(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.Hardswish(),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

        self.output_shape = output_shape

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.latent_to_feature(z)
        h = h.view(-1, 512, 5, 9)
        x_reconstructed = self.decoder(h)
        return x_reconstructed


class BigConvVAE(nn.Module):
    def __init__(self, input_shape: tuple, latent_dim: int) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim, input_shape)

    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x: torch.Tensor) -> tuple:
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mean, log_var
