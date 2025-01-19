from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

import wandb
from src.metrics import tpr_tnr


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mean, log_var) -> torch.Tensor:
    MSE = F.mse_loss(recon_x, x, reduction="sum")
    # BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return KLD + MSE  # + BCE


class Pipeline:
    """
    Model training, validation and testing.
    """

    def __init__(
        self,
        device: torch.device,
        project_name: str,
    ) -> None:
        self.device = device
        self.project_name = project_name

    def train(
        self,
        config: dict,
        model,
        optimizer,
        train_loader,
        val_proliv_loader,
        val_not_proliv_loader,
        model_save_folder: Path,
    ) -> None:
        with wandb.init(project=self.project_name, tags=["train"], config=config):
            example_ct = 0  # number of examples seen
            batch_ct = 0
            for epoch in tqdm(range(config["epochs"]), desc="Epochs"):
                # Train phase
                loss_history = []
                model.train()
                for images in tqdm(
                    train_loader, desc="Training", total=len(train_loader), leave=False
                ):
                    optimizer.zero_grad()
                    images = images.to(self.device)
                    recon_images, mean, log_var = model(images)
                    loss = loss_function(recon_images, images, mean, log_var) / len(images)
                    loss.backward()
                    optimizer.step()

                    loss_history.append(loss.detach().cpu().item())
                    example_ct += len(images)
                    batch_ct += 1

                    wandb.log({"epoch": epoch, "train loss (batch)": loss}, step=example_ct)

                wandb.log({"train loss (epoch)": np.mean(loss_history)}, step=example_ct)

                # Validation phase
                model.eval()
                with torch.no_grad():
                    # Proliv
                    loss_history = []
                    for images in tqdm(
                        val_proliv_loader,
                        desc="Validation (proliv)",
                        total=len(val_proliv_loader),
                        leave=False,
                    ):
                        images = images.to(self.device)
                        recon_images, mean, log_var = model(images)
                        loss = loss_function(recon_images, images, mean, log_var) / len(images)
                        loss_history.append(loss.detach().cpu().item())
                    wandb.log({"val proliv loss (epoch)": np.mean(loss_history)}, step=example_ct)

                    # Not proliv
                    loss_history = []
                    for images in tqdm(
                        val_not_proliv_loader,
                        desc="Validation (not proliv)",
                        total=len(val_not_proliv_loader),
                        leave=False,
                    ):
                        images = images.to(self.device)
                        recon_images, mean, log_var = model(images)
                        loss = loss_function(recon_images, images, mean, log_var) / len(images)
                        loss_history.append(loss.detach().cpu().item())
                    wandb.log(
                        {"val not proliv loss (epoch)": np.mean(loss_history)}, step=example_ct
                    )

                # Save model after each 10th epoch
                if epoch % 10 == 0:
                    Path.mkdir(model_save_folder, parents=True, exist_ok=True)
                    save_model_name = (
                        f"{config['vae_type']}_{config['latent_dim']}_{config['epochs']}_{epoch}"
                    )
                    torch.save(model, model_save_folder / f"{save_model_name}.pt")

            # Save last model
            Path.mkdir(model_save_folder, parents=True, exist_ok=True)
            save_model_name = f"{config['vae_type']}_{config['latent_dim']}_{config['epochs']}_last"
            torch.save(model, model_save_folder / f"{save_model_name}.pt")

        return model

    def test(self, config: dict, model, test_loader) -> Tuple[float, float]:
        with wandb.init(project=self.project_name, tags=["test"], config=config):
            true, pred = [], []

            model.eval()
            with torch.no_grad():
                for image, label in tqdm(
                    test_loader, desc="Test", total=len(test_loader), leave=False
                ):
                    image = image.to(self.device)
                    recon_image, mean, log_var = model(image)
                    loss = loss_function(recon_image, image, mean, log_var)

                    if loss > config["loss_threshold"]:
                        pred.append(1)
                    else:
                        pred.append(0)
                    true.append(label[0])

            # Log test metrics via wandb
            tpr, tnr = tpr_tnr(np.array(true), np.array(pred))
            wandb.log({"TPR": tpr, "TNR": tnr})

            return tpr, tnr
