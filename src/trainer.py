import multiprocessing
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import wandb
from src.cityscapes_data_handler import Cityscapes
from src.metrics import Metrics
from src.unet import UNet


class Trainer:
    """
    Represents model training, validation and testing pipelines.
    """

    def __init__(
        self,
        config: dict,
        device: torch.device,
        dtype: torch.dtype,
        ignore_index: int,
        idx2name: dict,
        project_name: str,
        model_save_folder: Path,
    ) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype
        self.ignore_index = ignore_index
        self.idx2name = idx2name
        self.project_name = project_name
        self.model_save_folder = model_save_folder

    def prepare(
        self, cityscapes_folder: str | Path, subset: bool = False, compile: bool = True
    ) -> None:
        """
        Prepares all aspects before training model

        Initializes data loaders, model, criterion, optimizer and metrics object.

        If `subset` = True, then will be created small subsets of datasets of sizes:
        (train=64, val=8, test=8), otherwise using whole datasets.
        """
        # Init data loaders
        train = Cityscapes(
            root=cityscapes_folder,
            split="train",
            is_transform=True,
            img_size=self.config["img_size"],
        )
        orig_val = Cityscapes(
            root=cityscapes_folder, split="val", is_transform=True, img_size=self.config["img_size"]
        )
        val = Subset(orig_val, indices=list(range(0, len(orig_val) // 2)))
        test = Subset(orig_val, indices=list(range(len(orig_val) // 2, len(orig_val))))

        if subset:
            # Then pick only small subset from dataset
            train_subset_ids = np.random.choice(len(train), size=64, replace=False)
            val_subset_ids = np.random.choice(len(val), size=8, replace=False)
            test_subset_ids = np.random.choice(len(test), size=8, replace=False)

            train = Subset(train, train_subset_ids)
            val = Subset(val, val_subset_ids)
            test = Subset(test, test_subset_ids)

        self.train_loader = DataLoader(
            train,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=multiprocessing.cpu_count(),
        )
        self.val_loader = DataLoader(
            val,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=multiprocessing.cpu_count(),
        )
        self.test_loader = DataLoader(
            test,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=multiprocessing.cpu_count(),
        )

        # Init model
        self.model = UNet(
            img_size=self.config["img_size"],
            in_channels=3,
            depth=self.config["depth"],
            n_classes=self.config["classes"],
            start_channels=self.config["start_channels"],
        ).to(self.device)

        if compile:
            self.model = torch.compile(self.model)

        # Init loss and optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])

        # Init metrics object
        self.metrics = Metrics(
            num_classes=self.config["classes"], idx_to_class_name=self.idx2name, device=self.device
        )

    def train_batch(self, images, labels):
        images, labels = images.to(self.device), labels.to(self.device)

        # Forward pass
        logits = self.model(images)
        loss = self.criterion(logits, torch.squeeze(labels, dim=1).to(dtype=torch.long))

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Step with optimizer
        self.optimizer.step()

        return loss.detach().cpu().item()

    def val_batch(self, images, labels):
        images, labels = images.to(self.device), labels.to(self.device)

        # Forward pass
        logits = self.model(images)

        # Calculate validation loss
        loss = self.criterion(logits, torch.squeeze(labels, dim=1).to(dtype=torch.long))

        # Accumulate metrics
        # 1. Reshape logits and labels
        logits = torch.permute(logits, (0, 2, 3, 1)).reshape(-1, self.config["classes"])
        labels = torch.squeeze(labels, dim=1).view(-1).to(dtype=torch.long)
        # 2. Calc ignored mask
        ign_mask = labels == self.ignore_index
        # 3. Update metrics
        self.metrics.update(logits[~ign_mask], labels[~ign_mask])

        return loss.detach().cpu().item()

    def train(self):
        self.metrics.reset()

        # Tell wandb to watch what the model gets up to: gradients, weights, and more!
        wandb.watch(self.model, self.criterion, log="all", log_freq=5)

        # Run training and track with wandb
        example_ct = 0  # number of examples seen
        batch_ct = 0
        for epoch in tqdm(range(self.config["epochs"]), desc="Epochs"):
            # Train phase
            loss_history = []
            self.model.train()
            for images, labels in tqdm(
                self.train_loader, desc="Training", total=len(self.train_loader), leave=False
            ):
                train_loss = self.train_batch(images, labels)
                loss_history.append(train_loss)
                example_ct += len(images)
                batch_ct += 1

                # Report metrics every second batch
                if ((batch_ct) % 2) == 0:
                    wandb.log({"epoch": epoch, "train batch loss": train_loss}, step=example_ct)

            wandb.log({"train epoch loss": np.mean(loss_history)}, step=example_ct)

            # Validation phase
            loss_history = []
            self.model.eval()
            with torch.no_grad():
                for images, labels in tqdm(
                    self.val_loader, desc="Validation", total=len(self.val_loader), leave=False
                ):
                    val_loss = self.val_batch(images, labels)
                    loss_history.append(val_loss)

            # Log validation loss and metrics via wandb
            wandb.log({"val loss": np.mean(loss_history)}, step=example_ct)
            self.metrics.compute_log_reset(example_ct=example_ct, test=False)

            # Save model after each epoch
            Path.mkdir(self.model_save_folder, parents=True, exist_ok=True)
            save_model_name = f'{self.config["architecture"]}_{self.config["start_channels"]}_{self.config["depth"]}_{self.config["epochs"]}_{epoch}'
            torch.save(self.model, self.model_save_folder / f"{save_model_name}.pt")

    def test(self):
        self.metrics.reset()

        self.model.eval()

        with torch.no_grad():
            for images, labels in tqdm(
                self.test_loader, desc="Test", total=len(self.test_loader), leave=False
            ):
                images, labels = images.to(self.device), labels.to(self.device)
                logits = self.model(images)

                # Accumulate metrics
                # 1. Reshape logits and labels
                logits = torch.permute(logits, (0, 2, 3, 1)).reshape(-1, self.config["classes"])
                labels = torch.squeeze(labels, dim=1).view(-1).to(dtype=torch.long)
                # 2. Calc ignored mask
                ign_mask = labels == self.ignore_index
                # 3. Update metrics
                self.metrics.update(logits[~ign_mask], labels[~ign_mask])

        # Log test metrics via wandb
        self.metrics.compute_log_reset(example_ct=None, test=True)

    def run_model_pipeline(self) -> UNet:
        """
        Run training and testing phases.
        Logges to wandb.
        Should be run after invoking prepare method.
        """
        # tell wandb to get started
        with wandb.init(project=self.project_name, config=self.config):
            # access all HPs through wandb.config, so logging matches execution!

            self.train()
            self.test()

        return self.model
