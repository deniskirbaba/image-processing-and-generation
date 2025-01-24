from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.metrics import Metrics
from src.utils import save_model


def train(
    model: torch.nn.Module,
    optimizer,
    loss_fn,
    trainloader: DataLoader,
    valloader: DataLoader,
    metrics: Metrics,
    device: torch.device,
    config: dict,
    project_name: str,
    save_folder: Path,
) -> torch.nn.Module:
    with wandb.init(project=project_name, tags=["train"], config=config):
        example_ct = 0  # number of examples seen
        batch_ct = 0
        for epoch in tqdm(range(config["epochs"]), desc="Epochs"):
            # Train phase
            epoch_loss = 0
            model.train()
            for images, labels in tqdm(
                trainloader, desc="Training", total=len(trainloader), leave=False
            ):
                optimizer.zero_grad()
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

                batch_loss = loss.detach().cpu().item()
                epoch_loss += batch_loss
                example_ct += len(images)
                batch_ct += 1

                wandb.log(
                    {"epoch": epoch, "train loss (batch)": batch_loss / len(images)},
                    step=example_ct,
                )

            wandb.log(
                {"train loss (epoch)": epoch_loss / len(trainloader.dataset)}, step=example_ct
            )

            # Validation phase
            model.eval()
            with torch.no_grad():
                epoch_loss = 0
                for images, labels in tqdm(
                    valloader,
                    desc="Validation",
                    total=len(valloader),
                    leave=False,
                ):
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    loss = loss_fn(logits, labels)

                    epoch_loss += loss.detach().cpu().item()
                    metrics.update(logits, labels)

                wandb.log(
                    {"val loss (epoch)": epoch_loss / len(valloader.dataset)}, step=example_ct
                )
                metrics.compute_log_reset(example_ct)

            # Save model after each epoch
            save_model(
                model,
                save_folder,
                f"{config['model_base']}_{config['n_layers']}_{config['epochs']}_{epoch}",
            )
    return model


def test(
    model: torch.nn.Module,
    testloader: DataLoader,
    metrics: Metrics,
    device: torch.device,
    config: dict,
    project_name: str,
) -> None:
    with wandb.init(project=project_name, tags=["test"], config=config):
        metrics.reset()
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(testloader, desc="Test", total=len(testloader), leave=False):
                images, labels = images.to(device), labels.to(device)
                logits = model(images)

                metrics.update(logits, labels)

        # Log test metrics via wandb
        metrics.compute_log_reset(example_ct=None, test=True)


def logits_distillation_train(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    optimizer,
    loss_fn,
    trainloader: DataLoader,
    valloader: DataLoader,
    metrics: Metrics,
    device: torch.device,
    config: dict,
    project_name: str,
    save_folder: Path,
) -> torch.nn.Module:
    with wandb.init(project=project_name, tags=["train"], config=config):
        example_ct = 0  # number of examples seen
        batch_ct = 0
        teacher.eval()
        for epoch in tqdm(range(config["epochs"]), desc="Epochs"):
            # Train phase
            epoch_loss = 0
            student.train()
            for images, labels in tqdm(
                trainloader, desc="Training", total=len(trainloader), leave=False
            ):
                images, labels = images.to(device), labels.to(device)
                # Calculate teacher logits
                with torch.no_grad():
                    teacher_logits = teacher(images)
                    teacher_probs = F.softmax(teacher_logits / config["temperature"], dim=-1)
                # Make gradient step based on distillation loss
                optimizer.zero_grad()
                student_logits = student(images)
                student_log_probs = F.log_softmax(student_logits / config["temperature"], dim=-1)
                distillation_loss = torch.sum(
                    teacher_probs * (teacher_probs.log() - student_log_probs)
                ) * (config["temperature"] ** 2)
                loss = (
                    loss_fn(student_logits, labels)
                    + config["distillation_factor"] * distillation_loss
                )
                loss.backward()
                optimizer.step()

                batch_loss = loss.detach().cpu().item()
                epoch_loss += batch_loss
                example_ct += len(images)
                batch_ct += 1

                wandb.log(
                    {"epoch": epoch, "train loss (batch)": batch_loss / len(images)},
                    step=example_ct,
                )

            wandb.log(
                {"train loss (epoch)": epoch_loss / len(trainloader.dataset)}, step=example_ct
            )

            # Validation phase
            student.eval()
            with torch.no_grad():
                epoch_loss = 0
                for images, labels in tqdm(
                    valloader,
                    desc="Validation",
                    total=len(valloader),
                    leave=False,
                ):
                    images, labels = images.to(device), labels.to(device)
                    # Calculate teacher logits
                    teacher_logits = teacher(images)
                    teacher_probs = F.softmax(teacher_logits / config["temperature"], dim=-1)
                    student_logits = student(images)
                    student_log_probs = F.log_softmax(
                        student_logits / config["temperature"], dim=-1
                    )
                    distillation_loss = torch.sum(
                        teacher_probs * (teacher_probs.log() - student_log_probs)
                    ) * (config["temperature"] ** 2)
                    loss = (
                        loss_fn(student_logits, labels)
                        + config["distillation_factor"] * distillation_loss
                    )

                    epoch_loss += loss.detach().cpu().item()
                    metrics.update(student_logits, labels)

                wandb.log(
                    {"val loss (epoch)": epoch_loss / len(valloader.dataset)}, step=example_ct
                )
                metrics.compute_log_reset(example_ct)

            # Save model after each epoch
            save_model(
                student,
                save_folder,
                f"{config['distillation_type']}_{config['distillation_factor']}_{config['temperature']}_{config['model_base']}_{config['n_layers']}_{config['epochs']}_{epoch}",
            )
    return student


def untrained_hidden_layer_distillation_train(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    optimizer,
    loss_fn,
    trainloader: DataLoader,
    valloader: DataLoader,
    metrics: Metrics,
    device: torch.device,
    config: dict,
    project_name: str,
    save_folder: Path,
) -> torch.nn.Module:
    with wandb.init(project=project_name, tags=["train"], config=config):
        example_ct = 0  # number of examples seen
        batch_ct = 0
        teacher.eval()

        for epoch in tqdm(range(config["epochs"]), desc="Epochs"):
            # Train phase
            epoch_loss = 0
            student.train()
            for images, labels in tqdm(
                trainloader, desc="Training", total=len(trainloader), leave=False
            ):
                images, labels = images.to(device), labels.to(device)
                # Calculate teacher hidden layer output
                with torch.no_grad():
                    teacher_hidden, _ = teacher(images)
                # Make gradient step based on distillation loss
                optimizer.zero_grad()
                student_logits, student_hidden = student(images)
                distillation_loss = F.mse_loss(student_hidden, teacher_hidden, reduction="mean")
                loss = (
                    loss_fn(student_logits, labels)
                    + config["distillation_factor"] * distillation_loss
                )
                loss.backward()
                optimizer.step()

                batch_loss = loss.detach().cpu().item()
                epoch_loss += batch_loss * len(images)
                example_ct += len(images)
                batch_ct += 1

                wandb.log(
                    {"epoch": epoch, "train loss (batch)": batch_loss},
                    step=example_ct,
                )

            wandb.log(
                {"train loss (epoch)": epoch_loss / len(trainloader.dataset)}, step=example_ct
            )

            # Validation phase
            student.eval()
            with torch.no_grad():
                epoch_loss = 0
                for images, labels in tqdm(
                    valloader,
                    desc="Validation",
                    total=len(valloader),
                    leave=False,
                ):
                    images, labels = images.to(device), labels.to(device)
                    teacher_hidden, _ = teacher(images)
                    student_logits, student_hidden = student(images)
                    distillation_loss = F.mse_loss(student_hidden, teacher_hidden, reduction="mean")
                    loss = (
                        loss_fn(student_logits, labels)
                        + config["distillation_factor"] * distillation_loss
                    )

                    epoch_loss += loss.detach().cpu().item() * len(images)
                    metrics.update(student_logits, labels)

                wandb.log(
                    {"val loss (epoch)": epoch_loss / len(valloader.dataset)}, step=example_ct
                )
                metrics.compute_log_reset(example_ct)

            # Save model after each epoch
            save_model(
                student,
                save_folder,
                f"{config['distillation_type']}_{config['distillation_factor']}_{config['model_base']}_{config['n_layers']}_{config['epochs']}_{epoch}",
            )
    return student


def hybrid_distillation_train(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    optimizer,
    loss_fn,
    trainloader: DataLoader,
    valloader: DataLoader,
    metrics: Metrics,
    device: torch.device,
    config: dict,
    project_name: str,
    save_folder: Path,
) -> torch.nn.Module:
    with wandb.init(project=project_name, tags=["train"], config=config):
        example_ct = 0  # number of examples seen
        batch_ct = 0
        teacher.eval()

        for epoch in tqdm(range(config["epochs"]), desc="Epochs"):
            # Train phase
            epoch_loss = 0
            student.train()
            for images, labels in tqdm(
                trainloader, desc="Training", total=len(trainloader), leave=False
            ):
                images, labels = images.to(device), labels.to(device)
                # Calculate teacher hidden layer output and logits
                with torch.no_grad():
                    teacher_logits, teacher_hidden = teacher(images)
                    teacher_probs = F.softmax(teacher_logits / config["temperature"], dim=-1)
                # Make gradient step based on distillation loss
                optimizer.zero_grad()
                student_logits, student_hidden = student(images)
                student_log_probs = F.log_softmax(student_logits / config["temperature"], dim=-1)
                # Logits distillation loss component
                logits_dist_loss = (
                    torch.sum(teacher_probs * (teacher_probs.log() - student_log_probs))
                    / len(images)
                    * (config["temperature"] ** 2)
                )
                # Hidden trained distillation loss component
                hidden_trained_dist_loss = F.mse_loss(
                    student_hidden, teacher_hidden, reduction="mean"
                )
                # Main loss
                loss = (
                    loss_fn(student_logits, labels)
                    + config["logits_dist_factor"] * logits_dist_loss
                    + config["hidden_trained_dist_factor"] * hidden_trained_dist_loss
                )
                loss.backward()
                optimizer.step()

                batch_loss = loss.detach().cpu().item()
                epoch_loss += batch_loss * len(images)
                example_ct += len(images)
                batch_ct += 1

                wandb.log(
                    {"epoch": epoch, "train loss (batch)": batch_loss},
                    step=example_ct,
                )

            wandb.log(
                {"train loss (epoch)": epoch_loss / len(trainloader.dataset)}, step=example_ct
            )

            # Validation phase
            student.eval()
            with torch.no_grad():
                epoch_loss = 0
                for images, labels in tqdm(
                    valloader,
                    desc="Validation",
                    total=len(valloader),
                    leave=False,
                ):
                    images, labels = images.to(device), labels.to(device)
                    teacher_logits, teacher_hidden = teacher(images)
                    teacher_probs = F.softmax(teacher_logits / config["temperature"], dim=-1)
                    student_logits, student_hidden = student(images)
                    student_log_probs = F.log_softmax(
                        student_logits / config["temperature"], dim=-1
                    )
                    # Logits distillation loss component
                    logits_dist_loss = (
                        torch.sum(teacher_probs * (teacher_probs.log() - student_log_probs))
                        / len(images)
                        * (config["temperature"] ** 2)
                    )
                    # Hidden trained distillation loss component
                    hidden_trained_dist_loss = F.mse_loss(
                        student_hidden, teacher_hidden, reduction="mean"
                    )
                    # Main loss
                    loss = (
                        loss_fn(student_logits, labels)
                        + config["logits_dist_factor"] * logits_dist_loss
                        + config["hidden_trained_dist_factor"] * hidden_trained_dist_loss
                    )
                    epoch_loss += loss.detach().cpu().item() * len(images)
                    metrics.update(student_logits, labels)

                wandb.log(
                    {"val loss (epoch)": epoch_loss / len(valloader.dataset)}, step=example_ct
                )
                metrics.compute_log_reset(example_ct)

            # Save model after each epoch
            save_model(
                student,
                save_folder,
                f"{config['distillation_type']}_{config['logits_dist_factor']}_{config['hidden_trained_dist_factor']}_{config['model_base']}_{config['n_layers']}_{config['epochs']}_{epoch}",
            )
    return student
