from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.v2.functional import to_dtype


def show_image(tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.uint8:
        tensor = to_dtype(tensor, torch.uint8, scale=True)
    plt.imshow(tensor.permute(1, 2, 0))


class CylindersTrain(Dataset):
    def __init__(self, data_folder: Path, transforms=None, dtype=torch.float32) -> None:
        self.data_folder = data_folder
        self.images_paths = list(self.data_folder.iterdir())

        # Read all files in RAM for speed-up training
        # Can do this because dataset is small (~50 MB)
        self.images = [
            to_dtype(transforms(read_image(img_path)), dtype, scale=True)
            if transforms
            else read_image(img_path)
            for img_path in self.images_paths
        ]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index) -> torch.Tensor:
        return self.images[index]


class CylindersTest(Dataset):
    def __init__(self, data_folder: Path, transforms=None, dtype=torch.float32) -> None:
        self.imgs_folder = data_folder / "imgs"
        self.images_paths = list(self.imgs_folder.iterdir())
        self.labels = {}
        with open(data_folder / "test_annotation.txt", "r") as f:
            for line in f:
                img_name, label = line.strip().split()
                self.labels[img_name] = int(label)

        # Read all files in RAM for speed-up training
        # Can do this because dataset is small (~15 MB)
        self.images = [
            to_dtype(transforms(read_image(img_path)), dtype, scale=True)
            if transforms
            else read_image(img_path)
            for img_path in self.images_paths
        ]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        return self.images[index], self.labels[self.images_paths[index].name]


class CylindersProliv(Dataset):
    def __init__(self, data_folder: Path, transforms=None, dtype=torch.float32) -> None:
        self.data_folder = data_folder
        self.images_paths = list(self.data_folder.iterdir())

        # Read all files in RAM for speed-up training
        # Can do this because dataset is small (~1 MB)
        self.images = [
            to_dtype(transforms(read_image(img_path)), dtype, scale=True)
            if transforms
            else read_image(img_path)
            for img_path in self.images_paths
        ]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index) -> torch.Tensor:
        return self.images[index]
