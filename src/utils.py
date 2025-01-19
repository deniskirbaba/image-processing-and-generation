import matplotlib.pyplot as plt
import torch


def show_images_side_by_side(first: torch.Tensor, second: torch.Tensor, titles: None | list[str] = None) -> None:
    """
    Displays two images side by side.
    """

    def to_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Helper function to convert tensor to the desired dtype."""
        tensor_min, tensor_max = tensor.min(), tensor.max()
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        return (tensor * 255).to(dtype)

    if first.dtype != torch.uint8:
        first = to_dtype(first, torch.uint8)
    if second.dtype != torch.uint8:
        second = to_dtype(second, torch.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(first.permute(1, 2, 0))
    axes[0].axis("off")

    axes[1].imshow(second.permute(1, 2, 0))
    axes[1].axis("off")

    if titles:
        axes[0].set_title(titles[0])
        axes[1].set_title(titles[1])

    plt.tight_layout()
    plt.show()
