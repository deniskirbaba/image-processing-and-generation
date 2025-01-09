from ignite.metrics.confusion_matrix import ConfusionMatrix, DiceCoefficient
from ignite.metrics.fbeta import Fbeta
from ignite.metrics.precision import Precision
from ignite.metrics.recall import Recall

import wandb


class Metrics:
    """
    Class for metrics calculation for semantic segmentation task.
    """

    def __init__(self, num_classes: int, idx_to_class_name: dict, device):
        self.idx_to_class_name = idx_to_class_name

        self.precision_per_class = Precision(average=False, device=device)
        self.precision_weighted = Precision(average="weighted", device=device)
        self.precision_macro = Precision(average="macro", device=device)

        self.recall_per_class = Recall(average=False, device=device)
        self.recall_weighted = Recall(average="weighted", device=device)
        self.recall_macro = Recall(average="macro", device=device)

        self.f1_per_class = Fbeta(beta=1, average=False, device=device)
        self.f1_avg = Fbeta(beta=1, average=True, device=device)

        self.cm_per_class = ConfusionMatrix(num_classes=num_classes, average=None, device=device)
        self.dice_per_class = DiceCoefficient(self.cm_per_class)

    def update(self, input, target):
        self.precision_per_class.update((input, target))
        self.precision_weighted.update((input, target))
        self.precision_macro.update((input, target))

        self.recall_per_class.update((input, target))
        self.recall_weighted.update((input, target))
        self.recall_macro.update((input, target))

        self.f1_per_class.update((input, target))
        self.f1_avg.update((input, target))

        self.dice_per_class.update((input, target))

    def compute(self):
        return {
            "precision_per_class": self.precision_per_class.compute(),
            "precision_weighted": self.precision_weighted.compute(),
            "precision_macro": self.precision_macro.compute(),
            "recall_per_class": self.recall_per_class.compute(),
            "recall_weighted": self.recall_weighted.compute(),
            "recall_macro": self.recall_macro.compute(),
            "f1_per_class": self.f1_per_class.compute(),
            "f1_avg": self.f1_avg.compute(),
            "dice_per_class": self.dice_per_class.compute(),
            "dice_avg": self.dice_per_class.compute().mean(),
        }

    def reset(self):
        self.precision_per_class.reset()
        self.precision_weighted.reset()
        self.precision_macro.reset()

        self.recall_per_class.reset()
        self.recall_weighted.reset()
        self.recall_macro.reset()

        self.f1_per_class.reset()
        self.f1_avg.reset()

        self.dice_per_class.reset()

    def compute_log_reset(self, example_ct: int | None = None, test: bool = False) -> None:
        """
        Computes, logs to wandb and resets the metrics objects.
        """
        # Compute
        metrics = self.compute()

        # Format the per class metrics
        precision_per_class = {
            f"precision_{self.idx_to_class_name[i]}": val
            for i, val in enumerate(metrics["precision_per_class"])
        }
        recall_per_class = {
            f"recall_{self.idx_to_class_name[i]}": val
            for i, val in enumerate(metrics["recall_per_class"])
        }
        f1_per_class = {
            f"f1_{self.idx_to_class_name[i]}": val for i, val in enumerate(metrics["f1_per_class"])
        }
        dice_per_class = {
            f"dice_{self.idx_to_class_name[i]}": val
            for i, val in enumerate(metrics["dice_per_class"])
        }

        # Merge all in one dict
        metrics.update(precision_per_class)
        metrics.pop("precision_per_class")
        metrics.update(recall_per_class)
        metrics.pop("recall_per_class")
        metrics.update(f1_per_class)
        metrics.pop("f1_per_class")
        metrics.update(dice_per_class)
        metrics.pop("dice_per_class")

        if test:
            for key, value in list(metrics.items()):
                metrics.pop(key)
                metrics[f"test_{key}"] = value

        wandb.log(metrics, step=example_ct)

        self.reset()

    def compute_wo_wandb_log(self) -> dict:
        """
        Computes metrics on test set without logging to wandb.
        """
        # Compute
        metrics = self.compute()

        # Format the per class metrics
        precision_per_class = {
            f"precision_{self.idx_to_class_name[i]}": val
            for i, val in enumerate(metrics["precision_per_class"])
        }
        recall_per_class = {
            f"recall_{self.idx_to_class_name[i]}": val
            for i, val in enumerate(metrics["recall_per_class"])
        }
        f1_per_class = {
            f"f1_{self.idx_to_class_name[i]}": val for i, val in enumerate(metrics["f1_per_class"])
        }
        dice_per_class = {
            f"dice_{self.idx_to_class_name[i]}": val
            for i, val in enumerate(metrics["dice_per_class"])
        }

        # Merge all in one dict
        metrics.update(precision_per_class)
        metrics.pop("precision_per_class")
        metrics.update(recall_per_class)
        metrics.pop("recall_per_class")
        metrics.update(f1_per_class)
        metrics.pop("f1_per_class")
        metrics.update(dice_per_class)
        metrics.pop("dice_per_class")

        for key, value in list(metrics.items()):
            metrics.pop(key)
            metrics[f"test_{key}"] = value

        return metrics
