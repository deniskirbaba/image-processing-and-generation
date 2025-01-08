import os

import matplotlib.pyplot as plt
import torch
from torch.utils import data
from torchvision.io import read_image
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2.functional import resize, to_dtype


def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def show_image(tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.uint8:
        tensor = to_dtype(tensor, torch.uint8)
    plt.imshow(tensor.permute(1, 2, 0))


def show_images_side_by_side(orig_tensor: torch.Tensor, seg_tensor: torch.Tensor) -> None:
    """
    Displays two images side by side: original on the left and segmented on the right.
    """
    def to_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Helper function to convert tensor to the desired dtype."""
        tensor_min, tensor_max = tensor.min(), tensor.max()
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min) 
        return (tensor * 255).to(dtype)

    if orig_tensor.dtype != torch.uint8:
        orig_tensor = to_dtype(orig_tensor, torch.uint8)
    if seg_tensor.dtype != torch.uint8:
        seg_tensor = to_dtype(seg_tensor, torch.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(orig_tensor.permute(1, 2, 0))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(seg_tensor.permute(1, 2, 0))
    axes[1].set_title("Segmented Image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()



class Cityscapes(data.Dataset):
    """
    Cityscapes dataset.

    Adapted from https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/cityscapes_loader.py.
    """

    colors = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    def __init__(
        self,
        root,
        # which data split to use
        split="train",
        # transform function activation
        is_transform=True,
        # image_size to use in transform function
        img_size: list[int] = [512, 1024],
        dtype=torch.bfloat16,
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 19
        self.img_size = img_size
        self.dtype = dtype
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)

        # contains list of all pngs inside all different folders. Recursively iterates
        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]

        # these are 19
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]

        # these are 19 + 1; "unlabelled" is extra
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        # for void_classes; useful for loss function
        self.ignore_index = 250

        # dictionary of valid classes 7:0, 8:1, 11:2
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        # prints number of images found
        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        # path of image
        img_path = self.files[self.split][index].rstrip()

        # path of label
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )

        # read image
        img = to_dtype(read_image(img_path), self.dtype)

        # read label
        lbl = read_image(lbl_path)
        # encode using encode_segmap function: 0...18 and 250
        lbl = self.encode_segmap(lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl) -> tuple[torch.Tensor, torch.Tensor]:
        return resize(img, self.img_size), resize(
            lbl, self.img_size, interpolation=InterpolationMode.NEAREST_EXACT
        )

    def decode_segmap(self, temp):
        r = temp.clone()
        g = temp.clone()
        b = temp.clone()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = torch.cat((r, g, b), dim=0)
        return rgb

    # there are different class 0...33
    # we are converting that info to 0....18; and 250 for void classes
    # final mask has values 0...18 and 250
    def encode_segmap(self, mask):
        # !! Comment in code had wrong informtion
        # Put all void classes to ignore_index
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask
