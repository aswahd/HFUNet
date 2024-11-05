import re
from collections import defaultdict
from functools import partial
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple

import kornia as K
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import io

from datasets.registry import register_dataset

from .utils import ResizeLongestSide, pad
from utils import add_salt_and_pepper_noise 


class CenterCrop:
    """
    CenterCrop class to crop an image and its mask centered on the mask.
    """

    def __init__(self, scale: Tuple[float, float], p=0.5) -> None:
        self.scale = scale
        self.p = p

    def __call__(
        self, img: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if np.random.rand() > self.p:
            return img, mask

        *_, H, W = img.shape

        # Randomly sample a scale factor
        scale = np.random.uniform(*self.scale)

        # Compute the center of the mask
        y_indices, x_indices = torch.nonzero(mask[0, 0] > 0, as_tuple=True)
        if len(x_indices) == 0:
            x_center = W / 2
            y_center = H / 2
            # If the mask is empty, default to the center of the image
            x_center = float(x_indices.float().mean())
            y_center = float(y_indices.float().mean())
        else:
            x_center = x_indices.float().mean()
            y_center = y_indices.float().mean()

        # Compute crop size
        new_H = int(H * scale)
        new_W = int(W * scale)

        # Calculate the top-left and bottom-right coordinates of the crop
        x1 = x_center - new_W / 2
        y1 = y_center - new_H / 2
        x2 = x1 + new_W
        y2 = y1 + new_H

        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)

        # Crop the image and mask
        mask = mask[:, :, int(y1) : int(y2), int(x1) : int(x2)]
        img = img[:, :, int(y1) : int(y2), int(x1) : int(x2)]
        return img, mask


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        img_files: List[PosixPath],
        gt_files: List[PosixPath],
        config: Dict,
        mode="Train",
    ):
        self.img_files = [str(f) for f in img_files]
        self.gt_files = [str(f) for f in gt_files]

        print(
            f"Found {len(self.img_files)} {'training' if mode == 'Train' else 'validation'} images and {len(self.gt_files)} masks."
        )

        self.img_size = config["image_size"]

        self.resize = ResizeLongestSide(self.img_size)
        self.pad = partial(pad, target=(self.img_size, self.img_size))

        self.trn_aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomAffine(
                degrees=30, translate=(0.2, 0.2), shear=0, p=0.3, align_corners=False
            ),
            K.augmentation.RandomHorizontalFlip(p=0.3),
            K.augmentation.RandomResizedCrop(
                size=(self.img_size, self.img_size),
                scale=(0.8, 1.0),
                p=0.3,
                align_corners=False,
            ),
            data_keys=["input", "mask"],
            random_apply=(2,),
        )
        self.center_crop = CenterCrop(scale=(0.5, 1.0), p=0.5)

        self.tst_aug = K.augmentation.AugmentationSequential(
            K.augmentation.Resize(
                size=(self.img_size, self.img_size), align_corners=False
            ),
            data_keys=["input", "mask"],
        )

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.mode = mode
        self.num_classes = config["num_classes"]
        self.label_to_class_id = config.get("label_to_class_id")

    def __len__(self):
        return len(self.img_files)

    @staticmethod
    def read_image(path):
        img = io.read_image(path)
        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)
        return img

    @staticmethod
    def read_mask(path):
        return io.read_image(path)

    def remap_labels(self, gt):
        """
        This function is designed to modify the labels in a given tensor, `gt`, according to a predefined mapping. It's particularly useful for when the raw labels are not ordered.

        Parameters:
        - gt (torch.Tensor): The input tensor containing the original labels.

        Returns:
        torch.Tensor: The tensor with specified labels remapped to the target label.
        """

        if hasattr(self, "label_to_class_id") and self.label_to_class_id is not None:
            for k, v in self.label_to_class_id.items():
                gt[gt == k] = v

        return gt

    def __getitem__(self, index):
        img = self.read_image(self.img_files[index]).float() / 255.0
        gt = self.read_mask(self.gt_files[index])
        gt = self.remap_labels(gt).float()
        original_size = gt.shape

        if self.trn_aug is not None and self.mode == "Train":
            img, gt = self.trn_aug(img[None], gt[None])
            img, gt = self.center_crop(img, gt)
        elif self.tst_aug is not None:
            img, gt = self.tst_aug(img[None], gt[None])

        img, gt = img[0], gt[0]
        # remove batch dimension
        # Resize to a square image
        img = self.resize(img[None], order=1)
        # Normalize image
        img = (img - self.mean) / self.std
        input_size = img.shape[2:]
        img = self.pad(img)
        gt = self.pad(self.resize(gt[None], order=0)).type(torch.uint8)
        # remove batch dimension
        img, gt = img[0], gt[0, 0]

        sample = {
            "images": img,
            "masks": gt,
            "input_size": input_size,
            "original_size": original_size,
        }

        return sample

    @staticmethod
    def get_corresponding_image_name(mask_file: str):
        return mask_file.replace("gts", "imgs")

    @staticmethod
    def split_train_test(file_names, test_size, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return train_test_split(file_names, test_size=test_size)

    @classmethod
    def from_path(cls, config, mode="Train"):
        path = Path(config.root)
        mask_file_names = [f for f in path.glob(f"{mode}/gts/*")]
        # mask_file_names = [f for f in path.glob(f"Train/gts/*")]
        mask_file_names.sort()

        if mode == "Train":
            trn_mask_file_names, val_mask_file_names = cls.split_train_test(
                mask_file_names, test_size=1 - config.split, seed=config.get("seed", 0)
            )

            trn_img_file_names = [
                cls.get_corresponding_image_name(str(f)) for f in trn_mask_file_names
            ]

            val_img_file_names = [
                cls.get_corresponding_image_name(str(f)) for f in val_mask_file_names
            ]

            return (
                cls(trn_img_file_names, trn_mask_file_names, config, mode="Train"),
                cls(val_img_file_names, val_mask_file_names, config, mode="Val"),
            )

        # Test
        img_file_names = [
            cls.get_corresponding_image_name(str(f)) for f in mask_file_names
        ]

        return cls(img_file_names, mask_file_names, config, mode="Test")


@register_dataset("default_test_dataset")
class TestDataset(BaseDataset):
    """
    Returns images without any augmentation.
    """

    def __getitem__(self, index):
        img = self.read_image(self.img_files[index])
        img = img.float() / 255.0

        # Uncomment the following lines to test for out-of-distribution detection
        # Add Gaussian noise
        # noise = torch.randn_like(img) * 0.05
        # img = img + noise

        # Blurring
        # img = K.filters.gaussian_blur2d(img[None], (5, 5), (3.0, 3.0))[0]
        # img = K.filters.gaussian_blur2d(img[None], (35, 35), (15.0, 15.0))[0]

        # Salt and pepper noise
        # img = add_salt_and_pepper_noise(img, prob=0.01)

        img_orig = (img.clone() * 255).byte()


        gt = self.read_mask(self.gt_files[index])
        gt = self.remap_labels(gt).long()
        img = (img - self.mean) / self.std
        img = self.resize(img[None], order=1)
        input_size = img.shape[2:]
        img = self.pad(img)[0]

        if gt.ndim == 3:
            gt = gt.squeeze(0)

        return img_orig.permute(1, 2, 0), img, gt, input_size, self.img_files[index]


def group_files_by_patient_id(files: List[PosixPath]) -> Dict[int, List[PosixPath]]:
    """
    Extracts the patient IDs from the filenames.
    """
    # .../1323DP_MX_25.png -> 1323
    pattern = re.compile(r"(\d+)[a-zA-Z]*")
    id2filename = defaultdict(list)
    for filename in files:
        match = pattern.match(filename.stem)
        if match:
            id2filename[int(match.group(1))].append(filename)

    return id2filename


def get_dataloaders(args: dict, dataset: Dataset) -> DataLoader:
    """
    Returns a DataLoader for the given dataset with specified arguments.

    Args:
        args (dict): A dictionary containing DataLoader parameters such as
                     'batch_size' and 'num_workers'.
        dataset (Dataset): The dataset to load.

    Returns:
        DataLoader: A DataLoader configured with the provided arguments.
    """

    return DataLoader(
        dataset,
        batch_size=args.get("batch_size", 4),
        shuffle=True,
        num_workers=args.get("num_workers", 4),
        drop_last=False,
    )
