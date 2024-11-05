# Description: this file registers commonly used datasets by their names.
import warnings
from pathlib import Path

import kornia as K
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torchvision import io

from datasets import BaseDataset, TestDataset
from datasets.main import CenterCrop, group_files_by_patient_id
from datasets.registry import register_dataset

warnings.filterwarnings("ignore")


@register_dataset("default_segmentation")
class SegmentationDataset(BaseDataset):
    def __init__(self, img_files, gt_files, mode, config):
        super().__init__(img_files, gt_files, config, mode)

        self.trn_aug = None


@register_dataset("shoulder")
class SegmentationDataset(BaseDataset):
    def __init__(self, img_files, gt_files, mode, config):
        super().__init__(img_files, gt_files, config, mode)

        self.trn_aug = K.augmentation.AugmentationSequential(
            # K.augmentation.RandomHorizontalFlip(p=0.2),
            # K.augmentation.RandomVerticalFlip(p=0.2),
            K.augmentation.RandomAffine(degrees=90, translate=(0.1, 0.1), p=0.5),
            data_keys=["input", "mask"],
            # extra_args={
            #     DataKey.MASK: {"resample": Resample.BILINEAR, "align_corners": None}
            # },
        )

    def remap_labels(self, gt):
        return (gt > 0).long()


@register_dataset("sonance")
class WristDataset(BaseDataset):
    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

        # Define custom augmentations here
        self.trn_aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomHorizontalFlip(p=0.4),
            K.augmentation.RandomVerticalFlip(p=0.4),
            K.augmentation.RandomAffine(degrees=30, translate=(0.0, 0.0), p=0.4),
            data_keys=["input", "mask"],
            # extra_args={
            #     DataKey.MASK: {"resample": Resample.BILINEAR, "align_corners": None}
            # },
        )

    @staticmethod
    def split_train_test(file_names, test_size, seed=None):
        """
        Group files by patient id and split them into train and validation sets.
        """
        patient_id_2_filename = group_files_by_patient_id(file_names)
        patient_ids = list(patient_id_2_filename.keys())

        train_ids, val_ids = train_test_split(
            patient_ids, test_size=test_size, random_state=seed
        )
        train_files, val_files = [], []
        for _id in train_ids:
            train_files.extend(patient_id_2_filename[_id])

        for _id in val_ids:
            val_files.extend(patient_id_2_filename[_id])

        return train_files, val_files

    @classmethod
    def from_path(cls, config, mode="Train"):
        """
        Dataset object from a directory containing images and masks.
        """

        path = Path(config.root)
        mask_file_names = [f for f in path.glob(f"{mode}/gts/*")]
        mask_file_names.sort()

        trn_mask_file_names, val_mask_file_names = cls.split_train_test(
            mask_file_names, 1 - config.split, seed=config.seed
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


@register_dataset("wrist")
class WristScans(BaseDataset):
    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

        # Define custom augmentations here
        self.trn_aug = K.augmentation.AugmentationSequential(
            # K.augmentation.RandomHorizontalFlip(p=0.2),
            # K.augmentation.RandomVerticalFlip(p=0.2),
            K.augmentation.RandomAffine(degrees=90, translate=(0.1, 0.1), p=0.5),
            data_keys=["input", "mask"],
        )

    def remap_labels(self, gt):
        return (gt > 0).long()


@register_dataset("hip")
class Hip(BaseDataset):
    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

    def remap_labels(self, gt):
        return (gt > 0).long()


@register_dataset("flare22")
class FLARE22(BaseDataset):
    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

        # Define custom augmentations here
        self.trn_aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomAffine(
                degrees=30, translate=(0.2, 0.2), shear=0, p=0.5, align_corners=False
            ),
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomVerticalFlip(p=0.5),
            data_keys=["input", "mask"],
            random_apply=1,
        )

        self.center_crop = CenterCrop(scale=(0.7, 1.0), p=0.5)

    @staticmethod
    def read_image(path):
        img = np.load(path)
        img = (img * 255).astype("uint8")
        return torch.from_numpy(img).permute(2, 0, 1)

    @staticmethod
    def read_mask(path):
        mask = torch.from_numpy(np.load(path))
        return mask


@register_dataset("flare22_test")
class FLARE22TestDataset(TestDataset):
    @staticmethod
    def read_image(path):
        img = np.load(path)
        img = (img * 255).astype("uint8")
        return torch.from_numpy(img).permute(2, 0, 1)

    @staticmethod
    def read_mask(path):
        mask = torch.from_numpy(np.load(path))
        return mask


@register_dataset("glas")
class WarwickGlassDataset(BaseDataset):
    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

        self.trn_aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomVerticalFlip(p=0.5),
            K.augmentation.RandomResizedCrop(
                size=(self.img_size, self.img_size),
                scale=(0.8, 1.0),
                p=0.5,
                align_corners=False,
            ),
            data_keys=["input", "mask"],
            random_apply=(2,),
        )

    @staticmethod
    def read_image(path):
        img = Image.open(path).convert("RGB")
        img = TF.to_tensor(img) * 255.0
        return img

    @staticmethod
    def read_mask(path):
        mask = Image.open(path).convert("L")
        mask = np.array(mask)
        mask = torch.from_numpy(mask).long()
        return mask

    def remap_labels(self, gt):
        return (gt > 0).long()


@register_dataset("glas_test_dataset")
class GlasTestDataset(TestDataset):
    @staticmethod
    def remap_labels(gt: torch.Tensor) -> torch.Tensor:
        return (gt > 0).long()

    @staticmethod
    def read_image(path):
        img = Image.open(path).convert("RGB")
        img = TF.to_tensor(img) * 255.0
        return img

    @staticmethod
    def read_mask(path):
        mask = Image.open(path).convert("L")
        mask = np.array(mask)
        mask = torch.from_numpy(mask).long()
        return mask


@register_dataset("acdc")
class ACDCDataset(BaseDataset):
    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

        self.trn_aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomAffine(
                degrees=30, translate=(0.2, 0.2), shear=0, p=0.5, align_corners=False
            ),
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomVerticalFlip(p=0.5),
            data_keys=["input", "mask"],
            random_apply=1,
        )

        self.center_crop = CenterCrop(scale=(0.6, 1.0), p=0.5)

    @staticmethod
    def read_image(path):
        img = io.read_image(path)
        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)
        return img

    @staticmethod
    def read_mask(path):
        mask = io.read_image(path)
        return mask


@register_dataset("acdc_test")
class ACDCTestDataset(TestDataset):
    @staticmethod
    def read_image(path):
        img = io.read_image(path)
        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)
        return img

    @staticmethod
    def read_mask(path):
        mask = io.read_image(path)
        return mask


@register_dataset("synapse")
class Synapse(BaseDataset):
    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

        self.trn_aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomVerticalFlip(p=0.5),
            K.augmentation.RandomAffine(degrees=30, translate=(0.2, 0.2), p=0.5),
            K.augmentation.RandomResizedCrop(
                size=(self.img_size, self.img_size),
                scale=(0.8, 1.0),
                p=0.5,
                align_corners=False,
            ),
            data_keys=["input", "mask"],
            random_apply=(2,),
        )

    @staticmethod
    def read_image(path):
        img = np.load(path)
        img = (img * 255).astype("uint8")
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=0)
        return torch.from_numpy(img)

    @staticmethod
    def read_mask(path):
        mask = torch.from_numpy(np.load(path))
        return mask


@register_dataset("us-nerve")
class USNerveSegmentation(BaseDataset):
    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

    @staticmethod
    def get_corresponding_image_name(mask_file: str):
        return mask_file.replace("_mask", "").replace("gts", "imgs")

    @staticmethod
    def read_image(path):
        img = Image.open(path).convert("RGB")
        img = TF.to_tensor(img) * 255.0

        return img

    @staticmethod
    def read_mask(path):
        mask = Image.open(path).convert("L")
        mask = TF.to_tensor(mask)
        return mask

    def remap_labels(self, gt):
        return (gt > 0).long()


@register_dataset("isic")
class ISICDataset(BaseDataset):
    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

    @staticmethod
    def get_corresponding_image_name(mask_file: str):
        # ISIC-2017_Validation_Part1_GroundTruth -> ISIC-2017_Validation_Data
        # ISIC-2017_Training_Part1_GroundTruth -> ISIC-2017_Training_Data

        if "val" in mask_file.lower():
            img_name = mask_file.replace(
                "ISIC-2017_Validation_Part1_GroundTruth", "ISIC-2017_Validation_Data"
            )
        else:
            img_name = mask_file.replace(
                "ISIC-2017_Training_Part1_GroundTruth", "ISIC-2017_Training_Data"
            )

        img_name = img_name.replace("_segmentation", "").replace(".png", ".jpg")
        return img_name

    @staticmethod
    def read_image(path):
        return io.read_image(path)

    @staticmethod
    def read_mask(path):
        mask = io.read_image(path)
        return mask

    @staticmethod
    def remap_labels(gt):
        return (gt > 0).long()

    @classmethod
    def from_path(cls, config):
        """
        Dataset object from a directory containing images and masks.
        """

        path = Path(config.root)
        mask_file_names = [
            f for f in path.glob("Train/ISIC-2017_Training_Part1_GroundTruth/*")
        ]
        mask_file_names.sort()

        if config.get("split", 1.0) < 1:
            trn_mask_file_names, val_mask_file_names = cls.split_train_test(
                mask_file_names, 1 - config.split, seed=config.seed
            )
        else:
            trn_mask_file_names = mask_file_names

        trn_img_file_names = [
            cls.get_corresponding_image_name(str(f)) for f in trn_mask_file_names
        ]

        val_mask_file_names = [
            f for f in path.glob("Val/ISIC-2017_Validation_Part1_GroundTruth/*")
        ]
        val_img_file_names = [
            cls.get_corresponding_image_name(str(f)) for f in val_mask_file_names
        ]

        return (
            cls(trn_img_file_names, trn_mask_file_names, config, mode="Train"),
            cls(val_img_file_names, val_mask_file_names, config, mode="Val"),
        )


@register_dataset("isic_test")
class ISICTestDataset(TestDataset):
    @staticmethod
    def get_corresponding_image_name(mask_file: str):
        return mask_file.replace(
            "ISIC-2017_Test_v2_Part1_GroundTruth", "ISIC-2017_Test_v2_Data"
        ).replace("_segmentation.png", ".jpg")

    @classmethod
    def from_path(cls, config, mode="Test"):
        """
        Dataset object from a directory containing images and masks.
        """

        path = Path(config.root)
        mask_file_names = [
            f for f in path.glob("Test/ISIC-2017_Test_v2_Part1_GroundTruth/*")
        ]
        mask_file_names.sort()

        img_file_names = [
            cls.get_corresponding_image_name(str(f)) for f in mask_file_names
        ]

        return cls(img_file_names, mask_file_names, config)

    @staticmethod
    def read_image(path):
        return io.read_image(path)

    @staticmethod
    def read_mask(path):
        mask = io.read_image(path)
        return mask

    @staticmethod
    def remap_labels(gt):
        return (gt > 0).long()


@register_dataset("3dus_chop")
class Hip3DChopDataset(BaseDataset):
    # Map RGB values to class IDs
    label_to_class_id = {
        (255, 0, 0): 1,  # Red -> Class 0
        (0, 255, 0): 2,  # Green -> Class 1
        (0, 0, 255): 3,  # Blue -> Class 2
    }

    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

        self.trn_aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomVerticalFlip(p=0.5),
            K.augmentation.RandomAffine(degrees=30, translate=(0.1, 0.1), p=0.5),
            data_keys=["input", "mask"],
        )
        self.center_crop = CenterCrop(scale=(0.7, 1.0), p=0.5)

    @staticmethod
    def remap_labels(gt: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        Remap RGB labels to class IDs.
        Assumes `gt` is a tensor of shape (3, H, W) or (
        H, W, 3) where each pixel is an RGB tuple.
        """
        # If the input is in shape (3, H, W), we need to permute it to (H, W, 3)
        if gt.shape[0] == 3:
            gt = gt.permute(1, 2, 0)  # Change from (3, H, W) to (H, W, 3)

        # Now proceed with the original logic
        class_map = torch.zeros(gt.shape[:2], dtype=torch.long)
        for color, class_id in Hip3DChopDataset.label_to_class_id.items():
            match = (gt == torch.tensor(color, dtype=gt.dtype)).all(dim=-1)
            class_map[match] = class_id
        return class_map


@register_dataset("3dus_chop_test")
class ChopTestDataset(Hip3DChopDataset):
    """
    Returns images without any augmentation.
    """

    def __getitem__(self, index):
        img = io.read_image(self.img_files[index]).float() / 255.0
        gt = io.read_image(self.gt_files[index])
        gt = self.remap_labels(gt).long()
        img = (img - self.mean) / self.std
        img = self.resize(img[None], order=1)
        input_size = img.shape[2:]
        img = self.pad(img)[0]

        return img, gt, input_size, self.img_files[index]


@register_dataset("jsrt")
class JSRTXRay(BaseDataset):
    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

        self.trn_aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomVerticalFlip(p=0.5),
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomAffine(degrees=30, translate=(0.1, 0.1), p=0.5),
            random_apply=(2,),
            data_keys=["input", "mask"],
        )

        self.center_crop = CenterCrop(scale=(0.8, 1.0), p=0.6)

    @staticmethod
    def remap_labels(mask):
        return (mask > 0).long()


@register_dataset("jsrt_test_dataset")
class JSRTXRayTestDataset(TestDataset):
    @staticmethod
    def remap_labels(gt: torch.Tensor) -> torch.Tensor:
        return (gt > 0).long()


@register_dataset("rectal_cancer_mri")
class RectalCancerMRIDataset(BaseDataset):
    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

        self.trn_aug = K.augmentation.AugmentationSequential(
            K.augmentation.RandomHorizontalFlip(p=0.5),
            K.augmentation.RandomVerticalFlip(p=0.5),
            K.augmentation.RandomRotation(degrees=180, p=0.5),
            # K.augmentation.RandomResizedCrop(
            #     size=(self.img_size, self.img_size),
            #     scale=(0.6, 1.0),
            #     p=0.8,
            #     align_corners=False,
            # ),
            data_keys=["input", "mask"],
            random_apply=(2,),
        )

        self.center_crop = CenterCrop(scale=(0.6, 1.0), p=0.6)

    def remap_labels(self, image_data: torch.Tensor):
        """
        Extract masks from overlaid image data.

        Args:
            image_data (torch.Tensor): Tensor of shape (3, H, W) representing the image data.

        Returns:
            torch.Tensor: Byte tensor of shape (H, W) representing the masks.
        """
        # Color encoding:  (240,164,12), (18, 31, 230), (209, 31, 240)
        red_mask_color = torch.tensor([240, 164, 12]).view(3, 1, 1)
        green_mask_color = torch.tensor([(209, 31, 240)]).view(3, 1, 1)
        blue_mask_color = torch.tensor([(18, 31, 230)]).view(3, 1, 1)

        # Define a tolerance for each color channel
        color_tolerance = 30

        # Create masks for each color
        r = torch.all(torch.abs(image_data - red_mask_color) < color_tolerance, dim=0)
        g = torch.all(torch.abs(image_data - green_mask_color) < color_tolerance, dim=0)
        b = torch.all(torch.abs(image_data - blue_mask_color) < color_tolerance, dim=0)
        mask = torch.zeros(r.shape)
        mask[r > 0] = 1
        mask[g > 0] = 2
        mask[b > 0] = 3

        if hasattr(self, "label_to_class_id") and self.label_to_class_id is not None:
            for k, v in self.label_to_class_id.items():
                mask[mask == k] = v

        return mask.byte()

    @staticmethod
    def read_image(path):
        image = io.read_image(path)
        if image.size(0) == 4:
            image = image[:3]
        return image

    @staticmethod
    def read_mask(path):
        mask = io.read_image(path)
        if mask.size(0) == 4:
            mask = mask[:3]
        return mask

    @staticmethod
    def get_corresponding_image_name(mask_file: str):
        return mask_file.replace("0.png", "1.png")

    @classmethod
    def from_path(cls, config):
        """
        Dataset object from a directory containing images and masks
        """

        path = Path(config.root)

        # recursively search for all images and the patient ID (folder name)
        mask_file_names = list((path / "Finished Data - Nolan").glob("**/*0.png"))

        # mask_file_names.extend(list((path / "Finished Data - Susie").glob("**/*0.png")))

        # mask_file_names = list((path / "Finished Data - Susie").glob("**/*0.png"))
        mask_file_names.sort()

        # Subfolders are patient IDs (e.g., Finished Data - Nolan/AHS2055168/ patient ID is AHS2055168)
        patient_ids = [
            [part for part in f.parts if part != "/"][5] for f in mask_file_names
        ]

        # Map patient IDs to integers
        patient_id_map = {pid: i for i, pid in enumerate(sorted(set(patient_ids)))}
        patient_ids = [patient_id_map[pid] for pid in patient_ids]

        # Train/test split without overlapping patients
        # Create GroupShuffleSplit object
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=1 - config.split,
            random_state=config.get("seed", 42),
        )

        # Split the dataset
        train_idx, test_idx = next(gss.split(mask_file_names, groups=patient_ids))

        # Create training and testing sets
        trn_mask_file_names = [mask_file_names[i] for i in train_idx]
        val_mask_file_names = [mask_file_names[i] for i in test_idx]

        # Get the corresponding image file names
        trn_img_file_names = [
            cls.get_corresponding_image_name(str(f)) for f in trn_mask_file_names
        ]

        val_img_file_names = [
            cls.get_corresponding_image_name(str(f)) for f in val_mask_file_names
        ]

        with open("logs/Train_Images.txt", "w") as f:
            for item in trn_img_file_names:
                f.write("%s\n" % item)

        with open("logs/Val_Images.txt", "w") as f:
            for item in val_img_file_names:
                f.write("%s\n" % item)

        return (
            cls(trn_img_file_names, trn_mask_file_names, config, mode="Train"),
            cls(val_img_file_names, val_mask_file_names, config, mode="Val"),
        )


@register_dataset("rectal_cancer_mri_test")
class RectalCancerMRITestDataset(RectalCancerMRIDataset):
    def __init__(self, img_files, gt_files, config, mode):
        super().__init__(img_files, gt_files, config, mode)

        self.center_crop = CenterCrop(scale=(0.5, 0.5), p=1.0)

    @classmethod
    def from_path(cls, config, mode="Test"):
        """
        Dataset object from a directory containing images and masks.
        """
        print(config.root)
        txt_path = Path(config.root)
        with open(txt_path, "r") as f:
            tst_img_files = f.readlines()
            tst_img_files = [x.strip() for x in tst_img_files]

        # corresponding masks
        tst_mask_files = [f.replace("1.png", "0.png") for f in tst_img_files]

        return cls(tst_img_files, tst_mask_files, config, mode=mode)

    def __getitem__(self, index):
        img = self.read_image(self.img_files[index])
        gt = self.read_mask(self.gt_files[index])
        gt = self.remap_labels(gt).long()
        img, gt = self.center_crop(img[None], gt[None, None])
        img, gt = img[0], gt[0, 0]
        img_orig = img

        img = img.float() / 255.0
        img = (img - self.mean) / self.std
        img = self.resize(img[None], order=1)
        input_size = img.shape[2:]
        img = self.pad(img)[0]

        if gt.ndim == 3:
            gt = gt.squeeze(0)

        return img_orig.permute(1, 2, 0), img, gt, input_size, self.img_files[index]
