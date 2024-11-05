import random
from functools import lru_cache

import cv2
import numpy as np
import torch


@lru_cache(maxsize=None)
def get_random_color(cls: int):
    return tuple(random.randint(0, 255) for _ in range(3))


def overlay_contours(img: np.ndarray, mask: np.array, thickness=2):
    unique_classes = np.unique(mask)

    for cls in unique_classes:
        if cls == 0:  # Assuming 0 is the background class
            continue

        binary_mask = (mask == cls).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        color = get_random_color(cls)
        cv2.drawContours(img, contours, -1, color, thickness)

    return img.astype(np.uint8)


def colorize_mask(mask: np.array, colors=None, num_classes=None):
    """
    Converts a semantic mask to a colorized mask.

    Args:
        mask (np.ndarray): A 2D array of shape (H, W) where each value represents a class label.
        colors (dict): A dictionary mapping class labels (int) to RGB color tuples.
                      Example: {1: (255, 0, 0), 2: (0, 255, 0)}
        num_classes (int): The number of classes (including background).

    Returns:
        np.ndarray: A colorized mask of shape (H, W, 3) where each channel represents an RGB component.
    """
    assert colors or num_classes, "Either colors or num_classes must be provided."
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)  # (H, W, 3) for RGB

    if colors is None:
        # Generate random colors for each class (excluding background class 0)
        colors = {
            class_id: get_random_color(class_id) for class_id in range(0, num_classes)
        }

    # Assign colors to each class
    for class_id, color in colors.items():
        color_mask[mask == class_id] = color  # Assign RGB color to pixels of this class

    return color_mask


def alpha_blend(img: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Alpha blends an image with a mask.

    Parameters:
        img (np.ndarray): Input RGB image of shape (H, W, 3).
        mask (np.ndarray): Colored mask of shape (H, W, 3).
        alpha (float, optional): Alpha blending factor. Default is 0.5.

    Returns:
        np.ndarray: Blended image of shape (H, W, 3).
    """
    # Ensure the images are float type for blending
    img = img.astype(np.float32)
    mask = mask.astype(np.float32)

    # Create a binary mask where mask is not zero
    binary_mask = np.any(mask > 0, axis=-1).astype(np.float32)  # Shape (H, W)

    # Expand binary_mask to have same shape as img and mask
    binary_mask = np.expand_dims(binary_mask, axis=-1)  # Shape (H, W, 1)

    # Compute the alpha values for blending
    alpha_mask = (
        binary_mask * alpha
    )  # Foreground pixels have alpha value; background is zero

    # Alpha blending
    blended = img * (1 - alpha_mask) + mask * alpha_mask

    # Convert back to uint8 and clip values to valid range [0, 255]
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended


def convert_to_semantic(mask: torch.Tensor):
    """
    Convert a multi-channel mask to a semantic segmentation mask.

    Args:
    mask: (B, C, H, W) - A PyTorch tensor where B is batch size, C is number of classes,
                        H is height, and W is width.

    Returns:
    sem_mask: (B, H, W) - A PyTorch tensor representing the semantic segmentation mask.
    """

    # Get the class with the highest probability for each pixel (adding 1 to account for ignored background)
    sem_mask = mask.argmax(dim=1) + 1

    # Create a foreground mask
    fg = (mask > 0).any(dim=1).float()

    # Apply the foreground mask to sem_mask
    sem_mask = sem_mask * fg

    return sem_mask.long()


class DotDict(dict):
    """A dictionary that supports dot notation."""

    def __getattr__(self, attr):
        if attr in self:
            value = self[attr]
            if isinstance(value, dict) and not isinstance(value, DotDict):
                value = DotDict(value)
                self[attr] = value  # Store back to ensure the same object is used
            return value
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{attr}'"
        )

    def __setattr__(self, attr, value):
        self[attr] = value

    def __delattr__(self, attr):
        del self[attr]

    @staticmethod
    def from_dict(data):
        """Recursively converts a dictionary to DotDict."""
        if not isinstance(data, dict):
            return data
        return DotDict({k: DotDict.from_dict(v) for k, v in data.items()})



def add_salt_and_pepper_noise(image, prob=0.05):
    """
    Adds salt and pepper noise to a 3-channel image tensor.

    Parameters:
    - image (torch.Tensor): Input image tensor of shape (3, H, W) with values in [0, 1].
    - prob (float): Probability of an element to be flipped (noise level).

    Returns:
    - torch.Tensor: Noisy image tensor of shape (3, H, W).
    """
    # Clone the image to avoid modifying the original
    noisy_image = image.clone()

    # Generate a random tensor for each pixel
    random_tensor = torch.rand_like(image)

    # Apply salt noise (set pixel to 1)
    noisy_image[random_tensor < (prob / 2)] = 1.0

    # Apply pepper noise (set pixel to 0)
    noisy_image[random_tensor > 1 - (prob / 2)] = 0.0

    return noisy_image