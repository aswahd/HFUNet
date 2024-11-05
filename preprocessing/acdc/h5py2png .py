# h5py to .png
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

path = Path(
    "/home/wahd/datasets/ACDCPreprocessed/Train/data_2D_size_212_212_res_1.36719_1.36719_onlytrain.hdf5"
)
dst = Path("/home/wahd/datasets/ACDCPreprocessed/Train/")

if not dst.exists():
    (dst / "imgs").mkdir(parents=True, exist_ok=True)
    (dst / "gts").mkdir(parents=True, exist_ok=True)

with h5py.File(path, "r") as f:
    # Print keys
    print(f"Keys: {list(f.keys())}")
    print(f["images_train"].shape)
    print(f["masks_train"].shape)

    for i, (image, mask) in tqdm(enumerate(zip(f["images_train"], f["masks_train"]))):
        image = (image - image.min()) / (image.max() - image.min()) * 255
        image = image.astype(np.uint8)

        # Save image
        img = Image.fromarray(image)
        img.save(dst / "imgs" / f"{i}.png")

        # Save mask
        img = Image.fromarray(mask)
        img.save(dst / "gts" / f"{i}.png")
