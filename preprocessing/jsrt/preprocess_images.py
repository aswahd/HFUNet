# Code adapted from https://github.com/ngaggion/Chest-xray-landmark-dataset/blob/main/Preprocess-JSRT.ipynb
import os
import pathlib
import re
from tqdm import tqdm
import cv2
import numpy as np


def preprocess(folderpath, flist):
    if not os.path.exists(folderpath):
        os.mkdir(folderpath)

    for f in tqdm(flist):
        w, h = 2048, 2048

        with open(f, "rb") as path:
            dtype = np.dtype(">u2")
            img = np.fromfile(path, dtype=dtype).reshape((h, w))

        img = 1 - img.astype("float") / 4096
        img = cv2.resize(img, (1024, 1024))
        img = img * 255

        file_name = os.path.basename(f).replace(".IMG", ".png")
        p = os.path.join(folderpath, file_name)
        cv2.imwrite(p, img.astype("uint8"))
     

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_)]


img_path = "All247images"

data_root = pathlib.Path(img_path)
all_files = list(data_root.glob("*.IMG"))
all_files = [str(path) for path in all_files]
all_files.sort(key=natural_key)

save_path = "JSRTPreprocessed/imgs"
preprocess(save_path, all_files)
