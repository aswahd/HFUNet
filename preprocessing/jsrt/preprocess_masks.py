import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def preprocess(folderpath, flist):
    if not os.path.exists(folderpath):
        os.mkdir(folderpath)
    for f in tqdm(flist):
        img = Image.open(f).convert("L").resize((1024, 1024))
        file_name = os.path.basename(f).replace(".tif", ".png")
        p = os.path.join(folderpath, file_name)
        img.save(p)


img_path = "Masks"

data_root = Path(img_path)
all_files = list(data_root.glob("JPC*.tif"))
all_files = [str(path) for path in all_files]
print(f"Found {len(all_files)} files")

save_path = "JSRTPreprocessed/gts"
preprocess(save_path, all_files)
