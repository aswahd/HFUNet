from pathlib import Path
from sklearn.model_selection import train_test_split

# Define the root directory and subdirectories
root = Path("JSRTPreprocessed")
imgs_dir = root / "imgs"
gts_dir = root / "gts"

# Get a list of all image files
img_paths = sorted(imgs_dir.iterdir())

# Split the image paths into training and testing sets
train_imgs, test_imgs = train_test_split(img_paths, test_size=0.4, random_state=42)

# Define the training and testing directories
train_root = root / "Train"
test_root = root / "Test"

# Create directories for training and testing datasets
(train_root / "imgs").mkdir(parents=True, exist_ok=True)
(train_root / "gts").mkdir(parents=True, exist_ok=True)
(test_root / "imgs").mkdir(parents=True, exist_ok=True)
(test_root / "gts").mkdir(parents=True, exist_ok=True)

# Move training images and ground truth files
for img_path in train_imgs:
    img_name = img_path.name
    gt_path = gts_dir / img_name

    # Destination paths
    img_dest = train_root / "imgs" / img_name
    gt_dest = train_root / "gts" / img_name

    # Move image and ground truth
    img_path.rename(img_dest)
    gt_path.rename(gt_dest)

# Move testing images and ground truth files
for img_path in test_imgs:
    img_name = img_path.name
    gt_path = gts_dir / img_name

    # Destination paths
    img_dest = test_root / "imgs" / img_name
    gt_dest = test_root / "gts" / img_name

    # Move image and ground truth
    img_path.rename(img_dest)
    gt_path.rename(gt_dest)

print("Done")