# fmt: off
import os
import sys

sys.path.insert(0, os.path.abspath("segmentation_models.pytorch"))
# fmt: on

import argparse
import logging
import os
import random
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from monai.metrics import DiceMetric, MeanIoU
from tqdm import tqdm

from datasets import DATASETS
from models import build_model
from utils import DotDict, overlay_contours

torch.set_float32_matmul_precision("high")

DEBUG = False
# DEBUG = True
if DEBUG:
    random.seed(42)


class Eval:
    def __init__(self, model, input_size: int):
        self.model = model
        self.input_size = (input_size, input_size)
        self.device = next(self.model.parameters()).device
        self.num_classes = self.model.num_classes
        self.dice_metric = DiceMetric(
            include_background=False,
            reduction="mean_batch",
            get_not_nans=False,
            ignore_empty=True,
        )
        self.iou_metric = MeanIoU(
            include_background=False,
            reduction="mean_batch",
            get_not_nans=False,
            ignore_empty=True,
        )

        self.reset()

        self.pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def one_hot(self, masks):
        return (
            F.one_hot(masks.long(), num_classes=self.model.num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )

    def reset(self):
        self.dice_metric.reset()
        self.iou_metric.reset()

    def post_process(self, pred, input_size, original_size):
        # Resize to model input size
        pred = F.interpolate(
            pred, self.input_size, mode="bilinear", align_corners=False
        )
        # Remove padding
        pred = pred[:, :, : input_size[0], : input_size[1]]
        return F.interpolate(pred, original_size, mode="bilinear", align_corners=False)

    def eval_batch(self, batch):
        images_orig, images, gt, input_size, filename = batch
        (
            images,
            gt,
        ) = (images.to(self.device), gt.to(self.device))

        bs, h, w = gt.shape
        pred = self.model(images)
        # The predicted mask at the last decoder layer (if deep supervision is used)
        pred = pred[-1]  # (bs, C, H, W)
        original_size = gt.shape[1:]
        pred = self.post_process(pred, input_size, original_size)
        pred = pred.argmax(1)

        if DEBUG:
            # Image
            plt.subplot(1, 4, 1)
            # img_gt = overlay_contours(
            #     images_orig[0].clone().numpy(), gt[0].cpu().numpy()
            # )
            img_gt = images_orig[0].clone().numpy()
            img_pred = overlay_contours(
                images_orig[0].clone().numpy(), pred[0].cpu().numpy()
            )
            plt.imshow(img_gt)
            # plt.gca().axis("off")
            plt.subplot(1, 4, 2)
            plt.imshow(img_pred)
            # plt.gca().axis("off")
            plt.subplot(1, 4, 3)
            plt.imshow(gt[0].cpu())
            # plt.gca().axis("off")
            plt.subplot(1, 4, 4)
            plt.imshow(pred[0].cpu())
            plt.gcf().set_size_inches(20, 5)
            # plt.gca().axis("off")
            # set hs, ws to zero
            plt.subplots_adjust(hspace=0, wspace=0.01)
            plt.savefig("debug_test.png", bbox_inches="tight")
            plt.close()
            import pdb

            pdb.set_trace()

        # Calculate dice and iou scores
        gt_onehot = self.one_hot(gt)
        pred_onehot = self.one_hot(pred)
        self.dice_metric(pred_onehot, gt_onehot)
        self.iou_metric(gt_onehot, pred_onehot)

    @torch.no_grad()
    def eval(self, dataloader):
        self.reset()
        with tqdm(dataloader, total=len(dataloader)) as pbar:
            for batch in dataloader:
                self.eval_batch(batch)
                pbar.set_description(
                    f"Dice: {self.dice_metric.aggregate(reduction='none').nanmean(0).nanmean(0).item():.4f}, IoU: {self.iou_metric.aggregate(reduction='none').nanmean(0).nanmean(0).item():.4f}"
                )
                pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="test_config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)]
    )

    with open(args.config, encoding="utf-8") as f:
        config = DotDict.from_dict(yaml.safe_load(f))

    ds = DATASETS.get(
        config.test_config.name, DATASETS["default_test_dataset"]
    ).from_path(config.dataset, mode="Test")

    tst_dl = torch.utils.data.DataLoader(
        ds,
        batch_size=1,  # Images can be of different size
        num_workers=config.get("num_workers", 1),
        pin_memory=True,
        shuffle=False,
    )

    print(f"Test dataset size: {len(tst_dl.dataset)}")

    model = build_model(config)
    print(model)
    # Load checkpoint
    checkpoint = torch.load(config.test_config.checkpoint_path, map_location="cpu")
    epoch = checkpoint["epoch"]
    checkpoint = checkpoint["state_dict"]
    checkpoint = {k[len("model.") :]: v for k, v in checkpoint.items()}
    print(model.load_state_dict(checkpoint))
    model = model.to("cuda:0")
    model.eval()
    print(f"Checkpoint loaded from {config.test_config.checkpoint_path}, epoch {epoch}")

    evaluator = Eval(model, config.test_config.image_size)
    evaluator.eval(tst_dl)

    # Prepare the output directory
    output_dir = os.path.join(
        os.path.dirname(config.test_config.checkpoint_path),
        "eval_results",
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "eval_results.txt")

    # Collect the per-class Dice and IoU scores
    dice_scores = evaluator.dice_metric.aggregate("none").nanmean(0)
    iou_scores = evaluator.iou_metric.aggregate("none").nanmean(0)

    # Collect the average Dice and IoU scores
    avg_dice = dice_scores.nanmean()
    avg_iou = iou_scores.nanmean()

    # Prepare the results as a formatted string
    results_str = "=== Evaluation Results ===\n\n"
    results_str += "Configuration:\n"
    results_str += "\nPer-Class Metrics:\n"

    # Format the per-class metrics
    results_str += "{:<10} {:>10} {:>10}\n".format("Class", "Dice", "IoU")
    results_str += "-" * 32 + "\n"
    for i, (dice_score, iou_score) in enumerate(zip(dice_scores, iou_scores)):
        results_str += "{:<10} {:>10.4f} {:>10.4f}\n".format(
            f"Class {i}", dice_score.item(), iou_score.item()
        )

    # Add average metrics
    results_str += "\nAverage Metrics:\n"
    results_str += "-" * 32 + "\n"
    results_str += "{:<10} {:>10.4f} {:>10.4f}\n".format("Average", avg_dice, avg_iou)

    print(results_str)

    results_str = yaml.dump(config) + results_str

    # Topk images with the lowest Dice score
    dice_scores_per_img = evaluator.dice_metric.aggregate("none").nanmean(dim=1)
    values, indices = torch.topk(dice_scores_per_img, k=5, largest=False)
    print("Top 5 images with the least Dice score:")
    results_str += "\nTop 5 images with the least Dice score:\n"
    for dice, idx in zip(values, indices):
        print(f"{tst_dl.dataset.img_files[idx.item()]}, Dice: {dice.item()}")
        results_str += f"{tst_dl.dataset.img_files[idx.item()]}, Dice: {dice.item()}\n"

    # Write the results to the output file
    with open(output_file, encoding="utf-8", mode="w") as f:
        f.write(results_str)
        f.write("\n")

    logging.info("Results saved to %s", output_file)
