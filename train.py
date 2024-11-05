import argparse
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import termcolor
import torch
import torch.nn.functional as F
import yaml
from monai.metrics import DiceMetric, MeanIoU
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

from datasets import DATASETS, get_dataloaders
from models import build_model
from utils import DotDict, overlay_contours

torch.set_float32_matmul_precision("high")


class SavePredictionsCallback(pl.Callback):
    def __init__(self):
        self.val_outputs = []
        self.train_outputs = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx < 1:
            self.train_outputs.append(outputs)

        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx < 5:
            self.val_outputs.append(outputs)

        return super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        outputs = self.val_outputs
        pred = torch.cat([o["pred"] for o in outputs], dim=0)
        gt = torch.cat([o["gt"] for o in outputs], dim=0)
        images = torch.cat([o["images"] for o in outputs], dim=0)
        fig, axes = plt.subplots(pred.size(0), 4, figsize=(4 * 4, pred.size(0) * 4))
        if pred.size(0) == 1:
            axes = axes[None, ...]

        for i, (p, g, img) in enumerate(zip(pred, gt, images)):
            img_gt = overlay_contours(
                img.cpu().numpy().astype(np.uint8), g.cpu().numpy().astype(np.uint8)
            )
            img_pred = overlay_contours(
                img.cpu().numpy().astype(np.uint8), p.cpu().numpy().astype(np.uint8)
            )

            axes[i, 0].imshow(img_gt)
            axes[i, 1].imshow(img_pred)
            axes[i, 2].imshow(g.cpu())
            axes[i, 3].imshow(p.cpu())

        plt.savefig("val_progress.png", bbox_inches="tight")
        plt.close()

        self.val_outputs.clear()

        return super().on_validation_epoch_end(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        outputs = self.train_outputs
        if len(outputs) > 0:
            pred = torch.cat([o["pred"] for o in outputs], dim=0)
            gt = torch.cat([o["gt"] for o in outputs], dim=0)
            images = torch.cat([o["images"] for o in outputs], dim=0)

            fig, axes = plt.subplots(pred.size(0), 4, figsize=(4 * 4, pred.size(0) * 4))
            if pred.size(0) == 1:
                axes = axes[None, ...]

            for i, (p, g, img) in enumerate(zip(pred, gt, images)):
                img_gt = overlay_contours(
                    img.cpu().numpy().astype(np.uint8), g.cpu().numpy().astype(np.uint8)
                )
                img_pred = overlay_contours(
                    img.cpu().numpy().astype(np.uint8), p.cpu().numpy().astype(np.uint8)
                )

                axes[i, 0].imshow(img_gt)
                axes[i, 1].imshow(img_pred)
                axes[i, 2].imshow(g.cpu())
                axes[i, 3].imshow(p.cpu())

            plt.savefig("train_progress.png", bbox_inches="tight")
            plt.close()
            self.train_outputs.clear()

        return super().on_train_epoch_end(trainer, pl_module)


class Learner(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float,
        pixel_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.num_classes = model.num_classes
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.pixel_mean = torch.tensor(pixel_mean).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(pixel_std).view(1, 3, 1, 1)

        self.train_dice_metric = DiceMetric(
            include_background=False,
            get_not_nans=False,
            ignore_empty=True,
        )
        self.train_iou_metric = MeanIoU(
            include_background=False,
            get_not_nans=False,
            ignore_empty=True,
        )
        self.val_dice_metric = DiceMetric(
            include_background=False,
            get_not_nans=False,
            ignore_empty=True,
        )
        self.val_iou_metric = MeanIoU(
            include_background=False,
            get_not_nans=False,
            ignore_empty=True,
        )

        # save all the arguments passed to the __init__ method
        self.save_hyperparameters(ignore=["model"])

    @staticmethod
    def reshape_inputs(batch):
        batch["boxes"] = batch["boxes"].reshape(-1, 4)
        batch["boxes_normalized"] = batch["boxes_normalized"].reshape(-1, 4)
        batch["ignore"] = batch["ignore"].reshape(-1)
        lr_masks = batch["low_res_masks"]
        batch["low_res_masks"] = lr_masks.reshape(
            -1, 1, lr_masks.size(2), lr_masks.size(3)
        )
        masks = batch["masks"]
        batch["masks"] = masks.reshape(-1, 1, masks.size(2), masks.size(3))

        return batch

    def loss_from_masks_list(self, preds: List[torch.Tensor], gt: torch.Tensor):
        losses = 0.0
        for p in preds:
            *_, h, w = p.shape
            _gt = (
                F.interpolate(gt.unsqueeze(1), size=(h, w), mode="nearest")
                .squeeze(1)
                .long()
            )
            losses += self.loss_fn(p, _gt)
        return losses

    def one_hot(self, masks):
        return (
            F.one_hot(masks.long(), num_classes=self.model.num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        gt = batch["masks"]  # (B, H, W)
        preds = self.model(images)
        loss = self.loss_from_masks_list(preds, gt)
        # The prediction from last decoder layer is used for evaluation
        pred = preds[-1].argmax(1)
        gt_onehot = self.one_hot(gt)
        pred_onehot = self.one_hot(pred)

        self.train_dice_metric(pred_onehot, gt_onehot)
        self.train_iou_metric(pred_onehot, gt_onehot)

        self.log_dict(
            {
                "train_loss": loss,
                "train_dice": self.train_dice_metric.aggregate(reduction="none")
                .nanmean(0)
                .nanmean(0),
                "train_iou": self.train_iou_metric.aggregate(reduction="none")
                .nanmean(0)
                .nanmean(0),
            },
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

        return {
            "loss": loss,
            "pred": pred,
            "gt": gt,
            "images": self.denormalize(images),
        }

    def denormalize(self, img):
        img = img * self.pixel_std.to(img.device) + self.pixel_mean.to(img.device)
        return (img * 255).type(torch.uint8).permute(0, 2, 3, 1).cpu()

    def on_validation_epoch_end(self) -> None:
        self.val_dice_metric.reset()
        self.val_iou_metric.reset()
        return super().on_validation_epoch_end()

    def on_train_epoch_end(self) -> None:
        self.train_dice_metric.reset()
        self.train_iou_metric.reset()
        return super().on_train_epoch_end()

    def resize_ground_truth_to_predictions(
        self, preds: List[torch.Tensor], gt: torch.Tensor
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        resized_preds: List[torch.Tensor] = []
        *_, h, w = gt.shape
        if isinstance(preds, torch.Tensor):
            return F.interpolate(
                preds, size=(h, w), mode="bilinear", align_corners=False
            )

        for p in preds:
            resized_preds.append(
                F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)
            )

        return resized_preds

    def validation_step(self, batch, batch_idx):
        images = batch["images"]
        gt = batch["masks"]
        preds = self.model(images)
        loss = self.loss_from_masks_list(preds, gt.float())
        # The prediction from last decoder layer is used for evaluation
        pred = preds[-1].argmax(1)
        gt_onehot = self.one_hot(gt)
        pred_onehot = self.one_hot(pred)

        self.val_dice_metric(pred_onehot, gt_onehot)
        self.val_iou_metric(pred_onehot, gt_onehot)

        self.log_dict(
            {
                "val_loss": loss,
                "val_iou": self.val_iou_metric.aggregate(reduction="none")
                .nanmean(0)
                .nanmean(0),
                "val_dice": self.val_dice_metric.aggregate(reduction="none")
                .nanmean(0)
                .nanmean(0),
            },
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

        return {
            "loss": loss,
            "pred": pred,
            "gt": gt,
            "images": self.denormalize(images),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=5e-1,
        )
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.max_epochs, eta_min=1e-5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "train_dice",
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mobile_sam_config.yaml")
    args = parser.parse_args()
    # Get dataset object
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
        config = DotDict(config)

    # Register a custom dataset or use a default one, e.g., dataset_obj = DATASETS["default_segmentation"]
    dataset_obj = DATASETS[config.dataset.name]

    trn_ds, val_ds = dataset_obj.from_path(config.dataset)
    # Debug: faster validation
    val_ds = torch.utils.data.Subset(val_ds, range(0, 50))

    trn_dl = get_dataloaders(config.dataset, trn_ds)
    val_dl = get_dataloaders(config.dataset, val_ds)

    print(f"Train dataset size: {len(trn_dl.dataset)}")
    print(f"Validation dataset size: {len(val_dl.dataset)}")

    model = build_model(config)
    print(model)

    termcolor.colored("Trainable parameters:", "red")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(termcolor.colored(f"{name} | {param.size()}", "red"))

    learner = Learner(model, lr=1e-4)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    wandb_logger = WandbLogger(
        project=config.get("wandb_project_name", "hfunet"),
        log_model=False,
        save_dir="./logs",
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_dice",
        dirpath="checkpoints"
        if config.get("save_path") is None
        else config.get("save_path"),
        save_last=True,
        filename="model_{epoch:02d}-{val_dice:.2f}",
        save_top_k=3,
        mode="max",
    )

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        enable_progress_bar=True,
        check_val_every_n_epoch=10,
        log_every_n_steps=10,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            lr_monitor,
            TQDMProgressBar(leave=True),
            SavePredictionsCallback(),
        ],
        accelerator="gpu",  # run on all available GPUs
    )

    trainer.fit(
        learner,
        train_dataloaders=trn_dl,
        val_dataloaders=val_dl,
        ckpt_path=config.get("resume"),
    )
