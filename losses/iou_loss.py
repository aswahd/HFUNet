import torch


def iou_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    smooth: float = 1e-5,
    reduction="mean",
) -> torch.Tensor:
    y_pred = y_pred.sigmoid()
    intersection = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    union = (
        torch.sum(y_true, dim=(1, 2, 3))
        + torch.sum(y_pred, dim=(1, 2, 3))
        - intersection
        + smooth
    )

    iou = (intersection) / (union + smooth)

    if reduction == "mean":
        return 1 - torch.mean(iou)
    elif reduction == "sum":
        return 1 - torch.sum(iou)
    else:
        return 1 - iou
