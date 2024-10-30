from torchvision.ops import nms


def compute_f1(pr: float, rc: float) -> float:
    return 2 * (pr * rc) / (pr + rc + 1e-10)


def postporc(pred: dict, conf_thr: float = 0.9, iou_thr: float = 0.5):
    keep_indices = pred["scores"] > conf_thr
    pred["boxes"] = pred["boxes"][keep_indices]
    pred["labels"] = pred["labels"][keep_indices]
    pred["scores"] = pred["scores"][keep_indices]
    pred["masks"] = pred["masks"][keep_indices]

    keep_indices = nms(
        boxes=pred["boxes"], scores=pred["scores"], iou_threshold=iou_thr
    )
    pred["boxes"] = pred["boxes"][keep_indices]
    pred["labels"] = pred["labels"][keep_indices]
    pred["scores"] = pred["scores"][keep_indices]
    pred["masks"] = pred["masks"][keep_indices]
