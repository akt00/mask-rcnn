import datetime
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import yaml

from src import postporc
from src.dataset import COCODataset, collate_fn
from src.engine import evaluate
from src.models import get_mask_rcnn


def eval(cfg: dict):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    val_dataset = COCODataset(
        image_path=Path(cfg["val"]),
        annotation_path=Path(cfg["val_annotation"]),
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg["batch"],
        shuffle=False,
        num_workers=cfg["workers"],
        collate_fn=collate_fn,
        persistent_workers=False,
    )

    model = get_mask_rcnn(num_classes=cfg["nc"], coco=cfg["coco"])
    ckpt = torch.load(cfg["weight"], weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device=device)

    evaluate(model=model, data_loader=val_loader, device=device)


def visualize(cfg: dict):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    now = datetime.datetime.now()
    time_str = now.strftime("%Y%m%d%H%M%S")

    res_dir = Path("results")

    if not res_dir.exists():
        os.makedirs("results")

    os.makedirs(res_dir / time_str)

    val_dataset = COCODataset(
        image_path=Path(cfg["val"]),
        annotation_path=Path(cfg["val_annotation"]),
    )

    model = get_mask_rcnn(num_classes=cfg["nc"], coco=cfg["coco"])
    ckpt = torch.load(cfg["weight"], weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device=device)
    model.eval()

    with torch.no_grad():
        for idx, (image, _) in enumerate(val_dataset):
            x = image.to(device=device)
            preds = model([x])
            pred = preds[0]
            # in-place
            postporc(pred=pred, conf_thr=cfg["conf_thr"], iou_thr=cfg["iou_thr"])
            # restore the original image
            image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(
                torch.uint8
            )
            image = image[:3, ...]
            # bboxes
            pred_labels = [
                f"{label}" for label, score in zip(pred["labels"], pred["scores"])
            ]
            pred_boxes = pred["boxes"].long()
            output_image = draw_bounding_boxes(
                image, pred_boxes, pred_labels, colors="red", width=2
            )
            # segmentation masks
            masks = (pred["masks"] > cfg["mask_thr"]).squeeze(1)
            output_image = draw_segmentation_masks(output_image, masks, alpha=0.8)
            # save image
            plt.figure(figsize=(6, 6))
            plt.imshow(output_image.permute(1, 2, 0))
            plt.savefig(res_dir / time_str / f"{idx}.jpg")
            plt.close()


if __name__ == "__main__":
    path = Path("cfg/config.yaml")

    with open(path) as fd:
        try:
            cfg = yaml.safe_load(fd)
        except yaml.YAMLError as e:
            print(f"Failed to load the YAML file: {e}")
            exit(1)

    if cfg["visualize"]:
        visualize(cfg=cfg)
    else:
        eval(cfg=cfg)
