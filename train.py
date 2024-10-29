from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
import yaml

from src.engine import train_one_epoch, evaluate
from src.dataset import COCODataset, collate_fn
from src.models import get_mask_rcnn


def train(cfg: dict):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_dataset = COCODataset(
        image_path=Path(cfg["train"]),
        annotation_path=Path(cfg["train_annotation"]),
        filter=cfg["filter"],
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg["batch"],
        shuffle=True,
        num_workers=cfg["workers"],
        collate_fn=collate_fn,
    )

    val_dataset = COCODataset(
        image_path=Path(cfg["val"]),
        annotation_path=Path(cfg["val_annotation"]),
        filter=cfg["filter"],
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg["batch"],
        shuffle=True,
        num_workers=cfg["workers"],
        collate_fn=collate_fn,
    )

    model = get_mask_rcnn(num_classes=cfg["nc"])
    model.to(device=device)

    params = [p for p in model.parameters() if p.requires_grad]
    if cfg["optimizer"] == "SGD":
        optim = torch.optim.SGD(
            params=params,
            lr=cfg["lr"],
            momentum=cfg["momentum"],
            weight_decay=cfg["decay"],
        )
    else:
        optim = torch.optim.AdamW(
            params=params,
            lr=cfg["lr"],
            weight_decay=cfg["decay"],
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optim, mode="min", factor=0.5, patience=3, min_lr=1e-5,
    )

    epochs = cfg["epoch"]

    for e in range(epochs):
        train_one_epoch(
            model=model,
            optimizer=optim,
            data_loader=train_loader,
            device=device,
            epoch=e,
            print_freq=1,
        )

        scheduler.step()

        evaluate(model=model, data_loader=val_loader, device=device)

    print("Done!")


if __name__ == "__main__":
    path = Path(sys.argv[-1]) if len(sys.argv) > 1 else Path("cfg/config.yaml")
    with open("cfg/config.yaml") as fd:
        try:
            cfg = yaml.safe_load(fd)
        except yaml.YAMLError as e:
            print(f"Failed to load the YAML file: {e}")
            exit(1)

    train(cfg=cfg)